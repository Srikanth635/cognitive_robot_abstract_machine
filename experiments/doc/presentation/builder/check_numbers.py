"""
Checks that every number a slide states occurs in the record it cites.

Reads the outline (``builder/outline.js``) through node, collects the numbers in each
slide's points and notes, and looks each one up in the files its sources name. Prints
the numbers it cannot find, slide by slide, and exits non-zero when there are any.

python builder/check_numbers.py [idea_outline.js]
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# %% where the record lives
PRESENTATION = Path(__file__).resolve().parent.parent
LLM_REASONER_DOC = PRESENTATION.parent
REPOSITORY = LLM_REASONER_DOC.parent.parent

SOURCE_FILES = {
    "cram_viz README": REPOSITORY / "cram_viz" / "README.md",
    "llm_reasoner_viz README": REPOSITORY / "llm_reasoner_viz" / "README.md",
    "doc/index.rst": REPOSITORY / "doc" / "index.rst",
    "cram_viz sample hypothesis": REPOSITORY
    / "cram_viz/src/cram_viz/resources/sample_hypotheses/pour_milk_sample/hypothesis.ttl",
}

# numbers worth checking: anything with a digit, with its decimals and ranges kept
NUMBER = re.compile(
    r"\$?\d{1,3}(?:,\d{3})+(?:\.\d+)?|\$?\d+(?:\.\d+)?(?:\s?[–-]\s?\$?\d+(?:\.\d+)?)?"
)
# numbers that are not claims about the record
IGNORED = {
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "2026",
    "1.0",
    "1.3",
    "1.1",
    "1.2",
    "16",
}


@dataclass
class SlideCheck:
    """
    The numbers of one slide and which of them its sources do not contain.
    """

    slide_id: str
    """
    The slide's identifier in the outline.
    """

    missing: list[str] = field(default_factory=list)
    """
    The numbers none of the slide's source files contains.
    """


def source_path(source: str) -> Path:
    """
    The file a source reference names.
    """
    for name, file_path in SOURCE_FILES.items():
        if source.startswith(name):
            return file_path
    return LLM_REASONER_DOC / source.split(" ")[0]


def normalise(text: str) -> str:
    """
    One spelling for the dashes and spaces numbers are written with.
    """
    return text.replace("–", "-").replace("−", "-").replace(" ", "").replace(" ", "")


def numbers_in(text: str) -> list[str]:
    """
    The numbers stated in a text, as written.
    """
    return [m.group(0) for m in NUMBER.finditer(text)]


def check(slide: dict) -> SlideCheck:
    """
    Which numbers of a slide its sources do not contain.
    """
    result = SlideCheck(slide_id=slide["id"])
    if slide["kind"] != "content":
        return result
    corpus = normalise("\n".join(source_path(s).read_text() for s in slide["sources"]))
    stated = " ".join(
        slide["points"] + [slide["notes"], slide["title"], slide["subtitle"]]
    )
    for number in numbers_in(stated):
        plain = normalise(number)
        if plain.lstrip("$") in IGNORED:
            continue
        parts = [plain] if "-" not in plain.strip("-") else plain.split("-")
        if all(part.lstrip("$") and part.lstrip("$") in corpus for part in parts):
            continue
        if plain not in result.missing:
            result.missing.append(number)
    return result


def main() -> int:
    """
    Runs the check over every slide and prints what is missing.
    """
    outline_file = (
        PRESENTATION / "builder" / (sys.argv[1] if len(sys.argv) > 1 else "outline.js")
    )
    outline = json.loads(
        subprocess.run(
            [
                "node",
                "-e",
                "console.log(JSON.stringify(require(process.argv[1]).slides))",
                str(outline_file),
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    checks = [check(slide) for slide in outline]
    failed = [c for c in checks if c.missing]
    for c in failed:
        print(f"{c.slide_id}: not found in its sources: {', '.join(c.missing)}")
    print(f"{len(checks)} slides checked, {len(failed)} with numbers to look at")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
