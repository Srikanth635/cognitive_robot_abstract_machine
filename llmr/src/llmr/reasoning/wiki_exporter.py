"""Wiki exporter reasoner — generates connected markdown wiki from llmr pipeline runs.

Place this last in the ``reasoners`` list so all other reasoners have already
populated ``semantics`` before this writes to disk.

Usage::

    from llmr.reasoning.wiki_exporter import WikiExporterReasoner

    backend = LLMBackend(
        llm=llm,
        instruction="pick up the milk from the table",
        symbol_type=WorldBody,
        reasoners=[
            FrameNetReasoner(llm=llm),
            FlanaganReasoner(llm=llm),
            WikiExporterReasoner(wiki_dir="~/LLM_WIKIs/LLMRWiki"),
        ],
    )
    action = next(iter(backend.evaluate(match)))
    # wiki pages auto-generated at ~/LLM_WIKIs/LLMRWiki/
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from llmr.reasoning import Reasoner

if TYPE_CHECKING:
    from llmr.bridge.match_reader import MatchSnapshot as MatchData
    from llmr.schemas import ActionAnnotationBundle as ActionSemantics

logger = logging.getLogger(__name__)

_SLUG_MAX_LEN = 60
_SLUG_RE = re.compile(r"[^a-z0-9]+")
_TABLE_SEP_RE = re.compile(r"^\|[-| ]+\|$")

_CLAUDE_MD = """\
# LLMRWiki — LLM Representation Wiki

## Purpose

Auto-generated knowledge base for llmr pipeline runs. Each run of `LLMBackend`
generates connected markdown pages capturing the LLM representations produced:
FrameNet semantic annotations, Flanagan motion plans, slot filling outputs, and
the executable PyCRAM action.

## Structure

- `runs/` — one page per run; shows all representations for that instruction
- `framenet/` — one page per FrameNet frame; accumulates runs that evoke it
- `flanagan/` — one page per action type; accumulates motion plans for it
- `index.md` — chronological run index (most recent first)

## Rules

- Files in `runs/`, `framenet/`, and `flanagan/` are auto-generated — do not edit manually
- `index.md` is auto-maintained
- Open this directory as an Obsidian vault to see the connection graph
- `[[wiki-links]]` connect run pages to representation pages
"""

_INDEX_HEADER = """\
# LLMRWiki — Run Index

| Date | Instruction | Action | FrameNet Frame | Flanagan Phases |
|------|-------------|--------|----------------|-----------------|
"""


class WikiExporterReasoner(Reasoner):
    """Write connected markdown wiki pages from a completed LLMBackend run.

    Does not call any LLM — reads the already-populated ``semantics`` bundle
    and writes markdown files to ``wiki_dir``. Failures are logged and do not
    interrupt backend execution.

    :param wiki_dir: Root directory for the wiki. Created on first use.
    """

    REASONER_NAME = "wiki_exporter"
    PROMPT_VERSION = "n/a"

    def __init__(self, wiki_dir: str | Path) -> None:
        self._wiki_dir = Path(wiki_dir).expanduser().resolve()

    def annotate(
        self,
        semantics: "ActionSemantics",
        match_data: "MatchData",
        world_context: str,
    ) -> None:
        try:
            self._ensure_wiki_structure()
            now = datetime.now()
            run_slug = self._make_run_slug(semantics.instruction, now)
            self._write_run_page(run_slug, semantics, now)
            if semantics.frames is not None:
                self._update_framenet_page(run_slug, semantics, now)
            if semantics.motion_phases is not None:
                self._update_flanagan_page(run_slug, semantics, now)
            self._update_index(run_slug, semantics, now)
            logger.debug("WikiExporterReasoner: wrote run page '%s'", run_slug)
        except Exception as exc:
            logger.warning("WikiExporterReasoner: failed to write wiki — %s", exc)

    # ── wiki structure ────────────────────────────────────────────────────────

    def _ensure_wiki_structure(self) -> None:
        for subdir in ("runs", "framenet", "flanagan"):
            (self._wiki_dir / subdir).mkdir(parents=True, exist_ok=True)
        claude_md = self._wiki_dir / "CLAUDE.md"
        if not claude_md.exists():
            claude_md.write_text(_CLAUDE_MD, encoding="utf-8")
        index_md = self._wiki_dir / "index.md"
        if not index_md.exists():
            index_md.write_text(_INDEX_HEADER, encoding="utf-8")

    # ── run page (written fresh each run) ────────────────────────────────────

    def _write_run_page(
        self,
        run_slug: str,
        semantics: "ActionSemantics",
        now: datetime,
    ) -> None:
        instruction = semantics.instruction or "(no instruction)"
        action_type = semantics.action_type
        date_str = now.strftime("%Y-%m-%d %H:%M:%S")

        lines: list[str] = [
            "---",
            f"instruction: {instruction}",
            f"action_type: {action_type}",
            f"run_date: {date_str}",
            "---",
            "",
            f"# Run: {instruction}",
            "",
            f"**Action**: `{action_type}`  ",
            f"**Date**: {date_str}",
            "",
        ]

        if semantics.slot_filling is not None:
            lines += self._render_slot_filling(semantics.slot_filling)

        if semantics.frames is not None:
            lines += self._render_framenet(semantics.frames)

        if semantics.motion_phases is not None:
            lines += self._render_flanagan(semantics.motion_phases, action_type)

        (self._wiki_dir / "runs" / f"{run_slug}.md").write_text(
            "\n".join(lines), encoding="utf-8"
        )

    def _render_slot_filling(self, slot_filling: object) -> list[str]:
        lines = ["## Slot Filling", ""]
        slots = list(getattr(slot_filling, "slots", []))
        if slots:
            lines += ["| Slot | Value | Reasoning |", "|------|-------|-----------|"]
            for slot in slots:
                val = getattr(slot, "value", None) or ""
                ed = getattr(slot, "entity_description", None)
                if not val and ed is not None:
                    val = getattr(ed, "name", "")
                reasoning = (getattr(slot, "reasoning", "") or "").replace("|", "\\|").replace("\n", " ")
                lines.append(f"| `{getattr(slot, 'field_name', '')}` | {val} | {reasoning} |")
        overall = getattr(slot_filling, "overall_reasoning", "") or ""
        if overall:
            lines += ["", f"_{overall}_"]
        lines.append("")
        return lines

    def _render_framenet(self, frames: object) -> list[str]:
        frame_name = getattr(frames, "frame", "Unknown")
        lines = [
            f"## FrameNet → [[framenet/{frame_name}]]",
            "",
            f"- **Frame**: {frame_name}",
            f"- **Lexical Unit**: {getattr(frames, 'lexical_unit', '')}",
            f"- **Framenet label**: {getattr(frames, 'framenet', '')}",
            "",
            "### Core Elements",
            "",
            "| Role | Filler |",
            "|------|--------|",
        ]
        core = getattr(frames, "core", None)
        if core is not None:
            for role_name, filler in self._iter_model_fields(core):
                if filler and str(filler).lower() not in ("null", "none", ""):
                    lines.append(f"| {role_name} | {filler} |")

        lines += ["", "### Peripheral Elements", "", "| Role | Filler |", "|------|--------|"]
        peripheral = getattr(frames, "peripheral", None)
        if peripheral is not None:
            for role_name, filler in self._iter_model_fields(peripheral):
                if filler and str(filler).lower() not in ("null", "none", ""):
                    lines.append(f"| {role_name} | {filler} |")
        lines.append("")
        return lines

    def _render_flanagan(self, motion_phases: object, action_type: str) -> list[str]:
        phases = list(getattr(motion_phases, "phases", []))
        lines = [
            f"## Flanagan Motion Plan → [[flanagan/{action_type}]]",
            "",
            f"**Phases**: {len(phases)}",
            "",
            "| # | Phase | Object | Description | Duration (s) |",
            "|---|-------|--------|-------------|--------------|",
        ]
        for i, phase in enumerate(phases):
            phase_name = getattr(phase, "phase", "")
            obj = getattr(phase, "target_object", "")
            desc = (getattr(phase, "description", "") or "").replace("|", "\\|").replace("\n", " ")
            tc = getattr(phase, "temporal_constraints", {}) or {}
            dur = tc.get("max_duration_sec", "-") if isinstance(tc, dict) else "-"
            lines.append(f"| {i} | {phase_name} | {obj} | {desc} | {dur} |")
        lines.append("")
        return lines

    # ── framenet page (accumulates across runs) ───────────────────────────────

    def _update_framenet_page(
        self,
        run_slug: str,
        semantics: "ActionSemantics",
        now: datetime,
    ) -> None:
        frame_name = getattr(semantics.frames, "frame", "Unknown")
        path = self._wiki_dir / "framenet" / f"{frame_name}.md"
        date_str = now.strftime("%Y-%m-%d")
        instruction = (semantics.instruction or "").replace("|", "\\|")
        new_row = f"| {date_str} | {instruction} | [[runs/{run_slug}]] |"

        if not path.exists():
            content = "\n".join([
                f"# Frame: {frame_name}",
                "",
                "FrameNet frame evoked in llmr pipeline runs.",
                "",
                "## Runs",
                "",
                "| Date | Instruction | Run page |",
                "|------|-------------|----------|",
                new_row,
                "",
            ])
        else:
            content = self._insert_table_row(path.read_text(encoding="utf-8"), new_row)

        path.write_text(content, encoding="utf-8")

    # ── flanagan page (accumulates across runs) ───────────────────────────────

    def _update_flanagan_page(
        self,
        run_slug: str,
        semantics: "ActionSemantics",
        now: datetime,
    ) -> None:
        action_type = semantics.action_type
        path = self._wiki_dir / "flanagan" / f"{action_type}.md"
        date_str = now.strftime("%Y-%m-%d")
        instruction = (semantics.instruction or "").replace("|", "\\|")
        phases = list(getattr(semantics.motion_phases, "phases", []))
        phase_names = "/".join(getattr(p, "phase", "?") for p in phases)
        new_row = f"| {date_str} | {instruction} | {phase_names} | [[runs/{run_slug}]] |"

        if not path.exists():
            content = "\n".join([
                f"# Action Type: {action_type} — Motion Plans",
                "",
                "## Runs",
                "",
                "| Date | Instruction | Phases | Run page |",
                "|------|-------------|--------|----------|",
                new_row,
                "",
            ])
        else:
            content = self._insert_table_row(path.read_text(encoding="utf-8"), new_row)

        path.write_text(content, encoding="utf-8")

    # ── index ─────────────────────────────────────────────────────────────────

    def _update_index(
        self,
        run_slug: str,
        semantics: "ActionSemantics",
        now: datetime,
    ) -> None:
        date_str = now.strftime("%Y-%m-%d")
        instruction = (semantics.instruction or "").replace("|", "\\|")
        action_type = semantics.action_type
        frame = getattr(semantics.frames, "frame", "-") if semantics.frames else "-"
        phases = (
            str(len(list(getattr(semantics.motion_phases, "phases", []))))
            if semantics.motion_phases
            else "-"
        )
        new_row = f"| {date_str} | {instruction} | `{action_type}` | {frame} | {phases} |"

        index_path = self._wiki_dir / "index.md"
        content = self._insert_table_row(index_path.read_text(encoding="utf-8"), new_row)
        index_path.write_text(content, encoding="utf-8")

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _make_run_slug(instruction: str | None, now: datetime) -> str:
        text = (instruction or "run").lower()
        slug = _SLUG_RE.sub("-", text).strip("-")[:_SLUG_MAX_LEN].rstrip("-")
        return f"{slug}-{now.strftime('%Y%m%d-%H%M%S')}"

    @staticmethod
    def _iter_model_fields(model_obj: object) -> list[tuple[str, str]]:
        model_dump = getattr(model_obj, "model_dump", None)
        if callable(model_dump):
            return list(model_dump().items())
        return [(k, v) for k, v in vars(model_obj).items() if not k.startswith("_")]

    @staticmethod
    def _insert_table_row(content: str, new_row: str) -> str:
        """Insert *new_row* after the last markdown table separator line."""
        lines = content.splitlines()
        insert_at = -1
        for i, line in enumerate(lines):
            if _TABLE_SEP_RE.match(line.strip()):
                insert_at = i
        if insert_at >= 0:
            lines.insert(insert_at + 1, new_row)
        else:
            lines.append(new_row)
        return "\n".join(lines) + "\n"
