"""
Reading the tagged sections of a prompt back, so that tests assert on what a prompt
holds rather than on its wording.
"""

from __future__ import annotations

import json
import re

from typing_extensions import Any, Dict, List

from llm_reasoner.resolver.prompt_format import PromptSection, SlotKey

TAGGED_SECTION = re.compile(
    r"<(?P<tag>\w+)(?P<attributes>[^>\n]*)>\n(?P<body>.*?)\n</(?P=tag)>", re.DOTALL
)
"""
One tagged section of a prompt: its tag, the attributes of the tag and its body.
"""


def bodies(prompt: str, section: PromptSection) -> List[str]:
    """
    :return: The bodies of every section of a prompt with the given tag, in order.
    """
    return [
        found["body"]
        for found in TAGGED_SECTION.finditer(prompt)
        if found["tag"] == section
    ]


def text_outside_sections(prompt: str) -> str:
    """
    :return: Whatever a prompt holds outside its tagged sections, blank space dropped.
    """
    return TAGGED_SECTION.sub("", prompt).strip()


def slot_entry(prompt: str, path: str) -> Dict[str, Any]:
    """
    :return: The entry of the slot with the given path in a prompt's slots section.
    """
    (slots,) = bodies(prompt, PromptSection.SLOTS)
    (entry,) = [entry for entry in json.loads(slots) if entry[SlotKey.PATH] == path]
    return entry
