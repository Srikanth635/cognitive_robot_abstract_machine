"""
The structure every prompt to a language model is written in: tagged sections, each
holding text or data, and rules each stated once under its own name.

Each component names its own sections and rules; this module only writes them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from typing_extensions import Dict, List

SECTION_SEPARATOR = "\n\n"
"""
What separates two sections of a prompt.
"""


@dataclass(frozen=True)
class PromptBlock:
    """
    One tagged section of a prompt.
    """

    section: StrEnum
    """
    The section, which names the tag.
    """

    body: str
    """
    What the section holds.
    """

    attributes: Dict[StrEnum, str] = field(default_factory=dict)
    """
    Attributes of the tag, such as the type a source belongs to.
    """

    def render(self) -> str:
        """
        :return: The section as its tag enclosing its body.
        """
        attributes = "".join(
            f' {name}="{value}"' for name, value in self.attributes.items()
        )
        return f"<{self.section}{attributes}>\n{self.body}\n</{self.section}>"


def render_blocks(blocks: List[PromptBlock]) -> str:
    """
    :return: The blocks as one prompt, in order.
    """
    return SECTION_SEPARATOR.join(block.render() for block in blocks)


@dataclass(frozen=True)
class PromptRule:
    """
    One rule a model answers by.
    """

    name: StrEnum
    """
    What the rule governs.
    """

    text: str
    """
    The rule, as one instruction.
    """

    def render(self) -> str:
        """
        :return: The rule as one line of the rules section.
        """
        return f"- {self.name}: {self.text}"
