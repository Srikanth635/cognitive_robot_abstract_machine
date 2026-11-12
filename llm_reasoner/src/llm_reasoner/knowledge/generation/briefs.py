"""
What a generation asks the model to describe.
"""

from __future__ import annotations

from dataclasses import dataclass

from llm_reasoner.prompt_structure import PromptRule


@dataclass(frozen=True)
class GenerationBrief:
    """
    What the model is asked to describe, and the rules about what its answer describes.
    """

    role: str
    """
    What the model is asked to do.
    """

    rules: tuple[PromptRule, ...]
    """
    The rules about what the answer describes, stated before the rules about its form.
    """
