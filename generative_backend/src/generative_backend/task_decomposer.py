"""Task decomposer: splits compound NL instructions into atomic sub-instructions.

The slot-filler and action dispatcher are designed to handle ONE action at a time.
Instructions like "fetch the milk and put it on the counter" contain two actions and
must be decomposed before reaching the slot-filler.

This module provides a lightweight LLM pass that detects compound instructions and
returns them as an ordered list of atomic sub-instructions.  Atomic instructions
are returned unchanged (as a single-element list).

Usage::

    from generative_backend.task_decomposer import TaskDecomposer

    decomposer = TaskDecomposer()
    decomposer.decompose("fetch the milk and place it on the island_countertop")
    # → ["Pick up the milk", "Place the milk on the island_countertop"]

    decomposer.decompose("pick up the milk")
    # → ["pick up the milk"]
"""

from __future__ import annotations

import logging
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from .workflows.llm_configuration import default_llm

logger = logging.getLogger(__name__)


# ── Output schema ───────────────────────────────────────────────────────────────


class _DecomposedInstructions(BaseModel):
    """LLM output schema for the decomposer."""

    instructions: List[str] = Field(
        description=(
            "Ordered list of atomic sub-instructions. "
            "Each entry must map to exactly one supported action type "
            "(PickUpAction or PlaceAction). "
            "If the original instruction is already atomic, return it as a single-element list."
        )
    )


# ── Prompt ─────────────────────────────────────────────────────────────────────

_DECOMPOSER_SYSTEM = """\
You are a robot task decomposer.

Your job is to split a natural language instruction into a list of ATOMIC
sub-instructions, where each sub-instruction corresponds to EXACTLY ONE of the
following supported action types:

  - PickUpAction  (triggered by: pick up, grab, grasp, lift, take, fetch, get, retrieve)
  - PlaceAction   (triggered by: place, put, set down, deposit, lay, put down, drop off)

Rules:
- If the instruction is already atomic (single action), return it as-is in a
  single-element list. Do NOT rephrase it.
- If the instruction is compound (multiple actions joined by "and", "then",
  "after that", etc.), split it into one entry per action.
- Preserve all object names, target names, and qualifiers VERBATIM from the original.
- Sub-instructions must be in execution order (e.g. pick up before placing).
- Replace pronouns ("it", "the object") in later steps with the actual object name
  from the earlier step, so each sub-instruction is self-contained.
- Do NOT invent new objects, targets, or actions not mentioned in the instruction.
- Do NOT decompose beyond the supported action types — if the instruction contains
  an unsupported action, include it verbatim as a single entry.

Examples:
  "fetch the milk and put it on the counter"
    → ["Pick up the milk", "Place the milk on the counter"]

  "grab the cereal box from the shelf and place it on the table"
    → ["Pick up the cereal box from the shelf", "Place the cereal box on the table"]

  "pick up the mug"
    → ["pick up the mug"]

  "place the bottle on the island_countertop"
    → ["place the bottle on the island_countertop"]
"""

_DECOMPOSER_HUMAN = """\
Instruction: {instruction}

Decompose into atomic sub-instructions.
"""

_decomposer_prompt = ChatPromptTemplate.from_messages(
    [("system", _DECOMPOSER_SYSTEM), ("human", _DECOMPOSER_HUMAN)]
)


# ── TaskDecomposer ─────────────────────────────────────────────────────────────


class TaskDecomposer:
    """Splits compound NL instructions into atomic sub-instructions.

    Each returned sub-instruction maps to exactly one supported action type
    (PickUpAction or PlaceAction) and can be fed directly into ``ActionPipeline``.
    """

    def __init__(self) -> None:
        self._llm = default_llm.with_structured_output(
            _DecomposedInstructions, method="function_calling"
        )
        self._chain = _decomposer_prompt | self._llm

    def decompose(self, instruction: str) -> List[str]:
        """Decompose *instruction* into an ordered list of atomic sub-instructions.

        :param instruction: Raw natural language instruction (may be compound).
        :return: Ordered list of atomic sub-instructions.  Always has at least one entry.
        """
        try:
            result: _DecomposedInstructions = self._chain.invoke(
                {"instruction": instruction}
            )
            sub_instructions = [s.strip() for s in result.instructions if s.strip()]
            if not sub_instructions:
                logger.warning(
                    "Decomposer returned empty list for '%s' – using original.", instruction
                )
                return [instruction]
            logger.info(
                "Decomposed '%s' → %s", instruction, sub_instructions
            )
            return sub_instructions
        except Exception as exc:
            logger.error(
                "Decomposer failed for '%s': %s – falling back to original.", instruction, exc
            )
            return [instruction]
