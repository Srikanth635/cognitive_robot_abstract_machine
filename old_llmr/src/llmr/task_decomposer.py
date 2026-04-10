
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing_extensions import Dict, List

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from llmr.workflows.llm_configuration import default_llm

logger = logging.getLogger(__name__)


# ── Public result type ──────────────────────────────────────────────────────────


@dataclass
class DecomposedPlan:
    """Result of decomposing a (possibly compound) NL instruction into atomic steps."""

    steps: List[str]
    dependencies: Dict[int, List[int]] = field(default_factory=dict)


# ── LLM output schema ───────────────────────────────────────────────────────────


class _AtomicStep(BaseModel):
    """One atomic robot action with its prerequisite step indices."""

    instruction: str = Field(description="Self-contained atomic sub-instruction text.")
    dependencies: List[int] = Field(
        default_factory=list,
        description=(
            "0-based indices of steps that must succeed before this one. "
            "Use object flow only — empty list if no shared objects."
        ),
    )


class _DecomposedInstructions(BaseModel):
    """Structured output for the task decomposer."""

    steps: List[_AtomicStep] = Field(
        description="One step per action verb; single-element list if already atomic."
    )


# ── Prompt ─────────────────────────────────────────────────────────────────────

_DECOMPOSER_SYSTEM = """\
You are a robot task decomposer.

Your job is to split a natural language instruction into a list of ATOMIC steps,
where each step corresponds to EXACTLY ONE of the following supported action types:

  - PickUpAction  (triggered by: pick up, grab, grasp, lift, take, fetch, get, retrieve)
  - PlaceAction   (triggered by: place, put, set down, deposit, lay, put down, drop off)

════════════════════════════════════════════════════
STEP COUNT RULE  (most important — check this last)
════════════════════════════════════════════════════
Count the distinct action verbs in the instruction.
The number of steps you return MUST equal that count.
  • 1 action verb  → return exactly 1 step
  • 2 action verbs → return exactly 2 steps
  • …and so on.

Never add implicit steps. Never split one verb into two steps.
Never merge two verbs into one step.

════════════════════════════════════════════════════
DEPENDENCY RULES
════════════════════════════════════════════════════
- Dependencies are based on OBJECT FLOW, not instruction order.
- A PlaceAction depends on the PickUpAction that acquires the same object.
- A PickUpAction on a DIFFERENT object is independent (empty dependencies).
- Do NOT add a dependency just because two steps are adjacent.

════════════════════════════════════════════════════
DECOMPOSITION RULES
════════════════════════════════════════════════════
- Preserve all object names, target names, and qualifiers VERBATIM.
- Steps must be in execution order (pick up before placing).
- Replace pronouns ("it", "the object") with the actual object name so each
  step is fully self-contained.
- Do NOT invent objects, targets, or actions not in the instruction.
- Unsupported actions: include verbatim as a single step, empty dependencies.
- Never repeat the same step — every step must be unique.
- Never invent implicit prerequisite steps.  "Place the bottle on the shelf"
  → return ONLY that one step; do NOT prepend a pick-up step.

════════════════════════════════════════════════════
SOURCE vs TARGET  (critical for pick-up instructions)
════════════════════════════════════════════════════
- "from", "off", "out of", "off of"  → SOURCE location for PickUpAction.
  These do NOT imply a PlaceAction.
- "on", "onto", "into", "at", "to"  → TARGET location for PlaceAction.

"Pick up the milk from the counter"
  → 1 verb ("pick up") → exactly 1 step: PickUpAction.
  "from the counter" is source context, NOT a place target.

════════════════════════════════════════════════════
EXAMPLES
════════════════════════════════════════════════════
"grab the bottle and put it on the shelf"        ← 2 verbs
  → 0: instruction="grab the bottle",              dependencies=[]
     1: instruction="put the bottle on the shelf", dependencies=[0]

"fetch the cereal box from the cabinet"          ← 1 verb
  → 0: instruction="fetch the cereal box from the cabinet", dependencies=[]

"pick up the book from the table and place it on the bookshelf"  ← 2 verbs
  → 0: instruction="pick up the book from the table",  dependencies=[]
     1: instruction="place the book on the bookshelf", dependencies=[0]

"grab the cup. grab the plate. put the cup on the tray"  ← 3 verbs
  → 0: instruction="grab the cup",            dependencies=[]
     1: instruction="grab the plate",          dependencies=[]
     2: instruction="put the cup on the tray", dependencies=[0]

"retrieve the bottle and place it on the dining table and pick up the box"  ← 3 verbs
  → 0: instruction="retrieve the bottle",                  dependencies=[]
     1: instruction="place the bottle on the dining table", dependencies=[0]
     2: instruction="pick up the box",                      dependencies=[]

"take the mug off the shelf"                     ← 1 verb
  → 0: instruction="take the mug off the shelf", dependencies=[]

════════════════════════════════════════════════════
SELF-CHECK before returning
════════════════════════════════════════════════════
1. Count action verbs in the original instruction.
2. Count steps you are about to return.
3. If the counts differ, fix your answer before returning.
"""

_DECOMPOSER_HUMAN = """\
Instruction: {instruction}

Decompose into atomic steps with dependencies.
"""

_decomposer_prompt = ChatPromptTemplate.from_messages(
    [("system", _DECOMPOSER_SYSTEM), ("human", _DECOMPOSER_HUMAN)]
)


# ── TaskDecomposer ─────────────────────────────────────────────────────────────


class TaskDecomposer:
    """Splits compound NL instructions into atomic steps with object-flow dependencies."""

    def __init__(self) -> None:
        llm = default_llm.with_structured_output(_DecomposedInstructions, method="function_calling")
        self._chain = _decomposer_prompt | llm

    # ── Public API ──────────────────────────────────────────────────────────────

    def decompose(self, instruction: str) -> DecomposedPlan:
        """Return a :class:`DecomposedPlan` for *instruction*, falling back to the original on error."""
        try:
            result: _DecomposedInstructions = self._chain.invoke({"instruction": instruction})
            dedup_steps = self._dedup_steps(result.steps, instruction)

            if not dedup_steps:
                logger.warning(
                    "Decomposer returned empty list for '%s' – using original.", instruction
                )
                return DecomposedPlan(steps=[instruction])

            steps = [s.instruction.strip() for s in dedup_steps]
            dependencies = self._build_dependencies(dedup_steps)

            logger.info(
                "Decomposed '%s' → steps=%s deps=%s", instruction, steps, dependencies
            )
            return DecomposedPlan(steps=steps, dependencies=dependencies)

        except Exception as exc:
            logger.error(
                "Decomposer failed for '%s': %s – falling back to original.", instruction, exc
            )
            return DecomposedPlan(steps=[instruction])

    # ── Private helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _dedup_steps(
        result_steps: List[_AtomicStep],
        instruction: str,
    ) -> List[_AtomicStep]:
        """Keep first occurrence of each unique step; warn and drop duplicates."""
        seen: Dict[str, int] = {}
        dedup: List[_AtomicStep] = []
        for step_obj in result_steps:
            text = step_obj.instruction.strip()
            if not text:
                continue
            if text not in seen:
                seen[text] = len(dedup)
                dedup.append(step_obj)
            else:
                logger.warning(
                    "Decomposer produced duplicate step '%s' for '%s' — removing.",
                    text,
                    instruction,
                )
        return dedup

    @staticmethod
    def _build_dependencies(
        dedup_steps: List[_AtomicStep],
    ) -> Dict[int, List[int]]:
        """Build dependency map; clamp out-of-range indices and drop self-references."""
        n = len(dedup_steps)
        dependencies: Dict[int, List[int]] = {}
        for i, step in enumerate(dedup_steps):
            valid_deps = [d for d in step.dependencies if 0 <= d < n and d != i]
            if valid_deps:
                dependencies[i] = valid_deps
        return dependencies
