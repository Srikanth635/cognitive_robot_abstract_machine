"""Task decomposer — splits compound NL instructions into atomic steps.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing_extensions import Dict, List, Optional

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from krrood.llmr_decoupled.workflows.llm_configuration import default_llm

logger = logging.getLogger(__name__)


# ── Public result type ──────────────────────────────────────────────────────────


@dataclass
class DecomposedPlan:
    """Result of decomposing a (possibly compound) NL instruction into atomic steps."""

    steps: List[str]
    dependencies: Dict[int, List[int]] = field(default_factory=dict)


# ── LLM output schema ───────────────────────────────────────────────────────────


class _AtomicStep(BaseModel):
    instruction: str = Field(description="Self-contained atomic sub-instruction text.")
    dependencies: List[int] = Field(
        default_factory=list,
        description=(
            "0-based indices of steps that must succeed before this one. "
            "Use object flow only — empty list if no shared objects."
        ),
    )


class _DecomposedInstructions(BaseModel):
    steps: List[_AtomicStep] = Field(
        description="One step per action verb; single-element list if already atomic."
    )


# ── Prompt builder ──────────────────────────────────────────────────────────────


def _build_decomposer_system(action_type_descriptions: Dict[str, str]) -> str:
    """Build the decomposer system prompt with injected action types."""
    if action_type_descriptions:
        action_block = "\n".join(
            f"  - {name}  ({desc})" for name, desc in action_type_descriptions.items()
        )
    else:
        action_block = "  (No specific action types registered — decompose by action verb count.)"

    return f"""\
You are a task decomposer.

Your job is to split a natural language instruction into a list of ATOMIC steps,
where each step corresponds to EXACTLY ONE of the following supported action types:

{action_block}

════════════════════════════════════════════════════
STEP COUNT RULE  (most important)
════════════════════════════════════════════════════
Count the distinct action verbs EXPLICITLY PRESENT in the instruction.
The number of steps you return MUST equal that count — no more, no less.
  • 1 action verb  → return exactly 1 step
  • 2 action verbs → return exactly 2 steps
  • …and so on.

CRITICAL — DO NOT ADD IMPLICIT STEPS:
  ✗ WRONG: "pick up the cup" → [pick up the cup, place the cup somewhere]
  ✓ RIGHT: "pick up the cup" → [pick up the cup]

  ✗ WRONG: "fetch the milk" → [pick up the milk, bring the milk to the table]
  ✓ RIGHT: "fetch the milk" → [fetch the milk]

If a step is not EXPLICITLY requested by the user, do NOT add it.
"Pick up" does NOT imply a subsequent "place". Only add a step if its verb
appears word-for-word in the instruction.

════════════════════════════════════════════════════
DEPENDENCY RULES
════════════════════════════════════════════════════
- Dependencies are based on OBJECT FLOW, not instruction order.
- A secondary action on the same object depends on the primary action.
- An action on a DIFFERENT object is independent (empty dependencies).
- Do NOT add a dependency just because two steps are adjacent.

════════════════════════════════════════════════════
DECOMPOSITION RULES
════════════════════════════════════════════════════
- Preserve all object names, target names, and qualifiers VERBATIM.
- Steps must be in execution order.
- Replace pronouns ("it", "the object") with the actual object name so each
  step is fully self-contained.
- Do NOT invent objects, targets, or actions not in the instruction.
- Unsupported actions: include verbatim as a single step, empty dependencies.
- Never repeat the same step — every step must be unique.

════════════════════════════════════════════════════
SELF-CHECK before returning
════════════════════════════════════════════════════
1. List every action verb explicitly written in the instruction.
2. Count those verbs — that is your required step count.
3. Count the steps you are about to return.
4. If the counts differ, REMOVE any step whose verb is not in the instruction.
"""


_DECOMPOSER_HUMAN = """\
Instruction: {instruction}

Decompose into atomic steps with dependencies.
"""


# ── TaskDecomposer ─────────────────────────────────────────────────────────────


class TaskDecomposer:
    """Splits compound NL instructions into atomic steps with object-flow dependencies.

    :param action_type_descriptions: ``{action_type_str: trigger_verb_description}``
    """

    def __init__(self, action_type_descriptions: Optional[Dict[str, str]] = None) -> None:
        system = _build_decomposer_system(action_type_descriptions or {})
        prompt = ChatPromptTemplate.from_messages(
            [("system", system), ("human", _DECOMPOSER_HUMAN)]
        )
        llm = default_llm.with_structured_output(_DecomposedInstructions, method="function_calling")
        self._chain = prompt | llm

    def decompose(self, instruction: str) -> DecomposedPlan:
        """Return a :class:`DecomposedPlan` for *instruction*, falling back to original on error."""
        try:
            result: _DecomposedInstructions = self._chain.invoke({"instruction": instruction})
            dedup_steps = self._dedup_steps(result.steps, instruction)

            if not dedup_steps:
                logger.warning("Decomposer returned empty list for '%s' – using original.", instruction)
                return DecomposedPlan(steps=[instruction])

            steps = [s.instruction.strip() for s in dedup_steps]
            dependencies = self._build_dependencies(dedup_steps)
            logger.info("Decomposed '%s' → steps=%s deps=%s", instruction, steps, dependencies)
            return DecomposedPlan(steps=steps, dependencies=dependencies)

        except Exception as exc:
            logger.error("Decomposer failed for '%s': %s – falling back to original.", instruction, exc)
            return DecomposedPlan(steps=[instruction])

    @staticmethod
    def _dedup_steps(result_steps: List[_AtomicStep], instruction: str) -> List[_AtomicStep]:
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
                logger.warning("Decomposer produced duplicate step '%s' for '%s' — removing.", text, instruction)
        return dedup

    @staticmethod
    def _build_dependencies(dedup_steps: List[_AtomicStep]) -> Dict[int, List[int]]:
        n = len(dedup_steps)
        dependencies: Dict[int, List[int]] = {}
        for i, step in enumerate(dedup_steps):
            valid_deps = [d for d in step.dependencies if 0 <= d < n and d != i]
            if valid_deps:
                dependencies[i] = valid_deps
        return dependencies
