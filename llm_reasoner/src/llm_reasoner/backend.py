"""
LLMBackend — a GenerativeBackend that uses an LLM as a reasoning engine.

This is the core of llm_reasoner. It follows exactly the same pattern as
ProbabilisticBackend in krrood/entity_query_language/backends.py but instead
of sampling from a probabilistic model, it asks an LLM to reason over the full
world state and produce concrete values for all free Match slots.

Variable assignment pattern (mirrors parameterizer.py and backends.py):
    mapped_var = expression._get_mapped_variable_by_name(field_name)
    mapped_var._value_ = resolved_value
    expression._update_kwargs_from_literal_values()
    yield expression.construct_instance()
"""
from __future__ import annotations

import typing
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple, TypeVar

from krrood.entity_query_language.backends import GenerativeBackend
from krrood.entity_query_language.query.match import Match

if typing.TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

T = TypeVar("T")


@dataclass
class LLMBackend(GenerativeBackend):
    """
    A GenerativeBackend that uses an LLM as a reasoning engine to fill
    underspecified Match expressions from natural-language instructions.

    Unlike ProbabilisticBackend (which samples from a probabilistic model),
    LLMBackend leverages the LLM's world knowledge and common-sense reasoning
    to infer concrete values for all free slots — including:

    - Entity grounding: "the milk on the table" → specific Body object in world
    - Parameter inference: arm, grasp type, approach direction from context
    - Constraint reasoning: physical feasibility, spatial relationships
    - Ambiguity resolution: when multiple candidates exist, pick the most salient

    The EQL / SymbolGraph layer is used only for final validation (does the
    resolved entity actually exist in the world?), not for the reasoning itself.

    Usage::

        context.query_backend = LLMBackend(
            instruction="pick up the milk from the table",
            llm=my_llm,
            world=context.world,
        )
        action = underspecified(PickUpAction)(object_designator=..., arm=..., grasp_description=...)
        plan = execute_single(action, context)
        plan.perform()
    """

    instruction: str
    """The natural-language instruction describing what the robot should do."""

    llm: "BaseChatModel"
    """
    A LangChain-compatible chat model used for reasoning.
    Inject via make_llm() or pass any BaseChatModel directly.
    No global singletons — the caller controls the model.
    """

    world: Any
    """
    The world object (SDT WorldLike) used to:
    - Serialize world state for LLM context
    - Validate that LLM-resolved entity names exist in the world
    """

    def _evaluate(self, expression: Match[T]) -> Iterable[T]:
        """
        Core evaluation: ask the LLM to fill all free slots in the Match,
        then assign the resolved values and yield a constructed instance.

        Mirrors the pattern from ProbabilisticBackend._evaluate() and
        EntityQueryLanguageBackend._generate_raw_results(), but instead of
        sampling from a model or enumerating a variable domain, we call the LLM.
        """
        from llm_reasoner.world_serializer import serialize_world
        from llm_reasoner.workflows.slot_filler import run_slot_filler

        # ------------------------------------------------------------------ #
        # Step 1: Identify free slots (Ellipsis) and already-fixed slots      #
        # This mirrors assignments_for_conditioning in parameterizer.py       #
        # ------------------------------------------------------------------ #
        free_slots: List[Tuple[str, Any]] = []   # [(field_name, field_type), ...]
        fixed_slots: Dict[str, Any] = {}

        for attr_match in expression.matches_with_variables:
            field_name = attr_match.name_from_variable_access_path
            value = attr_match.assigned_variable._value_

            if isinstance(value, type(Ellipsis)):
                field_type = attr_match.assigned_variable._type_
                free_slots.append((field_name, field_type))
            else:
                fixed_slots[field_name] = value

        if not free_slots:
            # Nothing to fill — construct directly from fixed values
            expression._update_kwargs_from_literal_values()
            yield expression.construct_instance()
            return

        # ------------------------------------------------------------------ #
        # Step 2: Serialize world state for LLM context                       #
        # ------------------------------------------------------------------ #
        world_context = serialize_world(self.world)

        # ------------------------------------------------------------------ #
        # Step 3: LLM reasoning — fill all free slots                         #
        # The LLM receives: instruction, action type, free slot names+types,  #
        # fixed slot values, and full world state. It reasons and returns      #
        # concrete values for every free slot.                                 #
        # ------------------------------------------------------------------ #
        resolved = run_slot_filler(
            instruction=self.instruction,
            action_type=expression.type.__name__,
            free_slots=free_slots,
            fixed_slots=fixed_slots,
            world_context=world_context,
            llm=self.llm,
        )

        if resolved is None:
            # LLM failed to produce a valid output — yield nothing
            return

        # ------------------------------------------------------------------ #
        # Step 4: Write resolved values back into the Match variable graph    #
        # Mirrors _generate_raw_results() in EntityQueryLanguageBackend and   #
        # create_instance_from_variables_and_sample() in parameterizer.py     #
        # ------------------------------------------------------------------ #
        for field_name, value in resolved.items():
            mapped_var = expression._get_mapped_variable_by_name(field_name)
            if mapped_var is not None:
                mapped_var._value_ = value

        expression._update_kwargs_from_literal_values()
        yield expression.construct_instance()
