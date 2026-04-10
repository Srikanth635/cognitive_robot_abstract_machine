"""
llm_reasoner — LLM-powered GenerativeBackend for KRROOD.

A standalone package that implements LLMBackend(GenerativeBackend), allowing
PyCRAM underspecified action Match expressions to be resolved via LLM reasoning.

Package-independence guarantees
---------------------------------
- No SDT (semantic_digital_twin) imports anywhere in this package.
- No hard pycram imports outside factory.py (integration layer).
- World context is derived from SymbolGraph (krrood), not from a world object.
- Robot components (Manipulator, Camera) are injected via the `context` dict.

Dependency direction: llm_reasoner → krrood (one-way, no circular imports).

Package layout
--------------
  backend.py              LLMBackend — the GenerativeBackend implementation
  factory.py              nl_plan() / nl_sequential() — user-facing entry points
  schemas/
    entities.py           EntityDescriptionSchema — pre-grounding entity description
    slots.py              SlotValue, ActionReasoningOutput, ActionClassification
  pycram_bridge/
    introspector.py       PycramIntrospector, FieldKind, ActionSchema, FieldSpec
  world/
    serializer.py         serialize_world_from_symbol_graph() — world → LLM string
    grounder.py           EntityGrounder — description → Symbol instance
  reasoning/
    slot_filler.py        run_slot_filler(), classify_action() — LLM prompt pipeline
    decomposer.py         TaskDecomposer — compound NL → atomic steps
    llm_config.py         make_llm(), LLMProvider — LLM factory

Quickstart — simple (fully NL-driven)
---------------------------------------
::

    from llm_reasoner import LLMBackend, nl_plan, nl_sequential
    from llm_reasoner.reasoning.llm_config import make_llm, LLMProvider
    from sdt_module import Body  # caller's groundable type

    llm = make_llm(LLMProvider.OPENAI, model="gpt-4o")

    plan = nl_plan(
        "pick up the milk from the table",
        context=context,
        llm=llm,
        groundable_type=Body,
    )
    plan.perform()

    for plan in nl_sequential(
        "go to the table, pick up the milk and put it in the fridge",
        context=context,
        llm=llm,
        groundable_type=Body,
    ):
        plan.perform()

Quickstart — power user (action type known, LLM fills free slots)
-------------------------------------------------------------------
::

    from krrood.entity_query_language.query.match import Match
    from pycram.plans.factories import execute_single
    from pycram.robot_plans.actions.core import PickUpAction

    context.query_backend = LLMBackend(
        instruction="pick up the milk from the table",
        llm=llm,
        groundable_type=Body,
        context={"manipulators": {"LEFT": left_manip, "RIGHT": right_manip}},
    )
    match = Match(PickUpAction)(object_designator=..., arm=..., grasp_description=...)
    plan = execute_single(match, context)
    plan.perform()
"""

from llm_reasoner.backend import LLMBackend
from llm_reasoner.factory import nl_plan, nl_sequential

__all__ = ["LLMBackend", "nl_plan", "nl_sequential"]
