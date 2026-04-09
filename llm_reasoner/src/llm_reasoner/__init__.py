"""
llm_reasoner — LLM-powered GenerativeBackend for KRROOD.

A standalone package that implements LLMBackend(GenerativeBackend), allowing
PyCRAM underspecified action Match expressions to be resolved via LLM reasoning
rather than probabilistic sampling or deterministic EQL lookup.

Dependency direction: llm_reasoner → krrood
krrood has zero knowledge of this package — no circular dependency.

Quickstart
----------
Simple user (fully NL-driven, LLM picks action type and fills all slots)::

    from llm_reasoner import LLMBackend, nl_plan, nl_sequential
    from llm_reasoner.workflows.llm_config import make_llm, LLMProvider

    llm = make_llm(LLMProvider.OPENAI, model="gpt-4o")

    plan = nl_plan("pick up the milk from the table", context=context, llm=llm)
    plan.perform()

    for plan in nl_sequential(
        "go to the table, pick up the milk and put it in the fridge",
        context=context,
        llm=llm,
    ):
        plan.perform()

Power user (knows action type, lets LLM fill free slots)::

    from krrood.entity_query_language.query.match import underspecified
    from pycram.plans.factories import execute_single
    from pycram.robot_plans.actions.core import PickUpAction

    context.query_backend = LLMBackend(
        instruction="pick up the milk from the table",
        llm=llm,
        world=context.world,
    )
    action = underspecified(PickUpAction)(object_designator=..., arm=..., grasp_description=...)
    plan = execute_single(action, context)
    plan.perform()
"""

from llm_reasoner.backend import LLMBackend
from llm_reasoner.nl_factory import nl_plan, nl_sequential

__all__ = ["LLMBackend", "nl_plan", "nl_sequential"]
