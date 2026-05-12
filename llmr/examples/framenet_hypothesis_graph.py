"""End-to-end example: FrameNet reasoner + hypothesis graph.

This example shows how to:

1. Resolve an underspecified KRROOD Match with llmr.
2. Run FrameNetReasoner as a pluggable reasoner.
3. Project the generated FrameNet sidecar into HypothesisGraph.
4. Query grounded FrameNet role hypotheses.

It assumes your environment already provides:

- a groundable Symbol subclass such as ``Body``
- an action type such as ``PickUpAction``
- a configured LangChain-compatible ``llm``

The example is intentionally conservative: grounded roles come only from
already resolved action slots.
"""

from __future__ import annotations

from krrood.entity_query_language.factories import an, entity, variable
from krrood.entity_query_language.query.match import Match

from llmr import instance_from_match
from llmr.hypotheses import (
    FrameNetFamily,
    RoleClaimNode,
    GroundingState,
    HypothesisGraph,
)
from llmr.reasoning.framenet_reasoner import FrameNetReasoner


def run_example(llm, Body, PickUpAction):
    """Resolve one action and project its FrameNet interpretation."""

    graph = HypothesisGraph()
    manager = FrameNetFamily.make_manager(graph)
    view = FrameNetFamily.make_view(graph)

    match = Match(PickUpAction)(object_designator=..., arm=..., grasp_description=...)

    action = instance_from_match(
        match,
        llm=llm,
        instruction="pick up the milk from the table",
        symbol_type=Body,
        reasoners=[FrameNetReasoner(llm=llm)],
        hypothesis_graph_manager=manager,
    )

    role = variable(RoleClaimNode, domain=view.roles())
    grounded_themes = list(
        an(
            entity(role).where(
                role.role_name == "theme",
                role.meta.grounding == GroundingState.SYMBOL_GROUNDED,
            )
        ).evaluate()
    )

    return {
        "action": action,
        "frames": view.frames(),
        "roles": view.roles(),
        "grounded_themes": grounded_themes,
    }
