"""
Resolutions for the tests: one as the resolver reports it, and a resolver replaying the
models' recorded answers.
"""

from __future__ import annotations

import json
from pathlib import Path

from llm_reasoner.clients.base import ScriptedClient
from llm_reasoner.knowledge.access import (
    AcceptanceSuite,
    HypothesisBackend,
    HypothesisStore,
    KnowledgeService,
)
from llm_reasoner.resolver.grounds import GroundKey

from cram_viz.resolution.designators import PouringVariant
from cram_viz.resolution.resolver import (
    DesignatorResolver,
    ResolvedAction,
    ResolvedDesignator,
    ResolvedSlot,
    ShownGround,
)


def vessel_into_bowl() -> ResolvedDesignator:
    """
    :return: Pouring from the vessel into the bowl, the arm and the tilt chosen, resting
        on a knowledge fragment and the bowl.
    """
    return ResolvedDesignator(
        variant=PouringVariant.VESSEL_INTO_BOWL,
        label="pour from the vessel into the bowl",
        task="",
        designator="PouringAction",
        actions=[
            ResolvedAction(
                slots=[
                    ResolvedSlot("target_container", "bowl.stl", given=True),
                    ResolvedSlot(
                        "source_container",
                        "PouringCup (Static_MilkPitcher.stl)",
                        given=True,
                    ),
                    ResolvedSlot("arm", "RIGHT", given=False),
                    ResolvedSlot("tilt_angle", "1.85", given=False),
                ],
                why="The default tilt lets the substance flow.",
                grounds=[
                    ShownGround(GroundKey.KNOWLEDGE, "ph_tilt"),
                    ShownGround(GroundKey.WORLD, "Body#1 (bowl.stl)"),
                ],
            )
        ],
        situation={},
        record_id=None,
        cost=0.05,
    )


CASES = Path(__file__).parent / "resolution_cases"
"""
The recorded answers the resolutions replay.
"""


def recorded(name: str) -> dict:
    """
    :return: A recording of the models' answers to one resolution.
    """
    return json.loads((CASES / f"{name}.json").read_text())


def resolver_replaying(name: str, store: HypothesisStore) -> DesignatorResolver:
    """
    :return: A resolver whose models replay the recording.
    """
    recording = recorded(name)
    return DesignatorResolver(
        client=ScriptedClient(
            tool_turns=[json.dumps(turn["answer"]) for turn in recording["turns"]]
        ),
        knowledge=KnowledgeService(
            backend=HypothesisBackend(
                client=ScriptedClient(
                    responses=[
                        json.dumps(answer) for answer in recording["generations"]
                    ]
                ),
                suite=AcceptanceSuite(rules=[]),
                store=store,
            )
        ),
    )
