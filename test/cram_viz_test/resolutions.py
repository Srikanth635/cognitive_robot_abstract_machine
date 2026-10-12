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

from cram_viz.resolution.designators import DesignatorQuery, PouringDesignator
from cram_viz.resolution.resolver import (
    DesignatorResolver,
    ResolvedAction,
    ResolvedDesignator,
    ResolvedSlot,
    ShownGround,
)
from cram_viz.resolution.runs import ResolutionRuns


def objects_given() -> ResolvedDesignator:
    """
    :return: Pouring from the given pouring cup into the given bowl, the arm and the
        tilt chosen, resting on a knowledge fragment and the bowl.
    """
    return ResolvedDesignator(
        query=DesignatorQuery.shipped(PouringDesignator.OBJECTS_GIVEN).source.text,
        designator="PouringAction",
        actions=[
            ResolvedAction(
                slots=[
                    ResolvedSlot(
                        "source_container",
                        "PouringCup (Static_MilkPitcher.stl)",
                        given=True,
                    ),
                    ResolvedSlot("target_container", "bowl.stl", given=True),
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
        run="a-run",
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


def resolver_replaying(
    name: str, store: HypothesisStore, runs: Path, times: int = 1
) -> DesignatorResolver:
    """
    :param name: The recording to replay.
    :param store: The knowledge store every run starts from.
    :param runs: Where the runs are kept.
    :param times: How many runs the recording is replayed for.
    :return: A resolver whose models replay the recording in each run.
    """
    recording = recorded(name)
    return DesignatorResolver(
        client=ScriptedClient(
            tool_turns=[json.dumps(turn["answer"]) for turn in recording["turns"]]
            * times
        ),
        baseline=KnowledgeService(
            backend=HypothesisBackend(
                client=ScriptedClient(
                    responses=[
                        json.dumps(answer) for answer in recording["generations"]
                    ]
                    * times
                ),
                suite=AcceptanceSuite(rules=[]),
                store=store,
            )
        ),
        runs=ResolutionRuns(directory=runs),
    )
