"""
The comparison of the resolver with and without the action frame, against the live
model, run only when asked to (``LLM_REASONER_LIVE=1``, an API key and
``FRAME_COMPARISON_OUTPUT`` naming where the records go), since every run costs money.

The arm without the frame runs first; both read the knowledge stored in
``LLM_REASONER_KNOWLEDGE_STORE``. ``python -m experiments.frame_comparison`` summarises
the records afterwards.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import pytest
from llm_reasoner.clients.anthropic_client import AnthropicClient
from llm_reasoner.clients.usage import SONNET_5_5_PRICES
from llm_reasoner.knowledge.access import HypothesisStore
from llm_reasoner.resolver.exceptions import AmbiguousReferent, NoFittingReferent

from cram_viz import paths
from cram_viz.resolution.designators import DesignatorQuery, PouringDesignator
from cram_viz.resolution.resolver import RESOLVER_MAX_TOKENS, DesignatorResolver
from cram_viz.resolution.overturned_scene import OverturnedScene
from cram_viz.resolution.runs import ResolutionRuns, RunFile
from experiments.frame_comparison.output import (
    KITCHEN_RUNS_PER_ARM,
    OPEN_CONTAINER_KITCHEN_CASES,
    OPEN_CONTAINER_RUNS_PER_ARM,
    POUR_RUNS_PER_ARM,
    ComparisonOutput,
    ComparisonVariable,
)
from experiments.frame_comparison.records import ComparisonArm, RunRecord
from experiments.llm_reasoner_kitchen.cases import KNOWLEDGE_CASES, RESOLUTION_CASES
from experiments.llm_reasoner_kitchen.live import LIVE_MODEL, LiveCheck
from experiments.llm_reasoner_kitchen.resolution import KitchenResolver

LIVE = LiveCheck()
"""
Whether and how the cases are checked against the live model.
"""

POURING_SCENE = "simple_pouring"
"""
The scene cram_viz's pour is resolved and tried in.
"""

FRAME_ARMS = (ComparisonArm.WITHOUT_FRAME, ComparisonArm.WITH_FRAME)
"""
The arms differing in the action frame, the only ones the kitchen, which tries no pour,
compares.
"""

COMPARED_KITCHEN_CASES = KNOWLEDGE_CASES[:5]
"""
The kitchen's knowledge cases K1 to K5.
"""

pytestmark = pytest.mark.skipif(
    not (LIVE.enabled() and os.environ.get(ComparisonVariable.OUTPUT)),
    reason=f"the comparison runs only live and with {ComparisonVariable.OUTPUT} set",
)


@pytest.fixture(scope="module")
def output() -> ComparisonOutput:
    """
    Where the comparison keeps its records.
    """
    return ComparisonOutput(directory=Path(os.environ[ComparisonVariable.OUTPUT]))


@pytest.fixture(scope="module")
def store(tmp_path_factory) -> HypothesisStore:
    """
    The knowledge both arms read, and the kitchen's generations add to.
    """
    return HypothesisStore(
        directory=LIVE.store_directory(tmp_path_factory.mktemp("knowledge"))
    )


def resolve_and_record(
    designator: PouringDesignator,
    arm: ComparisonArm,
    run: int,
    output: ComparisonOutput,
    store: HypothesisStore,
) -> None:
    """
    Resolve one of cram_viz's pouring designators in its scene, trying its pours in the
    simulation, and keep the run's record.
    """
    bundle = paths.scenes_dir() / POURING_SCENE
    if not bundle.is_dir():
        pytest.skip(f"the {POURING_SCENE} scene is not available")
    resolver = DesignatorResolver(
        client=AnthropicClient(model=LIVE_MODEL, max_tokens=RESOLVER_MAX_TOKENS),
        baseline=LIVE.reasoner(store).knowledge,
        runs=ResolutionRuns(directory=output.directory / "runs"),
        perspectives=arm.perspectives,
        predicts_pours=arm.predicts_pours,
    )
    resolved = resolver.resolve(
        DesignatorQuery.shipped(designator), POURING_SCENE, bundle
    ).resolution
    record = RunRecord.of_designator(
        resolved,
        output.directory / "runs" / resolved.run / RunFile.EVENTS,
        case=designator.value,
        arm=arm,
        run=run,
    )
    output.add(record)
    print(record.to_json())


@pytest.mark.parametrize("run", range(POUR_RUNS_PER_ARM))
@pytest.mark.parametrize("arm", list(ComparisonArm))
def test_the_pour_is_resolved_and_tried(arm, run, output, store):
    resolve_and_record(PouringDesignator.POUR_GEOMETRY_OPEN, arm, run, output, store)


@pytest.mark.parametrize("run", range(OPEN_CONTAINER_RUNS_PER_ARM))
@pytest.mark.parametrize("arm", list(ComparisonArm))
def test_the_pour_choosing_its_containers_is_resolved_and_tried(
    arm, run, output, store
):
    resolve_and_record(PouringDesignator.OBJECTS_OPEN, arm, run, output, store)


@pytest.mark.parametrize("run", range(OPEN_CONTAINER_RUNS_PER_ARM))
@pytest.mark.parametrize(
    "case",
    [
        case
        for case in RESOLUTION_CASES
        if case.identifier in OPEN_CONTAINER_KITCHEN_CASES
    ],
    ids=lambda case: case.identifier,
)
@pytest.mark.parametrize("arm", FRAME_ARMS)
def test_a_kitchen_case_choosing_its_containers_is_resolved(
    arm, case, run, kitchen, output, store
):
    reasoner = replace(LIVE.reasoner(store), perspectives=arm.perspectives)
    resolution = KitchenResolver(reasoner=reasoner).resolve(
        case.query(kitchen), case.task
    )
    record = RunRecord.of_kitchen(
        resolution,
        case.identifier,
        arm,
        run,
        SONNET_5_5_PRICES,
        meets=case.meets(resolution.actions),
    )
    output.add(record)
    print(record.to_json())


@pytest.mark.parametrize("run", range(KITCHEN_RUNS_PER_ARM))
@pytest.mark.parametrize(
    "case", COMPARED_KITCHEN_CASES, ids=lambda case: case.identifier
)
@pytest.mark.parametrize("arm", FRAME_ARMS)
def test_a_kitchen_case_is_resolved(arm, case, run, kitchen, output, store):
    reasoner = replace(LIVE.reasoner(store), perspectives=arm.perspectives)
    resolution = KitchenResolver(reasoner=reasoner).resolve(
        kitchen.pour_from_the_cup_into_the_bowl(), case.task
    )
    record = RunRecord.of_kitchen(
        resolution, case.identifier, arm, run, SONNET_5_5_PRICES
    )
    output.add(record)
    print(record.to_json())


# %% a goal that cannot receive the pour

OVERTURNED_GOAL = "bowl"
"""
The identifier of the object turned upside down in the scene the trap is run in: the
only bowl, so that every container left is a poor goal.
"""

TRAP_RECORDS = "overturned_goal.jsonl"
"""
The file, beside the comparison's records, that the trap's outcomes are kept in.
"""

TRAP_RUNS_PER_ARM = 2
"""
How many times each arm is asked to pour with the goal turned upside down.
"""


@dataclass
class TrapOutcome:
    """
    What a resolver did when the only bowl stood upside down.
    """

    arm: str
    """
    The arm that resolved.
    """

    run: int
    """
    Which run of the arm it was.
    """

    refused: str
    """
    The refusal the resolver answered with, as the name of its kind; empty when it
    resolved.
    """

    target_container: str
    """
    The goal it chose, as the resolved slot writes it; empty when it refused.
    """

    why: str
    """
    The reason it gave.
    """

    cost: float
    """
    What its conversations cost, in dollars, when it resolved.
    """


@pytest.mark.parametrize("run", range(TRAP_RUNS_PER_ARM))
@pytest.mark.parametrize("arm", list(ComparisonArm))
def test_the_pour_is_not_made_into_an_overturned_goal(
    arm, run, output, store, tmp_path_factory
):
    bundle = paths.scenes_dir() / POURING_SCENE
    if not bundle.is_dir():
        pytest.skip(f"the {POURING_SCENE} scene is not available")
    trap = OverturnedScene(bundle=bundle, identifier=OVERTURNED_GOAL).write_to(
        tmp_path_factory.mktemp("overturned") / POURING_SCENE
    )
    resolver = DesignatorResolver(
        client=AnthropicClient(model=LIVE_MODEL, max_tokens=RESOLVER_MAX_TOKENS),
        baseline=LIVE.reasoner(store).knowledge,
        runs=ResolutionRuns(directory=output.directory / "runs"),
        perspectives=arm.perspectives,
        predicts_pours=arm.predicts_pours,
    )
    query = DesignatorQuery.shipped(PouringDesignator.OBJECTS_OPEN)
    try:
        resolved = resolver.resolve(query, POURING_SCENE, trap).resolution
    except (NoFittingReferent, AmbiguousReferent) as refusal:
        outcome = TrapOutcome(
            arm=arm.value,
            run=run,
            refused=type(refusal).__name__,
            target_container="",
            why=refusal.reason,
            cost=0.0,
        )
    else:
        (action,) = resolved.actions[:1]
        outcome = TrapOutcome(
            arm=arm.value,
            run=run,
            refused="",
            target_container=next(
                slot.value for slot in action.slots if slot.name == "target_container"
            ),
            why=action.why,
            cost=resolved.cost,
        )
    with (output.directory / TRAP_RECORDS).open("a") as records:
        records.write(json.dumps(asdict(outcome)) + "\n")
    print(outcome)
