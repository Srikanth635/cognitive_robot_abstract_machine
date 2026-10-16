"""
The comparison of the resolver with and without the action frame, against the live
model, run only when asked to (``LLM_REASONER_LIVE=1``, an API key and
``FRAME_COMPARISON_OUTPUT`` naming where the records go), since every run costs money.

The arm without the frame runs first; both read the knowledge stored in
``LLM_REASONER_KNOWLEDGE_STORE``. ``python -m experiments.frame_comparison`` summarises
the records afterwards.
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import pytest
from llm_reasoner.clients.anthropic_client import AnthropicClient
from llm_reasoner.clients.usage import SONNET_5_5_PRICES
from llm_reasoner.knowledge.access import HypothesisStore

from cram_viz import paths
from cram_viz.resolution.designators import DesignatorQuery, PouringDesignator
from cram_viz.resolution.resolver import RESOLVER_MAX_TOKENS, DesignatorResolver
from cram_viz.resolution.runs import ResolutionRuns, RunFile
from experiments.frame_comparison.output import (
    KITCHEN_RUNS_PER_ARM,
    POUR_RUNS_PER_ARM,
    ComparisonOutput,
    ComparisonVariable,
)
from experiments.frame_comparison.records import ComparisonArm, RunRecord
from experiments.llm_reasoner_kitchen.cases import KNOWLEDGE_CASES
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


@pytest.mark.parametrize("run", range(POUR_RUNS_PER_ARM))
@pytest.mark.parametrize("arm", list(ComparisonArm))
def test_the_pour_is_resolved_and_tried(arm, run, output, store):
    bundle = paths.scenes_dir() / POURING_SCENE
    if not bundle.is_dir():
        pytest.skip(f"the {POURING_SCENE} scene is not available")
    resolver = DesignatorResolver(
        client=AnthropicClient(model=LIVE_MODEL, max_tokens=RESOLVER_MAX_TOKENS),
        baseline=LIVE.reasoner(store).knowledge,
        runs=ResolutionRuns(directory=output.directory / "runs"),
        perspectives=arm.perspectives,
    )
    resolved = resolver.resolve(
        DesignatorQuery.shipped(PouringDesignator.POUR_GEOMETRY_OPEN),
        POURING_SCENE,
        bundle,
    ).resolution
    record = RunRecord.of_designator(
        resolved,
        output.directory / "runs" / resolved.run / RunFile.EVENTS,
        case=PouringDesignator.POUR_GEOMETRY_OPEN.value,
        arm=arm,
        run=run,
    )
    output.add(record)
    print(record.to_json())


@pytest.mark.parametrize("run", range(KITCHEN_RUNS_PER_ARM))
@pytest.mark.parametrize(
    "case", COMPARED_KITCHEN_CASES, ids=lambda case: case.identifier
)
@pytest.mark.parametrize("arm", list(ComparisonArm))
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
