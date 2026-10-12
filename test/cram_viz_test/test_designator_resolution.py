"""
Tests of resolving the viewer's predefined pouring designators in a scene: what each
variant gives and leaves open, which scenes it can be resolved in, and what a resolution
reports, replayed from recorded answers.
"""

from __future__ import annotations

import json
from pathlib import Path

from llm_reasoner.knowledge.access import HypothesisStore
from llm_reasoner.resolver.grounds import GroundKey
from semantic_digital_twin.semantic_annotations.semantic_annotations import PouringCup

from cram_viz.resolution.designators import (
    POURING_VARIANTS,
    PouringVariant,
    SceneQueries,
    variant_named,
)
from cram_viz.resolution.scene_world import SceneKey, SceneWorldBuilder

from .resolutions import resolver_replaying

# %% helpers


def without_object(bundle: Path, identifier: str) -> Path:
    """
    :return: The bundle with the scene object of the given identifier left out.
    """
    scene_file = bundle / "scene.json"
    scene = json.loads(scene_file.read_text())
    scene[SceneKey.OBJECTS] = [
        entry
        for entry in scene[SceneKey.OBJECTS]
        if entry[SceneKey.IDENTIFIER] != identifier
    ]
    scene_file.write_text(json.dumps(scene))
    return bundle


# %% the variants


class TestVariants:
    def test_every_variant_is_written_on_a_scene_with_a_pouring_vessel(
        self, pouring_bundle
    ):
        queries = SceneQueries(world=SceneWorldBuilder().build(pouring_bundle))
        assert [variant.missing_in(queries) for variant in POURING_VARIANTS] == [
            []
        ] * len(POURING_VARIANTS)
        for variant in POURING_VARIANTS:
            variant.query(queries)

    def test_a_variant_without_a_pouring_vessel_names_what_is_missing(
        self, pouring_bundle
    ):
        world = SceneWorldBuilder().build(
            without_object(pouring_bundle, "Static_MilkPitcher")
        )
        variant = variant_named(PouringVariant.VESSEL_INTO_BOWL)
        assert variant.missing_in(SceneQueries(world=world)) == [PouringCup]

    def test_a_variant_states_what_it_gives_and_what_it_leaves_open(
        self, pouring_bundle
    ):
        queries = SceneQueries(world=SceneWorldBuilder().build(pouring_bundle))
        slots = variant_named(PouringVariant.INTO_THE_BOWL).slots(queries)
        assert slots.given == ["target_container"]
        assert slots.left_open == ["source_container", "arm", "tilt_angle"]


# %% resolving


class TestResolution:
    def test_a_resolution_names_the_value_of_every_slot(self, pouring_bundle, tmp_path):
        resolved = resolver_replaying(
            "vessel_into_bowl", HypothesisStore(directory=tmp_path / "store")
        ).resolve(
            variant_named(PouringVariant.VESSEL_INTO_BOWL), "pouring", pouring_bundle
        )
        (action,) = resolved.actions
        assert {slot.name: slot.value for slot in action.slots} == {
            "target_container": "bowl.stl",
            "source_container": "PouringCup (Static_MilkPitcher.stl)",
            "arm": "RIGHT",
            "tilt_angle": "1.85",
        }

    def test_a_resolution_names_what_its_answer_rests_on(
        self, pouring_bundle, tmp_path
    ):
        resolved = resolver_replaying(
            "vessel_into_bowl", HypothesisStore(directory=tmp_path / "store")
        ).resolve(
            variant_named(PouringVariant.VESSEL_INTO_BOWL), "pouring", pouring_bundle
        )
        (action,) = resolved.actions
        assert [(ground.source, ground.name) for ground in action.grounds] == [
            (GroundKey.KNOWLEDGE, "pour"),
            (GroundKey.WORLD, "Body#1 (bowl.stl)"),
        ]

    def test_a_resolution_reports_the_knowledge_it_consulted(
        self, pouring_bundle, tmp_path
    ):
        store = HypothesisStore(directory=tmp_path / "store")
        resolved = resolver_replaying("vessel_into_bowl", store).resolve(
            variant_named(PouringVariant.VESSEL_INTO_BOWL), "pouring", pouring_bundle
        )
        (record,) = store.records()
        assert resolved.record_id == record.record_id
        assert resolved.situation == {}
