"""
Tests of the pour check requiring the tilt at which the substance a container holds
starts to flow, as a physical measurement gives it: measured in a physics simulation, or
derived from the container's shape when no simulation measured it.

A pour tilted less than that lands its mouth over the opening and pours nothing, so it
is refused.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from llm_reasoner.knowledge.access import (
    MeasurementStore,
    action_core,
    core,
    flow_onset,
)
from llm_reasoner.resolver.exceptions import OutcomeFellShort
from typing_extensions import Any, Dict

from cram_viz.resolution.basin_geometry import ContainerBasin
from cram_viz.resolution.container_contents import ContainerContents
from cram_viz.resolution.flow_onset import (
    ContainerShapes,
    FlowOnsetMethod,
    MeasuredFlowOnset,
)
from cram_viz.resolution.scene_world import SCENE_FILE, SceneKey, SceneObjects
from cram_viz.resolution.trial import OutcomeKey

from .resolutions import LANDS, HandSide
from .test_pouring_trial import PourAnswer, checking

# %% helpers


def filled_object(bundle: Path):
    """
    :return: The object of the bundle's scene whose contents the scene states.
    """
    (filled,) = [
        scene_object
        for scene_object in SceneObjects.read(bundle).objects
        if scene_object.contents is not None
    ]
    return filled


def simulated_record(
    bundle: Path, onset: tuple[float, float]
) -> flow_onset.FlowOnsetTiltMeasurement:
    """
    :return: A simulated flow onset of the bundle's filled container, at the fill the
        scene states, tilted toward its own x axis.
    """
    filled = filled_object(bundle)
    return flow_onset.FlowOnsetTiltMeasurement(
        measured_for_container=ContainerShapes().container_model(bundle / filled.mesh),
        at_fill_fraction=filled.contents.fill_fraction,
        has_material_regime=filled.contents.material_regime,
        gravity_azimuth=0.0,
        has_measured_range=core.MagnitudeRange(
            lower_bound=onset[0],
            upper_bound=onset[1],
            has_unit=core.MeasurementUnit.DEGREE,
        ),
        follows_setup=flow_onset.PhysicsSimulationSetup(
            simulation_engine="Isaac Sim",
            simulation_engine_version="6.0.0.1",
            particle_spacing=0.002,
            tilt_rate=5.0,
            time_step=0.005,
            run_count=2,
            onset_particle_count=10,
            fluid_viscosity=0.0000017,
            fluid_cohesion=0.01,
            fluid_surface_tension=0.0074,
            has_settled_fill_range=core.MagnitudeRange(
                lower_bound=0.5, upper_bound=0.5, has_unit=core.MeasurementUnit.RATIO
            ),
        ),
    )


def tilted_outcome(bundle: Path, degrees: float) -> Dict[str, Any]:
    """
    :return: The outcome of a pour landing in the opening with the bundle's filled
        container tilted toward its own x axis.
    """
    return {
        **LANDS,
        OutcomeKey.SOURCE_CONTAINER: filled_object(bundle).body_name,
        OutcomeKey.SOURCE_TILT_DEGREES: degrees,
        OutcomeKey.SOURCE_GRAVITY_AZIMUTH_DEGREES: 0.0,
    }


def stating_contents(bundle: Path, contents: ContainerContents) -> Path:
    """
    :return: The bundle, its filled container now stated to hold the contents.
    """
    scene_file = bundle / SCENE_FILE
    scene = json.loads(scene_file.read_text())
    for entry in scene[SceneKey.OBJECTS]:
        if SceneKey.CONTENTS in entry:
            entry[SceneKey.CONTENTS] = contents.to_json()
    scene_file.write_text(json.dumps(scene))
    return bundle


@pytest.fixture
def store(tmp_path: Path) -> MeasurementStore:
    """
    An empty measurement store.
    """
    return MeasurementStore(directory=tmp_path / "measurements")


SIMULATED_ONSET = (60.0, 64.0)
"""
The lowest and highest onset, in degrees, over the runs of a simulated tilt.
"""


# %% the measured onset


class TestMeasuredFlowOnset:
    def test_a_simulated_onset_of_the_container_is_used(self, pouring_bundle, store):
        iri = store.put(simulated_record(pouring_bundle, SIMULATED_ONSET))
        estimate = MeasuredFlowOnset(bundle=pouring_bundle, store=store).onset_for(
            filled_object(pouring_bundle).body_name, 0.0
        )
        assert estimate.degrees == SIMULATED_ONSET[1]
        assert estimate.method is FlowOnsetMethod.PHYSICS_SIMULATION
        assert estimate.citation == iri

    def test_without_a_simulation_a_liquids_onset_is_derived_from_the_shape(
        self, pouring_bundle, store
    ):
        filled = filled_object(pouring_bundle)
        estimate = MeasuredFlowOnset(bundle=pouring_bundle, store=store).onset_for(
            filled.body_name, 0.0
        )
        derived = ContainerShapes().basin(pouring_bundle / filled.mesh)
        assert estimate.method is FlowOnsetMethod.GEOMETRIC_DERIVATION
        assert estimate.degrees == pytest.approx(
            derived.liquid_flow_onset(filled.contents.fill_fraction, 0.0).degrees
        )

    def test_a_derived_onset_is_stored_and_served_from_the_store_again(
        self, pouring_bundle, store
    ):
        body = filled_object(pouring_bundle).body_name
        first = MeasuredFlowOnset(bundle=pouring_bundle, store=store).onset_for(
            body, 0.0
        )
        (stored,) = store.flow_onsets()
        assert isinstance(
            stored.measurement.follows_setup, flow_onset.GeometricDerivationSetup
        )
        again = MeasuredFlowOnset(bundle=pouring_bundle, store=store).onset_for(
            body, 0.0
        )
        assert again.citation == first.citation == stored.iri
        assert len(store.flow_onsets()) == 1

    def test_a_simulation_is_preferred_to_a_derivation(self, pouring_bundle, store):
        measured = MeasuredFlowOnset(bundle=pouring_bundle, store=store)
        body = filled_object(pouring_bundle).body_name
        measured.onset_for(body, 0.0)
        store.put(simulated_record(pouring_bundle, SIMULATED_ONSET))
        assert measured.onset_for(body, 0.0).method is (
            FlowOnsetMethod.PHYSICS_SIMULATION
        )

    def test_a_container_whose_contents_are_not_stated_has_no_onset(
        self, pouring_bundle, store
    ):
        unstated = next(
            scene_object
            for scene_object in SceneObjects.read(pouring_bundle).objects
            if scene_object.contents is None
        )
        measured = MeasuredFlowOnset(bundle=pouring_bundle, store=store)
        assert measured.onset_for(unstated.body_name, 0.0) is None

    def test_a_granular_substance_without_a_simulation_has_no_onset(
        self, pouring_bundle, store
    ):
        bundle = stating_contents(
            pouring_bundle,
            ContainerContents(
                material_regime=action_core.MaterialRegime.GRANULAR, fill_fraction=0.5
            ),
        )
        measured = MeasuredFlowOnset(bundle=bundle, store=store)
        assert measured.onset_for(filled_object(bundle).body_name, 0.0) is None
        assert store.flow_onsets() == []


# %% the check


class TestPourFlows:
    def check_measuring(self, bundle: Path, store: MeasurementStore, outcome: dict):
        """
        :return: A check whose trial reports the outcome, the onset measured by a
            simulation of the bundle's filled container.
        """
        store.put(simulated_record(bundle, SIMULATED_ONSET))
        check = checking([outcome])
        check.flow_onset = MeasuredFlowOnset(bundle=bundle, store=store)
        return check

    def answer(self) -> PourAnswer:
        return PourAnswer(arm=HandSide.RIGHT, tilt_angle=0.2)

    def test_a_pour_landing_in_the_opening_below_the_onset_is_refused(
        self, pouring_bundle, store
    ):
        check = self.check_measuring(
            pouring_bundle, store, tilted_outcome(pouring_bundle, 30.0)
        )
        with pytest.raises(OutcomeFellShort) as refusal:
            check.check([self.answer()])
        assert refusal.value.outcome[OutcomeKey.TILT_REACHES_FLOW_ONSET] is False

    def test_the_refusal_states_the_onset_and_the_measurement_it_rests_on(
        self, pouring_bundle, store
    ):
        check = self.check_measuring(
            pouring_bundle, store, tilted_outcome(pouring_bundle, 30.0)
        )
        with pytest.raises(OutcomeFellShort) as refusal:
            check.check([self.answer()])
        outcome = refusal.value.outcome
        (stored,) = store.flow_onsets()
        assert outcome[OutcomeKey.FLOW_ONSET_TILT_DEGREES] == SIMULATED_ONSET[1]
        assert outcome[OutcomeKey.FLOW_ONSET_METHOD] == (
            FlowOnsetMethod.PHYSICS_SIMULATION
        )
        assert outcome[OutcomeKey.FLOW_ONSET_MEASUREMENT] == stored.iri

    def test_a_pour_tilted_to_the_onset_is_accepted(self, pouring_bundle, store):
        check = self.check_measuring(
            pouring_bundle, store, tilted_outcome(pouring_bundle, SIMULATED_ONSET[1])
        )
        check.check([self.answer()])
        assert len(check.trials) == 1

    def test_a_pour_below_the_onset_is_never_kept_as_a_last_resort(
        self, pouring_bundle, store
    ):
        check = self.check_measuring(
            pouring_bundle, store, tilted_outcome(pouring_bundle, 30.0)
        )
        with pytest.raises(OutcomeFellShort) as refusal:
            check.check([self.answer()])
        assert refusal.value.fallback_rank is None

    def test_without_a_measured_onset_a_shallow_pour_is_accepted(self, pouring_bundle):
        check = checking([tilted_outcome(pouring_bundle, 30.0)])
        check.check([self.answer()])
        assert len(check.trials) == 1

    def test_an_outcome_not_stating_the_sources_tilt_leaves_it_unchecked(
        self, pouring_bundle, store
    ):
        check = self.check_measuring(pouring_bundle, store, LANDS)
        check.check([self.answer()])
        assert len(check.trials) == 1
