"""
Tests of the measurement store: physical measurements written as Turtle, checked against
their ontology before they are stored, and found again for a container, a substance, a
fill and a direction of tilt.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from llm_reasoner.knowledge.ontologies.action_perspectives import action_core
from llm_reasoner.knowledge.ontologies.physical_measurements import flow_onset
from llm_reasoner.knowledge.ontologies.predictive_manipulation import core
from llm_reasoner.knowledge.storage.exceptions import InvalidMeasurement
from llm_reasoner.knowledge.storage.measurements import (
    FlowOnsetQuery,
    MeasurementStore,
)

# %% fixtures

CUP_DIGEST = "a" * 64
"""
The digest standing for the cup's mesh file.
"""


def measurement(
    fill: float = 0.5,
    azimuth: float = 90.0,
    onset: tuple[float, float] = (48.0, 48.0),
    setup: flow_onset.MeasurementSetup | None = None,
) -> flow_onset.FlowOnsetTiltMeasurement:
    """
    :return: A flow onset of a liquid in the cup, derived from its shape unless another
        setup is given.
    """
    return flow_onset.FlowOnsetTiltMeasurement(
        measured_for_container=flow_onset.ContainerModel(
            mesh_digest=CUP_DIGEST, mesh_file_name="cup.stl"
        ),
        at_fill_fraction=fill,
        has_material_regime=action_core.MaterialRegime.LIQUID,
        gravity_azimuth=azimuth,
        has_measured_range=core.MagnitudeRange(
            lower_bound=onset[0],
            upper_bound=onset[1],
            has_unit=core.MeasurementUnit.DEGREE,
        ),
        follows_setup=setup or flow_onset.GeometricDerivationSetup(voxel_pitch=0.001),
    )


def simulation_setup() -> flow_onset.PhysicsSimulationSetup:
    """
    :return: A setup of a simulated tilt, repeated three times.
    """
    return flow_onset.PhysicsSimulationSetup(
        simulation_engine="Isaac Sim",
        simulation_engine_version="6.0.0.1",
        particle_spacing=0.002,
        tilt_rate=5.0,
        time_step=0.005,
        run_count=3,
        onset_particle_count=10,
        fluid_viscosity=0.0,
        fluid_cohesion=0.01,
        fluid_surface_tension=0.0,
        has_settled_fill_range=core.MagnitudeRange(
            lower_bound=0.49, upper_bound=0.51, has_unit=core.MeasurementUnit.RATIO
        ),
    )


def query(fill: float = 0.5, azimuth: float = 90.0) -> FlowOnsetQuery:
    """
    :return: A question for the flow onset of a liquid in the cup.
    """
    return FlowOnsetQuery(
        mesh_digest=CUP_DIGEST,
        material_regime=action_core.MaterialRegime.LIQUID,
        fill_fraction=fill,
        gravity_azimuth=azimuth,
    )


@pytest.fixture
def store(tmp_path: Path) -> MeasurementStore:
    """
    An empty store.
    """
    return MeasurementStore(directory=tmp_path / "measurements")


# %% storing


class TestStoring:
    def test_a_stored_measurement_is_read_back_with_its_values(self, store):
        stored = measurement(setup=simulation_setup(), onset=(46.0, 49.5))
        store.put(stored)
        (read,) = store.flow_onsets()
        assert read.measurement.at_fill_fraction == stored.at_fill_fraction
        assert read.measurement.gravity_azimuth == stored.gravity_azimuth
        assert read.measurement.has_material_regime is stored.has_material_regime
        assert read.measurement.has_measured_range.lower_bound == 46.0
        assert read.measurement.has_measured_range.upper_bound == 49.5
        assert read.measurement.has_measured_range.has_unit is (
            core.MeasurementUnit.DEGREE
        )
        assert read.measurement.measured_for_container.mesh_digest == CUP_DIGEST
        assert isinstance(
            read.measurement.follows_setup, flow_onset.PhysicsSimulationSetup
        )
        assert read.measurement.follows_setup.run_count == 3

    def test_storing_names_the_measurement_by_the_iri_it_is_read_under(self, store):
        iri = store.put(measurement())
        (read,) = store.flow_onsets()
        assert read.iri == iri

    def test_measuring_the_same_case_again_replaces_the_earlier_value(self, store):
        store.put(measurement(onset=(48.0, 48.0)))
        store.put(measurement(onset=(47.0, 47.0)))
        (read,) = store.flow_onsets()
        assert read.measurement.has_measured_range.lower_bound == 47.0

    def test_a_measurement_breaking_its_ontology_is_refused_and_not_stored(self, store):
        with pytest.raises(InvalidMeasurement):
            store.put(measurement(fill=1.5))
        assert store.flow_onsets() == []


# %% finding


class TestFinding:
    def test_a_missing_measurement_is_none(self, store):
        assert store.flow_onset(query()) is None

    def test_the_measurement_nearest_the_asked_fill_is_found(self, store):
        store.put(measurement(fill=0.25, onset=(70.0, 70.0)))
        store.put(measurement(fill=0.5, onset=(48.0, 48.0)))
        found = store.flow_onset(query(fill=0.49))
        assert found.measurement.at_fill_fraction == 0.5

    def test_a_fill_beyond_the_tolerance_finds_nothing(self, store):
        store.put(measurement(fill=0.5))
        asked = query()
        assert (
            store.flow_onset(
                replace(asked, fill_fraction=0.5 + 2 * asked.fill_tolerance)
            )
            is None
        )

    def test_an_azimuth_is_near_across_the_full_turn(self, store):
        store.put(measurement(azimuth=359.0))
        assert store.flow_onset(query(azimuth=1.0)) is not None

    def test_another_container_or_substance_finds_nothing(self, store):
        store.put(measurement())
        assert store.flow_onset(replace(query(), mesh_digest="b" * 64)) is None
        assert (
            store.flow_onset(
                replace(query(), material_regime=action_core.MaterialRegime.GRANULAR)
            )
            is None
        )

    def test_a_simulation_is_preferred_to_a_derivation(self, store):
        store.put(measurement(onset=(48.0, 48.0)))
        store.put(measurement(onset=(45.0, 50.0), setup=simulation_setup()))
        found = store.flow_onset(query())
        assert isinstance(
            found.measurement.follows_setup, flow_onset.PhysicsSimulationSetup
        )
