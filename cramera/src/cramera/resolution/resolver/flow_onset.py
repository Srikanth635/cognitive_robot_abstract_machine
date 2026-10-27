"""
The tilt at which the substance a container holds starts to flow, as a physical
measurement gives it: measured in a physics simulation, or derived from the container's
shape when none was, and never stated by a language model.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import trimesh
from llm_reasoner.knowledge.access import (
    FlowOnsetQuery,
    MeasurementStore,
    StoredFlowOnset,
    action_core,
    core,
    flow_onset,
    pouring_frame,
)
from typing_extensions import Optional

from cramera.resolution.resolver.basin_geometry import ContainerBasin
from cramera.resolution.resolver.scene_world import SceneObjects

# %% estimates


class FlowOnsetMethod(StrEnum):
    """
    How the onset a pour is checked against was obtained.
    """

    PHYSICS_SIMULATION = "physics_simulation"
    """
    Measured by tilting the container, filled with simulated particles.
    """

    GEOMETRIC_DERIVATION = "geometric_derivation"
    """
    Derived from the container's shape, for a liquid whose surface stays level.
    """


@dataclass(frozen=True)
class OnsetEstimate:
    """
    The tilt beyond which the substance a container holds flows, and what it rests on.
    """

    degrees: float
    """
    The tilt from upright, in degrees, at which the substance flows: in every run of a
    simulation, or as derived from the shape.
    """

    method: FlowOnsetMethod
    """
    How the tilt was obtained.
    """

    citation: str
    """
    The IRI of the measurement the tilt is read from.
    """

    @classmethod
    def of(cls, stored: StoredFlowOnset) -> OnsetEstimate:
        """
        :return: The estimate a stored measurement gives.
        """
        return cls(
            degrees=stored.measurement.has_measured_range.upper_bound,
            method=(
                FlowOnsetMethod.PHYSICS_SIMULATION
                if stored.is_simulated
                else FlowOnsetMethod.GEOMETRIC_DERIVATION
            ),
            citation=stored.iri,
        )


# %% container shapes


@dataclass
class ContainerShapes:
    """
    The shapes of containers, read from their mesh files, each read once.
    """

    pitch: float = 0.001
    """
    The side, in metres, of the voxels a basin is derived on.
    """

    basins: dict[Path, ContainerBasin] = field(default_factory=dict, init=False)
    """
    The basins already read, by mesh file.
    """

    @staticmethod
    def container_model(mesh_file: Path) -> flow_onset.ContainerModel:
        """
        :return: The kind of container the mesh file makes, identified by the digest of
            its bytes.
        """
        return flow_onset.ContainerModel(
            mesh_digest=hashlib.sha256(mesh_file.read_bytes()).hexdigest(),
            mesh_file_name=mesh_file.name,
        )

    def basin(self, mesh_file: Path) -> ContainerBasin:
        """
        :return: The basin of the container the mesh file makes, in its own frame.
        """
        if mesh_file not in self.basins:
            self.basins[mesh_file] = ContainerBasin(
                mesh=trimesh.load(mesh_file, force="mesh"), pitch=self.pitch
            )
        return self.basins[mesh_file]


# %% the measured onset


@dataclass
class MeasuredFlowOnset:
    """
    The flow onset of the containers a scene bundle places, for what the scene states
    they hold.
    """

    bundle: Path
    """
    The scene's bundle directory, which states what each container holds.
    """

    store: MeasurementStore
    """
    The physical measurements the onset is read from, and derived onsets are kept in.
    """

    shapes: ContainerShapes = field(default_factory=ContainerShapes)
    """
    The shapes an onset is derived from when no measurement gives it.
    """

    objects: SceneObjects = field(init=False)
    """
    The objects the bundle's scene places.
    """

    def __post_init__(self) -> None:
        self.objects = SceneObjects.read(self.bundle)

    def onset_for(
        self, body_name: str, gravity_azimuth: float
    ) -> Optional[OnsetEstimate]:
        """
        :param body_name: The name of the container's body in the scene's world.
        :param gravity_azimuth: The side the container is tilted toward, in degrees about
            its own up axis.
        :return: The onset of what the scene states the container holds: a simulated
            one if one was measured, else one derived from the shape for a liquid;
            ``None`` when the scene states nothing it holds, or neither gives an onset.
        """
        scene_object = self.objects.with_body(body_name)
        if scene_object is None or scene_object.contents is None:
            return None
        mesh_file = self.bundle / scene_object.mesh
        container = self.shapes.container_model(mesh_file)
        query = FlowOnsetQuery(
            mesh_digest=container.mesh_digest,
            material_regime=scene_object.contents.material_regime,
            fill_fraction=scene_object.contents.fill_fraction,
            gravity_azimuth=gravity_azimuth,
        )
        stored = self.store.flow_onset(query)
        if stored is None:
            stored = self.derived(mesh_file, container, query)
        return None if stored is None else OnsetEstimate.of(stored)

    def derived(
        self,
        mesh_file: Path,
        container: flow_onset.ContainerModel,
        query: FlowOnsetQuery,
    ) -> Optional[StoredFlowOnset]:
        """
        Derive the onset from the container's shape and keep it in the store.

        :return: The derived onset as stored; ``None`` for a substance other than a
            liquid, an empty container, or one the liquid never leaves.
        """
        if (
            query.material_regime is not pouring_frame.MaterialRegime.LIQUID
            or query.fill_fraction == 0
        ):
            return None
        basin = self.shapes.basin(mesh_file)
        onset = basin.liquid_flow_onset(query.fill_fraction, query.gravity_azimuth)
        if onset is None:
            return None
        measurement = flow_onset.FlowOnsetTiltMeasurement(
            measured_for_container=container,
            at_fill_fraction=query.fill_fraction,
            has_material_regime=query.material_regime,
            gravity_azimuth=query.gravity_azimuth,
            has_measured_range=core.MagnitudeRange(
                lower_bound=onset.degrees,
                upper_bound=onset.degrees,
                has_unit=action_core.MeasurementUnit.DEGREE,
            ),
            follows_setup=flow_onset.GeometricDerivationSetup(voxel_pitch=onset.pitch),
        )
        return StoredFlowOnset(measurement=measurement, iri=self.store.put(measurement))
