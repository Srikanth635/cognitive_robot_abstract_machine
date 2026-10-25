"""
A store of physical measurements on disk, one Turtle file per measurement, each checked
against its ontology before it is stored.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from rdflib import Graph, URIRef
from typing_extensions import Optional

from llm_reasoner.knowledge.acceptance.derived import RangesHold, RestrictionsHold
from llm_reasoner.knowledge.acceptance.rules import AcceptanceSuite, Severity
from llm_reasoner.knowledge.acceptance.shapes import ShapesHold
from llm_reasoner.knowledge.ontologies.action_perspectives import action_core
from llm_reasoner.knowledge.ontologies.physical_measurements import flow_onset
from llm_reasoner.knowledge.representation.rdf import GraphReader, GraphWriter
from llm_reasoner.knowledge.storage.exceptions import InvalidMeasurement

# %% questions


@dataclass(frozen=True, order=True)
class MeasurementDistance:
    """
    How far a measurement's case lies from the one asked about, the fill compared first.
    """

    fill: float
    """
    The difference in fill fraction.
    """

    azimuth: float
    """
    The difference, in degrees, in the direction of tilt, the shorter way round.
    """


@dataclass(frozen=True)
class FlowOnsetQuery:
    """
    A question for the tilt at which a substance starts to leave a kind of container.
    """

    mesh_digest: str
    """
    The SHA-256 digest of the container's mesh file, in hexadecimal.
    """

    material_regime: action_core.MaterialRegime
    """
    How the substance flows.
    """

    fill_fraction: float
    """
    The substance's volume over the volume the container holds when upright.
    """

    gravity_azimuth: float
    """
    The direction, in degrees about the container's own up axis, toward which it tilts.
    """

    fill_tolerance: float = 0.02
    """
    How far, as a fraction of the container's volume, a measured fill may lie from the
    asked one.
    """

    azimuth_tolerance: float = 5.0
    """
    How far, in degrees, a measured direction of tilt may lie from the asked one.
    """

    def distance_to(
        self, measurement: flow_onset.FlowOnsetTiltMeasurement
    ) -> Optional[MeasurementDistance]:
        """
        :return: How far the measurement's fill and direction of tilt lie from the asked
            ones, or ``None`` when it is about another container or substance, or lies
            beyond a tolerance.
        """
        if (
            measurement.measured_for_container.mesh_digest != self.mesh_digest
            or measurement.has_material_regime is not self.material_regime
        ):
            return None
        fill_distance = abs(measurement.at_fill_fraction - self.fill_fraction)
        turn = abs(measurement.gravity_azimuth - self.gravity_azimuth) % 360.0
        azimuth_distance = min(turn, 360.0 - turn)
        if (
            fill_distance > self.fill_tolerance
            or azimuth_distance > self.azimuth_tolerance
        ):
            return None
        return MeasurementDistance(fill=fill_distance, azimuth=azimuth_distance)


# %% stored measurements


@dataclass(frozen=True)
class StoredFlowOnset:
    """
    A flow onset measurement as stored.
    """

    measurement: flow_onset.FlowOnsetTiltMeasurement
    """
    The measurement, with the container and setup it reaches.
    """

    iri: str
    """
    The IRI the measurement is stored under, which cites it.
    """

    @property
    def is_simulated(self) -> bool:
        """
        Whether the value was measured in a physics simulation rather than derived from
        the container's shape.
        """
        return isinstance(
            self.measurement.follows_setup, flow_onset.PhysicsSimulationSetup
        )


# %% the store


@dataclass
class MeasurementStore:
    """
    Physical measurements on disk.
    """

    directory: Path
    """
    The directory holding one Turtle file per measurement.
    """

    namespace: str = "http://www.ease-crc.org/ont/llm_reasoner/measurement/"
    """
    The namespace the stored measurements and the fragments they reach are named in.
    """

    acceptance: AcceptanceSuite = field(
        default_factory=lambda: AcceptanceSuite(
            rules=[RestrictionsHold(), RangesHold(), ShapesHold()]
        )
    )
    """
    The rules a measurement meets before it is stored.
    """

    def put(self, measurement: flow_onset.FlowOnsetTiltMeasurement) -> str:
        """
        Store a measurement, replacing one of the same container, substance, fill,
        direction of tilt and kind of setup.

        :return: The IRI the measurement is stored under.
        :raises InvalidMeasurement: When the measurement breaks its ontology.
        """
        fragments = ShapesHold.reachable([measurement])
        errors = [
            violation
            for violation in self.acceptance.violations(fragments)
            if violation.severity is Severity.ERROR
        ]
        if errors:
            raise InvalidMeasurement(violations=errors)
        name = self.name_of(measurement)
        names = {
            id(fragment): URIRef(
                f"{self.namespace}{name}"
                + ("" if position == 0 else f"/fragment_{position}")
            )
            for position, fragment in enumerate(fragments)
        }
        self.directory.mkdir(parents=True, exist_ok=True)
        GraphWriter(namespace=self.namespace).write_fragments(
            fragments, names
        ).serialize(self.directory / f"{name}.ttl", format="turtle")
        return str(names[id(measurement)])

    @staticmethod
    def name_of(measurement: flow_onset.FlowOnsetTiltMeasurement) -> str:
        """
        :return: The name a measurement is stored under, the same for every measurement
            of one case.
        """
        return "_".join(
            [
                "flow-onset",
                measurement.measured_for_container.mesh_digest,
                measurement.has_material_regime.name.lower(),
                f"fill-{measurement.at_fill_fraction:.3f}",
                f"azimuth-{measurement.gravity_azimuth:.1f}",
                type(measurement.follows_setup).__name__,
            ]
        )

    def flow_onsets(self) -> list[StoredFlowOnset]:
        """
        :return: Every stored flow onset measurement, in the order of their names.
        """
        if not self.directory.exists():
            return []
        stored = []
        for path in sorted(self.directory.glob("*.ttl")):
            read = GraphReader().read(Graph().parse(path, format="turtle"))
            (measurement,) = [
                fragment
                for fragment in read.fragments
                if isinstance(fragment, flow_onset.FlowOnsetTiltMeasurement)
            ]
            stored.append(
                StoredFlowOnset(
                    measurement=measurement, iri=read.identifiers[id(measurement)]
                )
            )
        return stored

    def flow_onset(self, query: FlowOnsetQuery) -> Optional[StoredFlowOnset]:
        """
        :return: The stored measurement answering the question: a simulated one before a
            derived one, and the nearest in fill and direction of tilt among those; none
            when no measurement lies within the question's tolerances.
        """
        answering = [
            (not stored.is_simulated, distance, stored.iri, stored)
            for stored in self.flow_onsets()
            if (distance := query.distance_to(stored.measurement)) is not None
        ]
        if not answering:
            return None
        return min(answering, key=lambda candidate: candidate[:3])[3]
