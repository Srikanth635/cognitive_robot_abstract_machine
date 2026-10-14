from __future__ import annotations

from enum import Enum
from typing import NamedTuple

import pytest
from llm_reasoner.knowledge_updated.microworlds.flanagan.alignments import (
    DUL,
    related_to,
    subclass_of,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.control import (
    ModelUpdate,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.conditions import (
    PhaseConditionType,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.events import (
    MechanicalEventType,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.models import (
    EstimateRepresentation,
    ForwardModel,
    InverseModel,
    ObjectRepresentation,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.parameters import (
    DynamicsEstimate,
    Estimate,
    MotorSetpoint,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.prediction import (
    SensoryEventPrediction,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.roles import RoleSlot
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.value_partitions import (
    ParameterValueForm,
    PhysicalDimension,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.object_dynamics.commands import (
    DesiredState,
    EfferenceCopy,
    MotorCommand,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.object_dynamics.composition import (
    InternalModelComposition,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.object_dynamics.dissociation import (
    RepresentationDissociation,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.object_dynamics.memory import (
    RetainedModelUpdate,
    SensorimotorMemory,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.object_dynamics.perturbation import (
    DynamicsPerturbation,
    DynamicsTransfer,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.object_dynamics.representations import (
    ContextualRepresentation,
    DynamicsModel,
    LimbDynamicsModel,
    LoadForceEstimate,
    ObjectDynamicsModel,
    PerceptualRepresentation,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.object_dynamics.structure import (
    ObjectDynamicsPhaseStructure,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.object_dynamics.value_partitions import (
    CoordinateFrame,
    MemoryPersistence,
    PerturbationKind,
    RepresentationPurpose,
    TransferOutcome,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.ontology import (
    CORE,
    FLANAGAN,
    OBJECT_DYNAMICS,
)
from llm_reasoner.knowledge_updated.ontological.documentation import (
    DocumentationReader,
)
from llm_reasoner.knowledge_updated.ontological.entities import KnowledgeEntity
from llm_reasoner.knowledge_updated.ontological.properties import PropertyMeta
from llm_reasoner.knowledge_updated.ontological.ranges import RangeResolver
from llm_reasoner.knowledge_updated.ontological.vocabulary import (
    EvidenceStatus,
    PropertyKind,
)

# %% the specification

ENUMERATIONS = (
    CoordinateFrame,
    RepresentationPurpose,
    MemoryPersistence,
    PerturbationKind,
    TransferOutcome,
)
"""
Every value partition of the module.
"""


class Holds(NamedTuple):
    """
    What one property of a concept must hold.
    """

    kind: PropertyKind
    range: type
    functional: bool
    minimum: int


def relation(range: type, minimum: int = 0, functional: bool = False) -> Holds:
    """
    Describe a property that relates to other concepts.
    """
    return Holds(PropertyKind.RELATION, range, functional, minimum)


def attribute(range: type, minimum: int = 1) -> Holds:
    """
    Describe a functional property that holds a value of a partition.
    """
    return Holds(PropertyKind.ATTRIBUTE, range, True, minimum)


PROPERTIES: dict[type, dict[str, Holds]] = {
    DesiredState: {"desired_condition": relation(PhaseConditionType, 1)},
    MotorCommand: {
        "pursues_state": relation(DesiredState, 1, functional=True),
        "generated_by": relation(InverseModel, 1, functional=True),
        "commands_setpoint": relation(MotorSetpoint, 1),
    },
    EfferenceCopy: {"copies_command": relation(MotorCommand, 1, functional=True)},
    ContextualRepresentation: {
        "holds_estimate": relation(Estimate, 1),
        "coordinate_frame": attribute(CoordinateFrame),
        "purpose": attribute(RepresentationPurpose),
        "represents_role": relation(RoleSlot, 1),
    },
    DynamicsModel: {"uses_efference_copy": relation(EfferenceCopy, 1)},
    SensorimotorMemory: {
        "stores_representation": relation(ContextualRepresentation, 1),
        "memory_persistence": attribute(MemoryPersistence),
    },
    RetainedModelUpdate: {
        "persists_in": relation(SensorimotorMemory, 1, functional=True)
    },
    InternalModelComposition: {
        "efference_copy": relation(EfferenceCopy, 1, functional=True),
        "dynamics_model": relation(DynamicsModel, 1, functional=True),
        "forward_model": relation(ForwardModel, 1, functional=True),
        "predicted_mechanical_consequence": relation(MechanicalEventType, 1),
        "predicted_sensory_consequence": relation(SensoryEventPrediction, 1),
    },
    DynamicsPerturbation: {
        "perturbation_kind": attribute(PerturbationKind),
        "affects_model": relation(DynamicsModel, 1, functional=True),
        "coordinate_frame": attribute(CoordinateFrame),
        "matched_with": relation(DynamicsPerturbation),
    },
    DynamicsTransfer: {
        "from_perturbation": relation(DynamicsPerturbation, 1, functional=True),
        "to_perturbation": relation(DynamicsPerturbation, 1, functional=True),
        "transfer_outcome": attribute(TransferOutcome),
    },
    RepresentationDissociation: {
        "sensorimotor_representation": relation(DynamicsModel, 1, functional=True),
        "perceptual_representation": relation(
            PerceptualRepresentation, 1, functional=True
        ),
        "sensorimotor_estimate": relation(Estimate, 1, functional=True),
        "perceptual_estimate": relation(Estimate, 1, functional=True),
    },
}
"""
The properties each concept of the module writes in its own body.
"""

STRUCTURE_COLLECTIONS: dict[str, type] = {
    "defines_desired_state": DesiredState,
    "defines_motor_command": MotorCommand,
    "defines_efference_copy": EfferenceCopy,
    "defines_internal_model_composition": InternalModelComposition,
    "defines_sensorimotor_memory": SensorimotorMemory,
    "defines_dynamics_perturbation": DynamicsPerturbation,
    "defines_dynamics_transfer": DynamicsTransfer,
    "defines_representation_dissociation": RepresentationDissociation,
}
"""
The collections a structure adds to the ones the core defines, beside what each holds.
"""

CONCEPTS = (
    DesiredState,
    MotorCommand,
    EfferenceCopy,
    ContextualRepresentation,
    DynamicsModel,
    ObjectDynamicsModel,
    LimbDynamicsModel,
    PerceptualRepresentation,
    LoadForceEstimate,
    SensorimotorMemory,
    RetainedModelUpdate,
    InternalModelComposition,
    DynamicsPerturbation,
    DynamicsTransfer,
    RepresentationDissociation,
    ObjectDynamicsPhaseStructure,
)
"""
Every concept of the module.
"""


def written_by(cls: type) -> set[str]:
    """
    Return the names of the properties a concept writes in its own body.
    """
    return {
        declaration.field_name
        for declaration in FLANAGAN.property_declarations(OBJECT_DYNAMICS.name)
        if declaration.owner is cls
    }


# %% value partitions


class TestPartitionsAreDeclared:
    """
    Each enumeration is recorded as a closed set of alternatives of this module.
    """

    @pytest.mark.parametrize("enumeration", ENUMERATIONS)
    def test_enumeration_is_declared_in_the_module(
        self, enumeration: type[Enum]
    ) -> None:
        partition = FLANAGAN.find_value_partition(enumeration)

        assert partition is not None
        assert partition.module == OBJECT_DYNAMICS.name
        assert partition.iri == OBJECT_DYNAMICS.namespace.term(enumeration.__name__)

    def test_module_declares_no_other_partitions(self) -> None:
        declared = {
            partition.enumeration
            for partition in FLANAGAN.value_partitions(OBJECT_DYNAMICS.name)
        }

        assert declared == set(ENUMERATIONS)

    @pytest.mark.parametrize("enumeration", ENUMERATIONS)
    def test_enumeration_and_its_members_are_documented(
        self, enumeration: type[Enum]
    ) -> None:
        reader = DocumentationReader()

        assert reader.read_class(enumeration) is not None
        assert reader.find_undocumented_members(enumeration) == ()

    def test_a_transfer_either_generalizes_or_does_not(self) -> None:
        assert {member.name for member in TransferOutcome} == {
            "GENERALIZES",
            "DOES_NOT_GENERALIZE",
        }

    def test_representations_serve_control_or_judgment(self) -> None:
        assert {member.name for member in RepresentationPurpose} == {
            "SENSORIMOTOR_CONTROL",
            "PERCEPTUAL_JUDGMENT",
        }


# %% taxonomy


class TestObjectDynamicsTaxonomy:
    """
    The concepts specialise the core and are declared in this module.
    """

    @pytest.mark.parametrize("cls", CONCEPTS)
    def test_concept_is_an_entity_of_this_module(self, cls: type) -> None:
        assert issubclass(cls, KnowledgeEntity)
        assert FLANAGAN.find_concept(cls).module == OBJECT_DYNAMICS.name

    def test_module_declares_no_other_concepts(self) -> None:
        declared = {meta.cls for meta in FLANAGAN.concepts(OBJECT_DYNAMICS.name)}

        assert declared == set(CONCEPTS)

    @pytest.mark.parametrize(
        ("child", "parent"),
        [
            (ContextualRepresentation, EstimateRepresentation),
            (DynamicsModel, ContextualRepresentation),
            (PerceptualRepresentation, ContextualRepresentation),
            (ObjectDynamicsModel, DynamicsModel),
            (ObjectDynamicsModel, ObjectRepresentation),
            (LimbDynamicsModel, DynamicsModel),
            (LoadForceEstimate, DynamicsEstimate),
            (RetainedModelUpdate, ModelUpdate),
        ],
    )
    def test_specialisations(self, child: type, parent: type) -> None:
        assert issubclass(child, parent)

    def test_only_the_grouping_concepts_are_abstract(self) -> None:
        abstract = {
            meta.cls
            for meta in FLANAGAN.concepts(OBJECT_DYNAMICS.name)
            if meta.abstract
        }

        assert abstract == {ContextualRepresentation, DynamicsModel}

    def test_object_and_limb_dynamics_are_different_models(self) -> None:
        assert FLANAGAN.find_concept(DynamicsModel).axioms.disjoint_siblings

    def test_a_dynamics_model_and_a_perceptual_representation_are_different(
        self,
    ) -> None:
        assert FLANAGAN.find_concept(ContextualRepresentation).axioms.disjoint_siblings

    def test_an_object_dynamics_model_is_also_an_object_representation(self) -> None:
        axioms = FLANAGAN.find_concept(ContextualRepresentation).axioms

        assert ObjectRepresentation.__name__ in axioms.may_overlap_with


# %% what a concept fixes


class TestFixedValues:
    """
    A concept states the values its kind determines.
    """

    @staticmethod
    def fixed(cls: type) -> dict[str, object]:
        return {
            restriction.property_name: restriction.value
            for restriction in FLANAGAN.find_concept(cls).axioms.fixed_values
        }

    def test_dynamics_models_serve_sensorimotor_control(self) -> None:
        assert self.fixed(DynamicsModel) == {
            "purpose": RepresentationPurpose.SENSORIMOTOR_CONTROL
        }

    def test_perceptual_representations_serve_perceptual_judgment(self) -> None:
        assert self.fixed(PerceptualRepresentation) == {
            "purpose": RepresentationPurpose.PERCEPTUAL_JUDGMENT
        }

    def test_a_load_force_estimate_is_a_quantity_of_force(self) -> None:
        assert self.fixed(LoadForceEstimate) == {
            "value_form": ParameterValueForm.QUANTITATIVE,
            "dimension": PhysicalDimension.FORCE,
        }


# %% properties


class TestProperties:
    """
    Each concept writes the properties its specification lists, and no others.
    """

    @pytest.mark.parametrize("cls", list(PROPERTIES))
    def test_concept_declares_exactly_its_properties(self, cls: type) -> None:
        assert written_by(cls) == set(PROPERTIES[cls])

    @pytest.mark.parametrize(
        ("cls", "name"),
        [(cls, name) for cls, held in PROPERTIES.items() for name in held],
    )
    def test_property_holds_what_it_is_specified_to(self, cls: type, name: str) -> None:
        expected = PROPERTIES[cls][name]
        declared = PropertyMeta.declared_on(cls)[name]
        resolved = RangeResolver().resolve(cls, name)

        assert declared.kind is expected.kind
        assert declared.functional is expected.functional
        assert declared.minimum_count == expected.minimum
        assert resolved.element_types == (expected.range,)

    def test_a_single_valued_link_may_be_left_out_of_the_type(self) -> None:
        resolved = RangeResolver().resolve(MotorCommand, "pursues_state")

        assert resolved.optional is True

    def test_the_composition_names_only_what_it_adds(self) -> None:
        assert not {"desired_state", "inverse_model", "motor_command"} & written_by(
            InternalModelComposition
        )


class TestStructureCollections:
    """
    A structure owns each fragment it declares.
    """

    def test_structure_adds_exactly_the_specified_collections(self) -> None:
        assert written_by(ObjectDynamicsPhaseStructure) == {
            *STRUCTURE_COLLECTIONS,
            "defines_internal_model",
        }

    @pytest.mark.parametrize("name", list(STRUCTURE_COLLECTIONS))
    def test_collection_owns_the_fragments_it_holds(self, name: str) -> None:
        declared = PropertyMeta.declared_on(ObjectDynamicsPhaseStructure)[name]
        resolved = RangeResolver().resolve(ObjectDynamicsPhaseStructure, name)

        assert declared.owns is True
        assert declared.functional is False
        assert resolved.element_types == (STRUCTURE_COLLECTIONS[name],)

    def test_a_structure_declares_at_least_one_internal_model(self) -> None:
        declared = PropertyMeta.declared_on(ObjectDynamicsPhaseStructure)[
            "defines_internal_model"
        ]

        assert declared.minimum_count == 1
        assert declared.owns is True

    def test_narrowing_the_internal_models_keeps_the_core_identifier(self) -> None:
        narrowed = next(
            declaration
            for declaration in FLANAGAN.property_declarations(OBJECT_DYNAMICS.name)
            if declaration.owner is ObjectDynamicsPhaseStructure
            and declaration.field_name == "defines_internal_model"
        )

        assert narrowed.iri == CORE.namespace.property_term("defines_internal_model")

    def test_retained_updates_are_declared_with_the_core_model_updates(self) -> None:
        assert "defines_model_update" not in written_by(ObjectDynamicsPhaseStructure)


# %% alignments


class TestAlignments:
    """
    Each concept is placed in the upper ontology only as far as its definition allows.
    """

    @staticmethod
    def alignments(cls: type) -> tuple:
        return FLANAGAN.find_concept(cls).alignments

    def test_a_perturbation_a_transfer_and_a_dissociation_are_situations(self) -> None:
        for cls in (DynamicsPerturbation, DynamicsTransfer, RepresentationDissociation):
            assert self.alignments(cls) == (subclass_of(DUL, "Situation"),)

    def test_a_copy_of_a_command_is_information(self) -> None:
        assert self.alignments(EfferenceCopy) == (
            subclass_of(DUL, "InformationObject"),
        )

    def test_a_desired_state_is_only_related_to_a_situation(self) -> None:
        assert self.alignments(DesiredState) == (related_to(DUL, "Situation"),)

    def test_a_command_a_memory_and_a_composition_are_only_related_to_descriptions(
        self,
    ) -> None:
        for cls in (MotorCommand, SensorimotorMemory, InternalModelComposition):
            assert self.alignments(cls) == (related_to(DUL, "Description"),)


# %% provenance

CLASSIFIED: dict[type, tuple[EvidenceStatus, tuple[str, ...]]] = {
    DesiredState: (EvidenceStatus.SUPPORTED_INTERPRETATION, ("FJ-002",)),
    MotorCommand: (EvidenceStatus.SUPPORTED_INTERPRETATION, ("FJ-002",)),
    EfferenceCopy: (EvidenceStatus.SUPPORTED_INTERPRETATION, ("FJ-008",)),
    ContextualRepresentation: (
        EvidenceStatus.ENGINEERING_EXTENSION,
        ("FJ-029", "FJ-031"),
    ),
    DynamicsModel: (EvidenceStatus.SUPPORTED_INTERPRETATION, ("FJ-027",)),
    ObjectDynamicsModel: (
        EvidenceStatus.SUPPORTED_INTERPRETATION,
        ("FJ-008", "FJ-027"),
    ),
    LimbDynamicsModel: (EvidenceStatus.SUPPORTED_INTERPRETATION, ("FJ-027",)),
    PerceptualRepresentation: (EvidenceStatus.EMPIRICAL_FINDING, ("FJ-031",)),
    LoadForceEstimate: (EvidenceStatus.EMPIRICAL_FINDING, ("FJ-025",)),
    SensorimotorMemory: (EvidenceStatus.EMPIRICAL_FINDING, ("FJ-009", "FJ-028")),
    RetainedModelUpdate: (EvidenceStatus.EMPIRICAL_FINDING, ("FJ-010", "FJ-011")),
    InternalModelComposition: (
        EvidenceStatus.SUPPORTED_INTERPRETATION,
        ("FJ-002", "FJ-008"),
    ),
    DynamicsPerturbation: (EvidenceStatus.EMPIRICAL_FINDING, ("FJ-026",)),
    DynamicsTransfer: (EvidenceStatus.EMPIRICAL_FINDING, ("FJ-030",)),
    RepresentationDissociation: (EvidenceStatus.EMPIRICAL_FINDING, ("FJ-031",)),
    ObjectDynamicsPhaseStructure: (EvidenceStatus.ENGINEERING_EXTENSION, ()),
}
"""
The status and claims the scientific contract gives each concept.
"""


class TestProvenance:
    """
    Each concept records how well the scientific contract supports it.
    """

    def test_every_concept_of_the_module_is_classified(self) -> None:
        assert set(CLASSIFIED) == set(CONCEPTS)

    @pytest.mark.parametrize("cls", CONCEPTS)
    def test_concept_records_the_status_and_claims_of_the_contract(
        self, cls: type
    ) -> None:
        status, claims = CLASSIFIED[cls]
        meta = FLANAGAN.find_concept(cls)

        assert meta.status is status
        assert meta.claims == claims


# %% documentation


class TestObjectDynamicsIsDocumented:
    """
    Every concept and property says what it means.
    """

    @pytest.mark.parametrize("cls", CONCEPTS)
    def test_concept_and_properties_are_documented(self, cls: type) -> None:
        reader = DocumentationReader()

        assert reader.read_class(cls) is not None
        assert reader.find_undocumented_fields(cls) == ()
