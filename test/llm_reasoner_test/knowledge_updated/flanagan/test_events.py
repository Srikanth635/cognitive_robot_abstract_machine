from __future__ import annotations

import pytest
from llm_reasoner.knowledge_updated.microworlds.flanagan.alignments import (
    DUL,
    subclass_of,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.events import (
    ContactEventType,
    ExternalTerminationEventType,
    ForceDynamicEventType,
    MechanicalEventType,
    PhaseBoundaryEventType,
    SetpointAttained,
    ThresholdCrossingEventType,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.parameters import (
    MechanicalThreshold,
    Setpoint,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.roles import RoleSlot
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.value_partitions import (
    ContactTransition,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.ontology import CORE, FLANAGAN
from llm_reasoner.knowledge_updated.ontological.documentation import (
    DocumentationReader,
)
from llm_reasoner.knowledge_updated.ontological.entities import KnowledgeEntity
from llm_reasoner.knowledge_updated.ontological.properties import PropertyMeta
from llm_reasoner.knowledge_updated.ontological.ranges import RangeResolver
from llm_reasoner.knowledge_updated.ontological.validation import (
    Check,
    OntologyValidator,
)
from llm_reasoner.knowledge_updated.ontological.vocabulary import PropertyKind

# %% fixtures

CONCRETE_EVENTS = (
    MechanicalEventType,
    ContactEventType,
    ThresholdCrossingEventType,
    ForceDynamicEventType,
    SetpointAttained,
    ExternalTerminationEventType,
)
"""
The event concepts a structure may instantiate directly.
"""

# %% taxonomy


class TestEventTaxonomy:
    """
    Boundary events are physical, setpoint-level or external, and never more than one.
    """

    @pytest.mark.parametrize("cls", (PhaseBoundaryEventType, *CONCRETE_EVENTS))
    def test_concept_is_a_core_entity(self, cls: type) -> None:
        described = FLANAGAN.find_concept(cls)

        assert issubclass(cls, KnowledgeEntity)
        assert described.module == CORE.name

    def test_only_the_grouping_event_is_abstract(self) -> None:
        assert FLANAGAN.find_concept(PhaseBoundaryEventType).abstract is True
        assert not any(FLANAGAN.find_concept(cls).abstract for cls in CONCRETE_EVENTS)

    @pytest.mark.parametrize(
        ("child", "parent"),
        [
            (MechanicalEventType, PhaseBoundaryEventType),
            (SetpointAttained, PhaseBoundaryEventType),
            (ExternalTerminationEventType, PhaseBoundaryEventType),
            (ContactEventType, MechanicalEventType),
            (ThresholdCrossingEventType, MechanicalEventType),
            (ForceDynamicEventType, ThresholdCrossingEventType),
        ],
    )
    def test_specialisations(self, child: type, parent: type) -> None:
        assert issubclass(child, parent)

    def test_reaching_a_setpoint_is_not_a_mechanical_event(self) -> None:
        assert not issubclass(SetpointAttained, MechanicalEventType)

    def test_no_separate_control_event_type_remains(self) -> None:
        assert (
            FLANAGAN.find_concept_by_iri(CORE.namespace.term("ControlEventType"))
            is None
        )

    def test_the_three_kinds_of_boundary_event_share_no_instance(self) -> None:
        assert FLANAGAN.find_concept(PhaseBoundaryEventType).axioms.disjoint_siblings

    def test_physical_events_may_be_contact_and_threshold_crossing_at_once(
        self,
    ) -> None:
        axioms = FLANAGAN.find_concept(ContactEventType).axioms
        report = OntologyValidator(FLANAGAN).validate(CORE.name)

        assert ThresholdCrossingEventType.__name__ in axioms.may_overlap_with
        assert report.of(Check.SIBLINGS_OVERLAP) == ()

    def test_boundary_events_are_aligned_as_event_types(self) -> None:
        assert FLANAGAN.find_concept(PhaseBoundaryEventType).alignments == (
            subclass_of(DUL, "EventType"),
        )


# %% properties


class TestEventProperties:
    """
    Each kind of event carries exactly the properties that make it that kind.
    """

    def test_every_event_says_whether_it_marks_failure(self) -> None:
        described = PropertyMeta.declared_on(PhaseBoundaryEventType)["is_failure_event"]

        assert described.kind is PropertyKind.ATTRIBUTE
        assert described.functional is True
        assert MechanicalEventType().is_failure_event is False

    @pytest.mark.parametrize("cls", [MechanicalEventType, SetpointAttained])
    def test_events_with_participants_name_their_roles(self, cls: type) -> None:
        described = PropertyMeta.declared_on(cls)["event_on_role"]

        assert described.kind is PropertyKind.RELATION
        assert described.minimum_count == 1
        assert RangeResolver().resolve(cls, "event_on_role").element_types == (
            RoleSlot,
        )

    def test_external_termination_has_no_participants(self) -> None:
        assert "event_on_role" not in PropertyMeta.declared_on(
            ExternalTerminationEventType
        )

    def test_contact_event_states_which_way_contact_changes(self) -> None:
        described = PropertyMeta.declared_on(ContactEventType)["contact_transition"]

        assert described.kind is PropertyKind.ATTRIBUTE
        assert described.functional is True
        assert described.minimum_count == 1
        assert RangeResolver().resolve(
            ContactEventType, "contact_transition"
        ).element_types == (ContactTransition,)

    def test_threshold_crossing_names_the_threshold_it_crosses(self) -> None:
        described = PropertyMeta.declared_on(ThresholdCrossingEventType)[
            "crosses_threshold"
        ]

        assert described.kind is PropertyKind.RELATION
        assert described.functional is True
        assert described.minimum_count == 1
        assert RangeResolver().resolve(
            ThresholdCrossingEventType, "crosses_threshold"
        ).element_types == (MechanicalThreshold,)

    def test_setpoint_attainment_names_the_setpoints_it_attains(self) -> None:
        described = PropertyMeta.declared_on(SetpointAttained)["attains_setpoint"]

        assert described.kind is PropertyKind.RELATION
        assert described.functional is False
        assert described.minimum_count == 1
        assert RangeResolver().resolve(
            SetpointAttained, "attains_setpoint"
        ).element_types == (Setpoint,)

    def test_external_termination_names_what_triggers_it(self) -> None:
        described = PropertyMeta.declared_on(ExternalTerminationEventType)[
            "triggered_by"
        ]

        assert described.kind is PropertyKind.REFERENCE
        assert described.functional is True
        assert described.minimum_count == 1

    def test_shared_participant_property_has_one_identifier(self) -> None:
        identifiers = {
            declaration.iri
            for declaration in FLANAGAN.property_declarations(CORE.name)
            if declaration.field_name == "event_on_role"
        }

        assert identifiers == {CORE.namespace.property_term("event_on_role")}


# %% documentation


class TestEventsAreDocumented:
    """
    Every event concept and property says what it means.
    """

    @pytest.mark.parametrize("cls", (PhaseBoundaryEventType, *CONCRETE_EVENTS))
    def test_concept_and_properties_are_documented(self, cls: type) -> None:
        reader = DocumentationReader()

        assert reader.read_class(cls) is not None
        assert reader.find_undocumented_fields(cls) == ()
