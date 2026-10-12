"""
Tests for the action designator the Designator tab shows: coraplex's action slots
followed by the generated hypothesis, on the checked-in sample hypothesis.
"""

import pytest

krrood = pytest.importorskip("krrood", reason="EQL requires krrood")

from llm_reasoner.knowledge.ontologies.terms import DescriptionEntity  # noqa: E402

from cram_viz import designator_knowledge, hypothesis_knowledge  # noqa: E402
from cram_viz.designator_description import (  # noqa: E402
    DesignatorDescriber,
    DesignatorDescription,
    ValueKind,
)
from cram_viz.hypothesis_knowledge.ontology_backing import OntologyBacking  # noqa: E402

# %% helpers


@pytest.fixture
def view():
    return hypothesis_knowledge.sample_view()


@pytest.fixture
def slots(view):
    return designator_knowledge.reflect_pouring_action([], view)


@pytest.fixture
def action(slots, view):
    return DesignatorDescriber(view=view).describe(slots)


def keys(description):
    return [entry.key for entry in description.properties]


def property_values(description, key):
    (entry,) = [entry for entry in description.properties if entry.key == key]
    return entry.values


def descriptions(description):
    """
    Every description in the tree, the given one included.
    """
    found = [description]
    for entry in description.properties:
        for value in entry.values:
            if isinstance(value, DesignatorDescription):
                found.extend(descriptions(value))
    return found


def references(description):
    return [
        value
        for nested in descriptions(description)
        for entry in nested.properties
        for value in entry.values
        if not isinstance(value, DesignatorDescription)
        and value.kind is ValueKind.REFERENCE
    ]


def structure(action):
    (root,) = property_values(action, "hypothesis")
    return root


# %% the coraplex action


class TestActionSlots:
    def test_action_is_typed_after_the_coraplex_action(self, action, slots):
        assert action.kind == "action"
        assert action.type == "pouring"
        assert slots.designator_name in action.note

    def test_every_slot_of_the_action_comes_first_in_declaration_order(
        self, action, slots
    ):
        assert keys(action)[: len(slots.slots)] == [
            slot.lisp_keyword.removeprefix(":") for slot in slots.slots
        ]

    def test_open_slot_is_a_variable_and_a_defaulted_slot_its_default(
        self, action, slots
    ):
        for slot in slots.slots:
            (value,) = property_values(action, slot.lisp_keyword.removeprefix(":"))
            assert value.text == slot.lisp_value
            if slot.lisp_value.startswith("?"):
                assert value.kind is ValueKind.VARIABLE

    def test_container_slot_refers_to_the_role_filling_it(self, action, view):
        (value,) = property_values(action, "target-container")
        (filler,) = [
            role
            for role in view.root.defines_role
            if role.term.iri.endswith("#TargetReceptacle")
        ]
        assert value.reference == view.citation_of(filler).fragment_name

    def test_without_a_hypothesis_only_the_slots_are_described(self, slots):
        action = DesignatorDescriber(view=None).describe(slots)
        assert keys(action) == [
            slot.lisp_keyword.removeprefix(":") for slot in slots.slots
        ]


# %% the generated hypothesis


class TestHypothesis:
    def test_hypothesis_describes_the_root_structure(self, action, view):
        assert structure(action).type == OntologyBacking().prefixed(
            type(view.root).term.iri
        )
        assert property_values(action, "instruction")[0].data == (
            view.record.instruction
        )

    def test_roles_come_before_the_phases(self, action):
        order = keys(structure(action))
        assert order.index("defines-role") < order.index("defines-phase")

    def test_phases_are_listed_in_the_order_they_run(self, action, view):
        assert [
            value.identifier
            for value in property_values(structure(action), "defines-phase")
        ] == [view.citation_of(phase).fragment_name for phase in view.phases()]

    def test_every_fragment_is_described_exactly_once(self, action, view):
        described = [nested.identifier for nested in descriptions(structure(action))]
        assert sorted(described) == sorted(
            view.citation_of(fragment).fragment_name for fragment in view.fragments
        )

    def test_every_reference_names_a_described_fragment(self, action):
        described = {nested.identifier for nested in descriptions(action)}
        assert {value.reference for value in references(action)} <= described

    def test_description_is_of_its_top_ontology_family(self, action, view):
        lift = next(
            nested
            for nested in descriptions(structure(action))
            if nested.type == "prehension:LiftPhase"
        )
        assert lift.kind == "action-phase"
        assert lift.article == "an"

    def test_enumerated_value_is_its_ontology_term(self, action, view):
        phase = view.phases()[0]
        described = next(
            nested
            for nested in descriptions(structure(action))
            if nested.identifier == view.citation_of(phase).fragment_name
        )
        (regime,) = property_values(described, "has-coupling-regime")
        assert regime.kind is ValueKind.TERM
        assert regime.text == OntologyBacking().prefixed(
            phase.has_coupling_regime.value
        )

    def test_list_valued_field_is_marked_as_holding_many_values(self, action):
        (entry,) = [
            entry
            for entry in structure(action).properties
            if entry.key == "defines-phase"
        ]
        assert entry.many
        assert not [entry for entry in action.properties if entry.key == "tilt-angle"][
            0
        ].many

    def test_payload_keeps_the_tree(self, action):
        payload = action.payload()
        assert payload["kind"] == "action"
        assert [entry["key"] for entry in payload["properties"]] == keys(action)

    def test_no_value_is_a_bare_fragment(self, action):
        assert not any(
            isinstance(value, DescriptionEntity)
            for nested in descriptions(action)
            for entry in nested.properties
            for value in entry.values
        )
