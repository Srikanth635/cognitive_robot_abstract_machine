"""
Tests that the direction in which something changes is one term of the action core: an
open value partition whose larger and smaller the core declares, which the failures and
the predictive manipulation core add their own directions to, and which a shape closes,
for each property taking a direction, to the members that property takes.
"""

from __future__ import annotations

import pytest
from rdflib import OWL, RDF, RDFS, Graph, Namespace, URIRef
from typing_extensions import Optional, get_type_hints

from llm_reasoner.knowledge.generation.schema import GenerationSchema
from llm_reasoner.knowledge.ontologies import (
    DUL_NAMESPACE,
    KNOWLEDGE_ONTOLOGY_MODULES,
    ActionPerspectiveFile,
    PredictiveManipulationOntologyFile,
)
from llm_reasoner.knowledge.ontologies.action_perspectives import (
    action_core,
    action_frame,
    failures,
)
from llm_reasoner.knowledge.ontologies.predictive_manipulation import (
    core,
    pouring,
    prehension,
)
from llm_reasoner.knowledge.ontologies.terms import DescriptionEntity, PropertyTerm
from llm_reasoner.knowledge.ontologies.vocabulary import Vocabulary
from llm_reasoner.knowledge.representation.rdf import GraphReader

DUL = Namespace(DUL_NAMESPACE)
"""
The terms of DOLCE+DnS Ultralite.
"""

CHANGE = action_core.ChangeDirection
"""
The direction in which something changes.
"""

DIRECTED_FIELDS = [
    (
        failures.FailureStereotype,
        failures.ParameterChange,
        "has_direction",
        {
            CHANGE.INCREASES,
            CHANGE.DECREASES,
            CHANGE.TOWARDS_TARGET,
            CHANGE.AWAY_FROM_TARGET,
            CHANGE.TO_ALTERNATIVE,
        },
    ),
    (
        pouring.PouringPhaseStructure,
        core.CorrectiveAction,
        "has_correction_direction",
        {
            CHANGE.INCREASES,
            CHANGE.DECREASES,
            CHANGE.HOLD_SETPOINT,
            CHANGE.REVERSE_CHANGE,
            CHANGE.STOP_CHANGE,
        },
    ),
    (
        pouring.PouringPhaseStructure,
        core.ModelUpdate,
        "has_update_direction",
        {CHANGE.INCREASES, CHANGE.DECREASES, CHANGE.REPLACE_ESTIMATE},
    ),
    (
        pouring.PouringPhaseStructure,
        prehension.AnticipatoryGripAdjustment,
        "has_adjustment_direction",
        {
            CHANGE.INCREASES,
            CHANGE.DECREASES,
            CHANGE.HOLD_SETPOINT,
            CHANGE.REVERSE_CHANGE,
            CHANGE.STOP_CHANGE,
        },
    ),
]
"""
Each field taking a direction, with a root whose answers state it, its class, its name
and the members it may take.
"""

# %% fixtures


@pytest.fixture(scope="module")
def graph() -> Graph:
    """
    Every module of both families in one graph.
    """
    union = Graph()
    for module in KNOWLEDGE_ONTOLOGY_MODULES:
        union.parse(module.path, format="turtle")
    return union


# %% the term


class TestTheTerm:
    def test_the_action_core_declares_it_as_a_dul_concept(self, graph):
        assert (URIRef(CHANGE.INCREASES.value), RDF.type, OWL.NamedIndividual) in graph
        assert (
            URIRef(ActionPerspectiveFile.ACTION_CORE.namespace + "ChangeDirection"),
            RDFS.subClassOf,
            DUL.Concept,
        ) in graph

    def test_the_action_frame_uses_it_as_it_is(self):
        assert "ChangeDirection" not in vars(action_frame)
        assert (
            get_type_hints(action_frame.AdditionalConsequence)["changes_in_direction"]
            == Optional[CHANGE]
        )

    @pytest.mark.parametrize(
        "module, directions",
        [
            (ActionPerspectiveFile.ACTION_CORE, {"INCREASES", "DECREASES"}),
            (
                ActionPerspectiveFile.FAILURES,
                {"TOWARDS_TARGET", "AWAY_FROM_TARGET", "TO_ALTERNATIVE"},
            ),
            (
                PredictiveManipulationOntologyFile.CORE,
                {"HOLD_SETPOINT", "REVERSE_CHANGE", "STOP_CHANGE", "REPLACE_ESTIMATE"},
            ),
        ],
    )
    def test_each_module_declares_its_own_directions(self, module, directions):
        assert {
            member.name for member in CHANGE if member.declaring_module is module
        } == directions

    @pytest.mark.parametrize("module", [failures, core])
    def test_no_module_keeps_a_direction_class_of_its_own(self, module):
        assert {
            name
            for name in vars(module)
            if name.endswith("Direction") and isinstance(vars(module)[name], type)
        } == set()


# %% the properties taking a direction


class TestClosedProperties:
    @pytest.mark.parametrize("root, owner, name, members", DIRECTED_FIELDS)
    def test_every_property_taking_a_direction_ranges_over_it(
        self, graph, root, owner, name, members
    ):
        assert graph.value(
            URIRef(PropertyTerm.of_field(owner, name).iri), RDFS.range
        ) == URIRef(ActionPerspectiveFile.ACTION_CORE.namespace + "ChangeDirection")

    @pytest.mark.parametrize("root, owner, name, members", DIRECTED_FIELDS)
    def test_a_model_is_offered_only_the_members_the_property_takes(
        self, root, owner, name, members
    ):
        field_schema = (
            GenerationSchema.for_root(root).get(owner.__name__).field_named(name)
        )
        assert set(field_schema.options) == {member.name for member in members}


# %% reading a direction


def stated(owner: type[DescriptionEntity], name: str, member: CHANGE) -> Graph:
    """
    :return: A fragment of a class stating a direction for one of its fields.
    """
    graph = Graph()
    individual = URIRef("http://example.org/hypothesis#directed")
    graph.add((individual, RDF.type, URIRef(owner.term.iri)))
    graph.add(
        (
            individual,
            URIRef(PropertyTerm.of_field(owner, name).iri),
            URIRef(member.value),
        )
    )
    return graph


class TestReadingADirection:
    def test_a_stated_direction_is_the_cores_member(self):
        (correction,) = (
            GraphReader()
            .read(
                stated(
                    core.CorrectiveAction, "has_correction_direction", CHANGE.INCREASES
                )
            )
            .fragments
        )
        assert correction.has_correction_direction is CHANGE.INCREASES

    def test_an_entailed_direction_is_the_cores_member(self):
        reduction = prehension.LoadRateReduction()
        Vocabulary().fill_entailed_values(reduction)
        assert reduction.has_correction_direction is CHANGE.DECREASES
