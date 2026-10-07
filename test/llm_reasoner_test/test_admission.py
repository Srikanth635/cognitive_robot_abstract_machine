"""
Tests of admitting a generated hypothesis: writing it as RDF and reading it back,
checking it against the ontology in the open world, locating contradictions, and
recording and storing it.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import pytest
from rdflib import RDF, RDFS, Graph, URIRef
from rdflib.compare import isomorphic

from krrood.entity_query_language.factories import the

from llm_reasoner.acceptance.rules import AcceptanceSuite
from llm_reasoner.admission.consistency import (
    ConsistencyCheck,
    ContradictionLocaliser,
    OpenWorldCheck,
)
from llm_reasoner.admission.hermit import HermitCheck
from llm_reasoner.admission.rdf import GraphReader, GraphWriter
from llm_reasoner.admission.record import (
    ConsistencyVerdict,
    HypothesisRecord,
    current_module_versions,
    qualified_name,
)
from llm_reasoner.admission.store import HypothesisStore
from llm_reasoner.clients.base import Exchange, ScriptedClient
from llm_reasoner.generation.backend import HypothesisBackend
from llm_reasoner.generation.schema import GenerationSchema
from llm_reasoner.generation.wire import DecodedHypothesis, FragmentDecoder
from llm_reasoner.ontologies import (
    CRAM_AICOR_NAMESPACE,
    LinkedOntologyVariable,
    PredictiveManipulationOntologyFile,
)
from llm_reasoner.ontologies.predictive_manipulation import core, pouring, prehension
from llm_reasoner.ontologies.terms import ConceptReference, PropertyTerm

from .test_generation import GenerationCase

CASES = Path(__file__).parent / "predictive_manipulation_ontology_cases"
"""
The directory of the ontology's own reasoning cases.
"""

HYPOTHESIS_NAMESPACE = "http://example.org/hypothesis/test#"
"""
The namespace the tests write fragments in.
"""

MICROWORLD_FILES = [
    module_file.path for module_file in PredictiveManipulationOntologyFile
]
"""
The microworld's module files, without the linked ontologies.
"""

# %% helpers


def chained_structure() -> DecodedHypothesis:
    """
    Return the decoded scripted structure whose transition leads from tilting to
    regulating the flow.
    """
    decoder = FragmentDecoder(GenerationSchema.for_root(core.ActionPhaseStructure))
    return decoder.decode(GenerationCase.CHAINED_STRUCTURE.text).hypothesis


def written(hypothesis: DecodedHypothesis) -> Graph:
    """
    Return a hypothesis written as RDF in the tests' namespace.
    """
    return GraphWriter(namespace=HYPOTHESIS_NAMESPACE).write(hypothesis)


# %% writing and reading


class TestGraphWriter:
    """
    Fragments are written with the IRIs of the ontology terms they stand for.
    """

    def test_fragment_is_typed_by_its_ontology_class(self):
        graph = written(chained_structure())
        assert (
            URIRef(HYPOTHESIS_NAMESPACE + "tilt"),
            RDF.type,
            URIRef(pouring.TiltToFlowPhase.term.iri),
        ) in graph

    def test_link_is_written_with_its_property(self):
        graph = written(chained_structure())
        assert (
            URIRef(HYPOTHESIS_NAMESPACE + "tilt_to_regulate"),
            URIRef(PropertyTerm.of_field(core.PhaseTransition, "from_phase").iri),
            URIRef(HYPOTHESIS_NAMESPACE + "tilt"),
        ) in graph

    def test_enumeration_member_is_written_as_its_individual(self):
        graph = written(chained_structure())
        assert (
            URIRef(HYPOTHESIS_NAMESPACE + "tilt"),
            URIRef(PropertyTerm.of_field(core.ActionPhase, "has_coupling_regime").iri),
            URIRef(core.CouplingRegime.COUPLED.value),
        ) in graph

    def test_inferred_super_property_is_not_written(self):
        graph = written(chained_structure())
        defines_component = URIRef(
            PropertyTerm.of_field(core.ActionPhaseStructure, "defines_component").iri
        )
        assert list(graph.triples((None, defines_component, None))) == []


class TestGraphReader:
    """
    RDF typed with ontology classes is read back into vocabulary instances.
    """

    def test_written_hypothesis_reads_back_with_its_links(self):
        fragments = GraphReader().read(written(chained_structure())).fragments
        (structure,) = [
            fragment
            for fragment in fragments
            if isinstance(fragment, core.ActionPhaseStructure)
        ]
        (transition,) = structure.defines_transition
        assert isinstance(transition.from_phase, pouring.TiltToFlowPhase)
        assert structure.decomposes_task == ConceptReference(
            iri=CRAM_AICOR_NAMESPACE + "Pouring"
        )

    def test_individual_of_several_classes_becomes_the_most_specific(self):
        graph = Graph().parse(CASES / "pouring_prototype.ttl", format="turtle")
        read = GraphReader().read(graph)
        (structure,) = [
            fragment
            for fragment in read.fragments
            if isinstance(fragment, core.ActionPhaseStructure)
        ]
        assert type(structure) is pouring.PouringPhaseStructure

    def test_statement_outside_the_vocabulary_is_reported(self):
        graph = written(chained_structure())
        graph.add(
            (
                URIRef(HYPOTHESIS_NAMESPACE + "tilt"),
                URIRef(HYPOTHESIS_NAMESPACE + "unknownProperty"),
                URIRef(HYPOTHESIS_NAMESPACE + "regulate"),
            )
        )
        assert [
            statement.predicate
            for statement in GraphReader().read(graph).unread_statements
        ] == [URIRef(HYPOTHESIS_NAMESPACE + "unknownProperty")]

    def test_label_is_read_back(self):
        decoder = FragmentDecoder(GenerationSchema.for_root(core.ActionPhaseStructure))
        looping = decoder.decode(GenerationCase.LOOPING_STRUCTURE.text).hypothesis
        (tilt,) = [
            fragment
            for fragment in GraphReader().read(written(looping)).fragments
            if isinstance(fragment, pouring.TiltToFlowPhase)
        ]
        (written_tilt,) = [
            fragment
            for fragment in looping.fragments
            if isinstance(fragment, pouring.TiltToFlowPhase)
        ]
        assert tilt.label == written_tilt.label

    def test_values_the_class_entails_are_read_although_not_stated(self):
        graph = Graph()
        graph.add(
            (
                URIRef(HYPOTHESIS_NAMESPACE + "contact"),
                RDF.type,
                URIRef(prehension.ContactEstablishmentPhase.term.iri),
            )
        )
        (contact,) = GraphReader().read(graph).fragments
        assert contact.realizes_action_category == [
            ConceptReference(iri=CRAM_AICOR_NAMESPACE + "Grasping")
        ]


# %% locating contradictions


@dataclass
class PairConflict(ConsistencyCheck):
    """
    A stand-in reasoner that finds a graph inconsistent exactly when it states something
    about both of two individuals.
    """

    first: URIRef
    """
    One individual of the conflicting pair.
    """

    second: URIRef
    """
    The other individual of the conflicting pair.
    """

    calls: int = 0
    """
    How many graphs the stand-in judged.
    """

    def is_consistent(self, graph: Graph) -> bool:
        self.calls += 1
        subjects = set(graph.subjects())
        return not {self.first, self.second} <= subjects


class TestContradictionLocaliser:
    """
    A contradiction is narrowed down to a smallest set of fragments that cause it.
    """

    def test_consistent_graph_has_no_conflict(self):
        check = PairConflict(
            first=URIRef(HYPOTHESIS_NAMESPACE + "tilt"),
            second=URIRef(HYPOTHESIS_NAMESPACE + "missing"),
        )
        assert (
            ContradictionLocaliser(check).conflict(written(chained_structure())) == []
        )

    def test_conflict_is_the_pair_of_fragments_causing_it(self):
        tilt = URIRef(HYPOTHESIS_NAMESPACE + "tilt")
        regulate = URIRef(HYPOTHESIS_NAMESPACE + "regulate")
        conflict = ContradictionLocaliser(
            PairConflict(first=tilt, second=regulate)
        ).conflict(written(chained_structure()))
        assert set(conflict) == {tilt, regulate}


# %% HermiT


def configured_path(variable: LinkedOntologyVariable) -> Path:
    """
    Return the ontology file an environment variable names, or skip when it names none.
    """
    location = os.environ.get(variable)
    if location is None or not Path(location).is_file():
        pytest.skip(f"set {variable} to the ontology file")
    return Path(location)


@pytest.fixture(scope="module")
def hermit_directory() -> Path:
    """
    The HermiT installation owlready2 ships, or a skip when it or Java is missing.
    """
    if shutil.which("java") is None:
        pytest.skip("HermiT needs a Java runtime")
    directory = HermitCheck.shipped_directory()
    if directory is None:
        pytest.skip("HermiT ships with owlready2, which is not installed")
    return directory


def liftoff_driven_by_grip() -> DecodedHypothesis:
    """
    Return a hypothesis claiming lift-off occurs when grip force overcomes the weight.
    """
    squeeze = core.ForceComponent(has_force_kind=core.ForceKind.GRIP_FORCE)
    weight = core.ForceComponent(has_force_kind=core.ForceKind.WEIGHT)
    opposition = core.ForceOpposition(
        has_driving_force=squeeze, has_resisting_force=weight
    )
    liftoff = prehension.Liftoff(resolves_opposition=opposition)
    phase = core.ActionPhase(label="load")
    fragments = [squeeze, weight, opposition, liftoff, phase]
    names = ["squeeze", "weight", "opposition", "liftoff", "load"]
    return DecodedHypothesis(
        root=liftoff,
        fragments=fragments,
        identifiers={id(fragment): name for fragment, name in zip(fragments, names)},
    )


@pytest.mark.slow
class TestHermitCheck:
    """
    HermiT judges a written hypothesis together with the ontology.
    """

    def test_microworld_check_finds_the_grip_driven_liftoff_inconsistent(
        self, hermit_directory
    ):
        check = HermitCheck(background_files=MICROWORLD_FILES)
        assert not check.is_consistent(written(liftoff_driven_by_grip()))

    def test_contradiction_is_located_in_the_force_fragments(self, hermit_directory):
        check = HermitCheck(background_files=MICROWORLD_FILES)
        conflict = ContradictionLocaliser(check).conflict(
            written(liftoff_driven_by_grip())
        )
        assert URIRef(HYPOTHESIS_NAMESPACE + "load") not in conflict
        assert URIRef(HYPOTHESIS_NAMESPACE + "squeeze") in conflict

    def test_contradicting_answer_is_repaired_through_the_reasoner(
        self, hermit_directory
    ):
        backend = HypothesisBackend(
            client=ScriptedClient(
                responses=[
                    GenerationCase.GRIP_DRIVEN_LIFTOFF.text,
                    GenerationCase.CHAINED_STRUCTURE.text,
                ]
            ),
            suite=AcceptanceSuite(rules=[]),
            consistency=HermitCheck(
                background_files=[
                    *MICROWORLD_FILES,
                    configured_path(LinkedOntologyVariable.CRAM_AICOR_ONTOLOGY),
                    configured_path(LinkedOntologyVariable.DUL_ONTOLOGY),
                ]
            ),
        )
        query = the(core.ActionPhaseStructure)(
            decomposes_task=ConceptReference(iri=CRAM_AICOR_NAMESPACE + "Pouring")
        )
        list(query.evaluate(backend=backend))
        repair_prompt = backend.log[1].user
        assert str(OpenWorldCheck.CONSISTENT_WITH_ONTOLOGY) in repair_prompt
        assert "'squeeze'" in repair_prompt
        assert backend.last_record.consistency == ConsistencyVerdict.CONSISTENT

    def test_consistent_hypothesis_passes_with_cram_aicor_and_dul(
        self, hermit_directory
    ):
        check = HermitCheck(
            background_files=[
                *MICROWORLD_FILES,
                configured_path(LinkedOntologyVariable.CRAM_AICOR_ONTOLOGY),
                configured_path(LinkedOntologyVariable.DUL_ONTOLOGY),
            ]
        )
        assert check.is_consistent(written(chained_structure()))


# %% records and the store


def record_of(
    hypothesis_instruction: str = "pour milk from the carton into the bowl",
) -> HypothesisRecord:
    """
    Return a record of a generated structure with one exchange and one warning.
    """
    return HypothesisRecord(
        instruction=hypothesis_instruction,
        root_type=qualified_name(core.ActionPhaseStructure),
        fixed_values={"decomposes_task": "aicor:Pouring"},
        client_identity="scripted",
        exchanges=[
            Exchange(
                system="system",
                user="user",
                raw_response="{}",
                client_identity="scripted",
                attempt=0,
            )
        ],
        warnings=["[phase_endings_distinct] ActionPhaseStructure 'pour': ..."],
        consistency=ConsistencyVerdict.NOT_CHECKED,
        module_versions=current_module_versions(),
    )


class TestHypothesisStore:
    """
    Admitted hypotheses are stored as their record and their Turtle, and found again.
    """

    def test_record_and_hypothesis_read_back(self, tmp_path):
        store = HypothesisStore(directory=tmp_path)
        record, graph = record_of(), written(chained_structure())
        store.put(record, graph)
        stored = store.get(record.record_id)
        assert stored.record == record
        assert isomorphic(stored.graph, graph)

    def test_hypothesis_is_found_for_the_same_request(self, tmp_path):
        store = HypothesisStore(directory=tmp_path)
        record = record_of()
        store.put(record, written(chained_structure()))
        found = store.find(
            instruction=record.instruction,
            root_type=record.root_type,
            fixed_values=record.fixed_values,
        )
        assert found.record == record

    def test_hypothesis_of_older_ontology_versions_is_not_found(self, tmp_path):
        store = HypothesisStore(directory=tmp_path)
        record = record_of()
        record.module_versions = {name: "0.0.0" for name in record.module_versions}
        store.put(record, written(chained_structure()))
        assert (
            store.find(
                instruction=record.instruction,
                root_type=record.root_type,
                fixed_values=record.fixed_values,
            )
            is None
        )
