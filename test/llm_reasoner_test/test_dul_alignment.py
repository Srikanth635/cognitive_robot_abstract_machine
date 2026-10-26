"""
Tests of aligning the knowledge with DUL: CRAM-AICOR keeps its own copies of DUL's
classes (``aicor:dul_*``), and the alignment module equates them with DUL. With CRAM-
AICOR's own root causes subtracted, as if it had repaired them, no class of ours may
fall into two DUL categories DUL declares disjoint, and the merged ontology must stay
consistent and satisfiable with every shipped hypothesis.

The fast tests read the told axioms with rdflib; the slow ones run HermiT.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import pytest
from rdflib import OWL, RDF, RDFS, BNode, Graph, URIRef
from rdflib.collection import Collection
from typing_extensions import Iterable

from llm_reasoner.knowledge.admission.hermit import HermitCheck
from llm_reasoner.knowledge.admission.report import OntologyMerge
from llm_reasoner.knowledge.ontologies import CRAM_AICOR_NAMESPACE, DUL_NAMESPACE
from llm_reasoner.knowledge.ontologies.linked import AlignmentFile, LinkedOntologyFile

CASES = Path(__file__).parent / "linked_ontology_cases"
"""
The fixtures of the linked ontologies.
"""

ROOT_CAUSES = CASES / "upstream_root_causes.ttl"
"""
The five CRAM-AICOR axioms the tests subtract, as if CRAM-AICOR had repaired them.
"""

CLASH_LINKS = CASES / "clash_links.ttl"
"""
The DUL links our modules dropped, re-added.
"""

CLASHING_CLASSES = {
    "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ActionPhase",
    "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#PhaseBoundaryEventType",
    "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#PhaseConditionType",
    "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#SensoryObservationType",
    "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/gaze#ClosestApproach",
}
"""
The classes the dropped links put into two disjoint DUL categories.
"""

SHIPPED_HYPOTHESES = (
    Path(__file__).parents[2] / "cram_viz" / "src" / "cram_viz" / "resources"
).glob("sample_hypotheses/*/hypothesis.ttl")
"""
The hypotheses shipped with cram_viz.
"""

# %% reading told axioms


def named(graph: Graph, node) -> Iterable[URIRef]:
    """
    :return: The named classes a class expression is or intersects.
    """
    if isinstance(node, URIRef):
        yield node
        return
    members = graph.value(node, OWL.intersectionOf)
    if members is not None:
        for member in Collection(graph, members):
            yield from named(graph, member)


def told_superclasses(graph: Graph, cls: URIRef) -> set[URIRef]:
    """
    :return: The named classes the class is told to specialise or equal, directly or
        through others, itself included.
    """
    found = {cls}
    pending = deque([cls])
    while pending:
        current = pending.popleft()
        parents = [
            *graph.objects(current, RDFS.subClassOf),
            *graph.objects(current, OWL.equivalentClass),
            *graph.subjects(OWL.equivalentClass, current),
        ]
        for parent in parents:
            for superclass in named(graph, parent):
                if superclass not in found:
                    found.add(superclass)
                    pending.append(superclass)
    return found


def disjoint_pairs(graph: Graph) -> set[frozenset[URIRef]]:
    """
    :return: Every pair of named classes the graph declares disjoint.
    """
    pairs = {
        frozenset((first, second))
        for first, second in graph.subject_objects(OWL.disjointWith)
        if isinstance(first, URIRef) and isinstance(second, URIRef)
    }
    for axiom in graph.subjects(RDF.type, OWL.AllDisjointClasses):
        members = [
            member
            for member in Collection(graph, graph.value(axiom, OWL.members))
            if isinstance(member, URIRef)
        ]
        pairs |= {
            frozenset((first, second))
            for index, first in enumerate(members)
            for second in members[index + 1 :]
        }
    return pairs


def clash_of(
    superclasses: set[URIRef], disjoint: set[frozenset[URIRef]]
) -> set[frozenset[URIRef]]:
    """
    :return: The disjoint pairs of which a class specialises both members.
    """
    return {pair for pair in disjoint if pair <= superclasses}


def is_ours(node) -> bool:
    """
    :return: Whether a term is one of the knowledge's own.
    """
    return isinstance(node, URIRef) and OntologyMerge.is_ours(str(node))


@pytest.fixture(scope="module")
def repaired() -> Graph:
    """
    The pinned CRAM-AICOR without its root causes, DUL, the knowledge's modules and the
    alignment, told axioms only.
    """
    return OntologyMerge.knowledge(left_out=(ROOT_CAUSES,)).graph()


# %% the alignment module


class TestAlignmentModule:
    def test_it_equates_exactly_the_copies_cram_aicor_declares_with_dul(self):
        cram_aicor = Graph().parse(LinkedOntologyFile.CRAM_AICOR, format="turtle")
        copies = {
            subject
            for subject in cram_aicor.subjects(RDF.type, OWL.Class)
            if str(subject).startswith(CRAM_AICOR_NAMESPACE + "dul_")
        }
        alignment = Graph().parse(AlignmentFile.DUL_ALIGNMENT, format="turtle")
        assert set(alignment.subject_objects(OWL.equivalentClass)) == {
            (
                copy,
                URIRef(DUL_NAMESPACE + str(copy)[len(CRAM_AICOR_NAMESPACE + "dul_") :]),
            )
            for copy in copies
        }

    def test_no_class_of_ours_falls_into_two_disjoint_dul_categories(self, repaired):
        disjoint = disjoint_pairs(repaired)
        clashing = {
            cls
            for cls in set(repaired.subjects(RDF.type, OWL.Class))
            if is_ours(cls) and clash_of(told_superclasses(repaired, cls), disjoint)
        }
        assert clashing == set()

    def test_no_property_of_ours_specialises_a_dul_property_against_its_domain(
        self, repaired
    ):
        disjoint = disjoint_pairs(repaired)
        clashing = set()
        for prop, specialised in repaired.subject_objects(RDFS.subPropertyOf):
            if not is_ours(prop) or not str(specialised).startswith(DUL_NAMESPACE):
                continue
            for kind in (RDFS.domain, RDFS.range):
                demanded = set(repaired.objects(specialised, kind))
                for stated in repaired.objects(prop, kind):
                    for superclass in demanded:
                        both = told_superclasses(repaired, stated) | told_superclasses(
                            repaired, superclass
                        )
                        if clash_of(both, disjoint):
                            clashing.add((prop, kind, superclass))
        assert clashing == set()


# %% reasoning


@pytest.fixture(scope="module")
def hermit() -> HermitCheck:
    """
    HermiT with nothing in its background, or a skip when it is not installed.
    """
    if HermitCheck.shipped_directory() is None:
        pytest.skip("needs owlready2's HermiT")
    return HermitCheck(background_files=[], java_memory_megabytes=6000)


@pytest.mark.slow
class TestReasoning:
    def test_the_knowledge_is_consistent_and_satisfiable_once_cram_aicor_is_repaired(
        self, hermit
    ):
        merged = OntologyMerge.knowledge(left_out=(ROOT_CAUSES,)).graph()
        assert hermit.is_consistent(merged)
        assert hermit.unsatisfiable_classes(merged) == []

    def test_every_shipped_hypothesis_is_consistent_with_it(self, hermit):
        merged = OntologyMerge.knowledge(
            left_out=(ROOT_CAUSES,), added=tuple(SHIPPED_HYPOTHESES)
        ).graph()
        assert hermit.is_consistent(merged)

    def test_with_cram_aicor_as_it_is_our_unsatisfiable_classes_follow_from_it(
        self, hermit
    ):
        merged = OntologyMerge.knowledge().graph()
        causes = {
            subject
            for subject in Graph().parse(ROOT_CAUSES, format="turtle").subjects()
        }
        unexplained = {
            iri
            for iri in hermit.unsatisfiable_classes(merged)
            if OntologyMerge.is_ours(iri)
            and not reaches_any(merged, URIRef(iri), causes)
        }
        assert unexplained == set()

    def test_the_dropped_links_would_make_our_classes_unsatisfiable(self, hermit):
        merged = OntologyMerge.knowledge(
            left_out=(ROOT_CAUSES,), added=(CLASH_LINKS,)
        ).graph()
        assert set(hermit.unsatisfiable_classes(merged)) >= CLASHING_CLASSES


def reaches_any(graph: Graph, cls: URIRef, targets: set) -> bool:
    """
    :return: Whether the class specialises one of the targets, or requires a value of a
        class that does, through told superclasses and the fillers of their restrictions.
    """
    seen = set()
    pending = deque([cls])
    while pending:
        current = pending.popleft()
        if current in seen:
            continue
        seen.add(current)
        superclasses = told_superclasses(graph, current)
        if superclasses & targets:
            return True
        for superclass in superclasses:
            for parent in graph.objects(superclass, RDFS.subClassOf):
                if isinstance(parent, BNode):
                    for quantifier in (OWL.someValuesFrom, OWL.allValuesFrom):
                        filler = graph.value(parent, quantifier)
                        pending.extend(named(graph, filler) if filler else ())
    return False
