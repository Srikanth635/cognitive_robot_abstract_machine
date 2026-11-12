"""
Checking a hypothesis against the ontology in the open world, and locating what makes it
inconsistent.

The acceptance rules read a hypothesis closed-world and catch what is missing. A
reasoner reads it open-world, together with the microworld and the ontologies it links
to, and catches what contradicts them. A reasoner only answers yes or no, so the
contradiction is located by asking it about parts of the hypothesis.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from enum import StrEnum

from rdflib import OWL, RDF, Graph
from rdflib.term import Node
from typing_extensions import Iterable, Sequence

from llm_reasoner.knowledge.vocabulary.ontology_modules import OntologyModuleFile


class OpenWorldCheck(StrEnum):
    """
    The checks a reasoner makes of a hypothesis.
    """

    CONSISTENT_WITH_ONTOLOGY = "consistent_with_ontology"
    """
    The hypothesis contradicts nothing the microworld and the ontologies it links to
    state, such as what kind of thing a fragment may be part of.
    """


class ConsistencyCheck(ABC):
    """
    Decides whether a hypothesis is consistent with the ontology it is checked against.
    """

    @abstractmethod
    def is_consistent(self, graph: Graph) -> bool:
        """
        :return: Whether the hypothesis, together with the ontology, is consistent.
        """

    def covering(self, module_files: Iterable[OntologyModuleFile]) -> ConsistencyCheck:
        """
        :param module_files: The ontology modules a hypothesis's vocabulary uses.
        :return: A check that knows those modules too; this check itself when it judges
            by means other than the modules' files.
        """
        return self


@dataclass
class ContradictionLocaliser:
    """
    Finds a smallest set of fragments whose statements together contradict the ontology,
    with the QuickXplain algorithm (Junker, 2004).

    Each fragment's statements are kept or left out together, and the reasoner is asked
    about subsets of them.
    """

    check: ConsistencyCheck
    """
    The reasoner judging each subset.
    """

    def conflict(self, graph: Graph) -> list[Node]:
        """
        :return: The individuals whose statements together contradict the ontology,
            such that leaving any one out removes the contradiction; none when the
            hypothesis is consistent.
        """
        statements = self.statements_by_subject(graph)
        subjects = sorted(statements, key=str)
        if self.check.is_consistent(graph):
            return []
        return self.explain([], False, subjects, statements)

    @staticmethod
    def statements_by_subject(
        graph: Graph,
    ) -> dict[Node, list[tuple[Node, Node, Node]]]:
        """
        :return: The graph's statements, grouped by the fragment they are about: a
            statement about a concept's anonymous instance, or about a class of the
            hypothesis's own, goes with every fragment citing it.
        """
        fragments = set(graph.subjects(RDF.type, OWL.NamedIndividual))
        grouped: dict[Node, list[tuple[Node, Node, Node]]] = {}
        for statement in graph:
            for fragment in ContradictionLocaliser.citing(
                graph, statement[0], fragments
            ):
                grouped.setdefault(fragment, []).append(statement)
        return grouped

    @staticmethod
    def citing(graph: Graph, node: Node, fragments: set[Node]) -> set[Node]:
        """
        :return: The fragments citing a node, directly or through nodes that are no
            fragments; the node itself when it is a fragment or nothing cites it.
        """
        if node in fragments:
            return {node}
        citing, reached, pending = set(), {node}, deque([node])
        while pending:
            for subject in graph.subjects(None, pending.popleft()):
                if subject in fragments:
                    citing.add(subject)
                elif subject not in reached:
                    reached.add(subject)
                    pending.append(subject)
        return citing or {node}

    def consistent(
        self,
        subjects: Sequence[Node],
        statements: dict[Node, list[tuple[Node, Node, Node]]],
    ) -> bool:
        """
        :return: Whether the statements about the given individuals are consistent with
            the ontology.
        """
        graph = Graph()
        for subject in subjects:
            for statement in statements[subject]:
                graph.add(statement)
        return self.check.is_consistent(graph)

    def explain(
        self,
        background: list[Node],
        background_changed: bool,
        candidates: list[Node],
        statements: dict[Node, list[tuple[Node, Node, Node]]],
    ) -> list[Node]:
        """
        :return: A smallest subset of the candidates that, with the background,
            contradicts the ontology.
        """
        if background_changed and not self.consistent(background, statements):
            return []
        if len(candidates) == 1:
            return list(candidates)
        middle = len(candidates) // 2
        first, second = candidates[:middle], candidates[middle:]
        from_second = self.explain(background + first, True, second, statements)
        from_first = self.explain(
            background + from_second, bool(from_second), first, statements
        )
        return from_first + from_second
