"""
Checking a hypothesis against the ontology in the open world, and locating what makes it
inconsistent.

The acceptance rules read a hypothesis closed-world and catch what is missing. A
reasoner reads it open-world, together with the microworld, CRAM-AICOR and DUL, and
catches what contradicts them, such as a lift-off said to be driven by grip force. A
reasoner only answers yes or no, so the contradiction is located by asking it about
parts of the hypothesis.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum

from rdflib import Graph
from rdflib.term import Node
from typing_extensions import Sequence


class OpenWorldCheck(StrEnum):
    """
    The checks a reasoner makes of a hypothesis.
    """

    CONSISTENT_WITH_ONTOLOGY = "consistent_with_ontology"
    """
    The hypothesis contradicts nothing the microworld, CRAM-AICOR and DUL state, such as
    which force overcomes which at a lift-off, or what kind of thing a fragment may be
    part of.
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
        :return: The graph's statements, grouped by the individual they are about.
        """
        grouped: dict[Node, list[tuple[Node, Node, Node]]] = {}
        for statement in graph:
            grouped.setdefault(statement[0], []).append(statement)
        return grouped

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
