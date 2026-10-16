"""
The knowledge component as the resolver reaches it: the version of this contract, the
names of the knowledge tools, and what the resolver may ask of the knowledge.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum

from krrood.entity_query_language.query.match import Match
from typing_extensions import Mapping, Optional

from llm_reasoner.clients.tools import Tool
from llm_reasoner.knowledge.ontologies.frame_fields import FramePartDescription

KNOWLEDGE_ACCESS_VERSION = "1.3"
"""
The version of this contract: of the operations below and of the definitions and answer
shapes of the knowledge tools.
"""


class KnowledgeToolName(StrEnum):
    """
    The knowledge tools, each named for the question it answers about one hypothesis.
    """

    KNOWLEDGE_OVERVIEW = "knowledge_overview"
    """
    What the action is, the situation the knowledge is for, and what the action goes
    through.
    """

    FIND_FRAGMENTS = "find_fragments"
    """
    Which fragments of the knowledge belong to a vocabulary class.
    """

    DESCRIBE_FRAGMENT = "describe_fragment"
    """
    What a fragment is, states and is linked from.
    """


class KnowledgeAccess(ABC):
    """
    What the resolver may ask of the knowledge component: the hypothesis for a request,
    the tools a language model reads a hypothesis with, and the knowledge's domains.
    """

    @property
    def version(self) -> str:
        """
        The version of the contract this knowledge fulfils.
        """
        return KNOWLEDGE_ACCESS_VERSION

    @abstractmethod
    def serve(self, query: Match, instruction: str = "") -> str:
        """
        :param query: An underspecified match on a vocabulary class, its fixed values
            the request's situation.
        :param instruction: The action the hypothesis is for, recorded as provenance.
        :return: The record identifier of the stored hypothesis answering the query,
            generated and stored first when there is none.
        """

    @abstractmethod
    def tools(self, *record_ids: str) -> list[Tool]:
        """
        :return: The tools a language model reads the hypotheses with the given record
            identifiers with, together with the hypotheses they build on, as one body of
            knowledge, named by :class:`KnowledgeToolName`.
        """

    @abstractmethod
    def citation(self, record_id: str, name: str) -> Optional[str]:
        """
        :param record_id: The record identifier of a stored hypothesis.
        :param name: The name of a fragment, as the knowledge tools name it.
        :return: The IRI citing the named fragment of the hypothesis, ``None`` when the
            hypothesis itself has no fragment of that name.
        """

    def frame_parts_filled(
        self,
        record_id: str,
        frame_fields: Mapping[str, FramePartDescription],
    ) -> dict[str, list[str]]:
        """
        :param record_id: The record identifier of a stored hypothesis about an
            action's frame in the action core.
        :param frame_fields: The part of the frame each field of the action's designator
            fills, described in the core's vocabulary, by field name.
        :return: The names the knowledge tools give the parts of that frame each field
            fills, by field name; a field filling none is absent, and knowledge without
            frames names none.
        """
        return {}

    @abstractmethod
    def instances_of(self, kind: type) -> list:
        """
        :param kind: A vocabulary class.
        :return: The stored fragments that are instances of the class, a domain for an
            EQL variable over it.
        """

    @abstractmethod
    def is_vocabulary_class(self, kind: type) -> bool:
        """
        :return: Whether a class belongs to the knowledge's vocabulary, so its
            instances are knowledge rather than parts of the world.
        """
