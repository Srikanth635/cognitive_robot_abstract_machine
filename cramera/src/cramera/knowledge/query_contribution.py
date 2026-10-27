"""
Knowledge a viewer extension adds to what the EQL panel can query in a recorded scene,
beside the episode's own.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from typing_extensions import TYPE_CHECKING, Any, Dict, List

from cramera.knowledge.query_domain import QueryDomain
from cramera.knowledge.query_runner import AnswerRendering

if TYPE_CHECKING:
    from cramera.knowledge.presets import Preset


class QueryContribution(ABC):
    """
    What a viewer extension lets the EQL panel ask about: variables over its entities,
    further names, the questions it offers, and how answers about its values read.
    """

    @abstractmethod
    def domains(self) -> List[QueryDomain]:
        """
        :return: A ready-made query variable per kind of entity contributed.
        """

    @abstractmethod
    def names(self) -> Dict[str, Any]:
        """
        :return: Further names a query may use, such as the vocabulary its answers name.
        """

    @abstractmethod
    def presets(self) -> List[Preset]:
        """
        :return: The questions offered about what is contributed.
        """

    @abstractmethod
    def answers(self) -> AnswerRendering:
        """
        :return: How answers about the contributed values read.
        """
