"""
What reading the ontology modules into the Python vocabulary refuses.
"""

from __future__ import annotations

from dataclasses import dataclass

from krrood.exceptions import DataclassException

from llm_reasoner.exceptions import LLMReasonerError

# %% value partitions


@dataclass
class PartitionClosureConflict(LLMReasonerError, DataclassException):
    """
    A class is marked an open value partition and defined by enumerating its members, so
    whether other modules may add members is undecided.
    """

    partitions: list[str]
    """
    The IRIs of the classes marked both ways.
    """

    def error_message(self) -> str:
        return (
            "classes are both marked open value partitions and defined by owl:oneOf: "
            + ", ".join(self.partitions)
        )

    def suggest_correction(self) -> str:
        return (
            "keep owl:oneOf for a class whose members are fixed by its definition, and "
            "the open value partition marker for one other modules may add members to"
        )


@dataclass
class MembersNotStatedDistinct(LLMReasonerError, DataclassException):
    """
    A module declares members of an open value partition without one
    ``owl:AllDifferent`` stating them distinct from each other and from the members the
    modules it imports declare.
    """

    partition: str
    """
    The IRI of the open value partition.
    """

    module: str
    """
    The file name of the module declaring the members.
    """

    members: list[str]
    """
    The IRIs of the members that one statement of distinctness has to list.
    """

    def error_message(self) -> str:
        return (
            f"{self.module} declares members of {self.partition} that no "
            "owl:AllDifferent of the module states distinct from each other and from "
            "the members its imports declare"
        )

    def suggest_correction(self) -> str:
        return f"state in {self.module} one owl:AllDifferent listing " + ", ".join(
            self.members
        )
