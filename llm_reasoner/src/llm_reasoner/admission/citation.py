"""
Citing a fragment of a stored hypothesis by the IRI it is stored under.
"""

from __future__ import annotations

from dataclasses import dataclass

from llm_reasoner.admission.rdf import HYPOTHESIS_NAMESPACE_ROOT
from llm_reasoner.exceptions import LLMReasonerError

NAMESPACE_SEPARATOR = "#"
"""
The character separating a stored hypothesis's namespace from a fragment's name.
"""


@dataclass
class NotAHypothesisCitation(LLMReasonerError):
    """
    An IRI does not name a fragment of a stored hypothesis.
    """

    iri: str
    """
    The IRI.
    """

    def __post_init__(self):
        super().__init__(f"{self.iri} names no fragment of a stored hypothesis")


@dataclass(frozen=True)
class Citation:
    """
    A fragment of a stored hypothesis, named by its record and its name there.
    """

    record_id: str
    """
    The identifier of the record of the hypothesis.
    """

    fragment_name: str
    """
    The name of the fragment within the hypothesis.
    """

    @staticmethod
    def namespace_of(record_id: str) -> str:
        """
        :return: The namespace the fragments of a stored hypothesis are named in.
        """
        return f"{HYPOTHESIS_NAMESPACE_ROOT}{record_id}{NAMESPACE_SEPARATOR}"

    @property
    def iri(self) -> str:
        """
        The IRI the fragment is stored under.
        """
        return self.namespace_of(self.record_id) + self.fragment_name

    @classmethod
    def from_iri(cls, iri: str) -> Citation:
        """
        :return: The citation of the fragment stored under an IRI.
        :raises NotAHypothesisCitation: If the IRI is not in a stored hypothesis's
            namespace.
        """
        record_id, separator, fragment_name = iri.removeprefix(
            HYPOTHESIS_NAMESPACE_ROOT
        ).partition(NAMESPACE_SEPARATOR)
        if not iri.startswith(HYPOTHESIS_NAMESPACE_ROOT) or not separator:
            raise NotAHypothesisCitation(iri=iri)
        return cls(record_id=record_id, fragment_name=fragment_name)
