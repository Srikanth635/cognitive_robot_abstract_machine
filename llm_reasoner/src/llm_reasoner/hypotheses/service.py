"""
Answering requests for knowledge with stored hypotheses, generating and admitting one
when the store holds none for the request.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from krrood.entity_query_language.query.match import Match

from llm_reasoner.admission.citation import Citation
from llm_reasoner.admission.store import HypothesisStore
from llm_reasoner.exceptions import LLMReasonerError
from llm_reasoner.generation.backend import HypothesisBackend
from llm_reasoner.hypotheses.view import HypothesisView
from llm_reasoner.knowledge.ontologies.terms import DescriptionEntity


@dataclass
class StoreRequired(LLMReasonerError):
    """
    The backend stores no hypotheses, so nothing it generates can be read back or cited.
    """

    def __post_init__(self):
        super().__init__("the knowledge service needs a backend with a store")


@dataclass
class KnowledgeService:
    """
    Views of stored hypotheses, by request or by record.
    """

    backend: HypothesisBackend
    """
    The backend generating and storing a hypothesis when none is stored for a request.
    """

    def __post_init__(self):
        if self.backend.store is None:
            raise StoreRequired()

    @property
    def store(self) -> HypothesisStore:
        """
        The store the backend admits hypotheses into.
        """
        return self.backend.store

    def view_for(self, instruction: str, query: Match) -> HypothesisView:
        """
        :return: The view of the stored hypothesis answering the query for the
            instruction, generated and stored first when there is none.
        :raises HypothesisRejected: If no generated answer is admitted.
        """
        backend = replace(self.backend, instruction=instruction)
        list(query.evaluate(backend=backend))
        return self.view(backend.last_record.record_id)

    def view(self, record_id: str) -> HypothesisView:
        """
        :return: The view of the stored hypothesis with the given record identifier.
        """
        return HypothesisView.of(self.store.get(record_id))

    def fragment_cited(self, citation: Citation) -> DescriptionEntity:
        """
        :return: The stored fragment a citation names.
        :raises UnknownCitation: If its stored hypothesis has no such fragment.
        """
        return self.view(citation.record_id).fragment_cited(citation)
