"""
Answering requests for knowledge with stored hypotheses, generating and admitting one
when the store holds none for the request.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from krrood.entity_query_language.query.match import Match
from typing_extensions import Optional, Sequence

from llm_reasoner.clients.tools import Tool
from llm_reasoner.knowledge.storage.citation import Citation
from llm_reasoner.knowledge.storage.store import HypothesisStore
from llm_reasoner.exceptions import LLMReasonerError
from llm_reasoner.knowledge.generation.backend import HypothesisBackend
from llm_reasoner.knowledge.hypotheses.knowledge_access import KnowledgeAccess
from llm_reasoner.knowledge.hypotheses.knowledge_graph import KnowledgeGraph
from llm_reasoner.knowledge.hypotheses.questions import KnowledgeQuestions
from llm_reasoner.knowledge.hypotheses.tools import KNOWLEDGE_TOOL_KINDS
from llm_reasoner.knowledge.hypotheses.view import HypothesisView
from llm_reasoner.knowledge.ontologies.terms import DescriptionEntity


@dataclass
class StoreRequired(LLMReasonerError):
    """
    The backend stores no hypotheses, so nothing it generates can be read back or cited.
    """

    def __post_init__(self):
        super().__init__("the knowledge service needs a backend with a store")


@dataclass
class KnowledgeService(KnowledgeAccess):
    """
    Views of stored hypotheses, by request or by record, and the knowledge-access
    contract over them.
    """

    backend: HypothesisBackend
    """
    The backend generating and storing a hypothesis when none is stored for a request.
    """

    graph: KnowledgeGraph = field(init=False)
    """
    The knowledge component's own graph over the backend's store, which hypotheses are
    read through.
    """

    def __post_init__(self):
        if self.backend.store is None:
            raise StoreRequired()
        self.graph = KnowledgeGraph(store=self.backend.store)

    @property
    def store(self) -> HypothesisStore:
        """
        The store the backend admits hypotheses into.
        """
        return self.backend.store

    def serve(self, query: Match, instruction: str = "") -> str:
        """
        Serve the hypothesis answering a request, generating, admitting and storing it
        first when the store holds none for it.

        :param query: The underspecified match on a vocabulary class, its fixed values
            the request's situation.
        :param instruction: The action the hypothesis is generated for, recorded as
            provenance when one is generated; it does not decide which stored hypothesis
            answers the query.
        :return: The record identifier of the hypothesis answering the query.
        :raises HypothesisRejected: If no generated answer is admitted.
        """
        backend = replace(self.backend, instruction=instruction)
        list(query.evaluate(backend=backend))
        return backend.last_record.record_id

    def tools(self, *record_ids: str) -> list[Tool]:
        questions = KnowledgeQuestions(hypotheses=self.views_read_with(record_ids))
        return [kind(questions=questions) for kind in KNOWLEDGE_TOOL_KINDS]

    def citation(self, record_id: str, name: str) -> Optional[str]:
        cited = KnowledgeQuestions(hypotheses=[self.view(record_id)]).citation_named(
            name
        )
        return cited.iri if cited is not None and cited.record_id == record_id else None

    def views_read_with(self, record_ids: Sequence[str]) -> list[HypothesisView]:
        """
        :return: The views of the hypotheses with the given records, followed by those
            of the hypotheses they build on, each once.
        """
        pending = list(record_ids)
        read: dict[str, HypothesisView] = {}
        while pending:
            record_id = pending.pop(0)
            if record_id in read:
                continue
            read[record_id] = self.view(record_id)
            pending.extend(read[record_id].record.builds_on)
        return list(read.values())

    def instances_of(self, kind: type) -> list:
        return self.graph.instances_of(kind)

    def is_vocabulary_class(self, kind: type) -> bool:
        return isinstance(kind, type) and issubclass(kind, DescriptionEntity)

    def view_for(self, query: Match, instruction: str = "") -> HypothesisView:
        """
        :param query: The underspecified match on a vocabulary class.
        :param instruction: The action the hypothesis is generated for, as provenance.
        :return: The view of the stored hypothesis answering the query, generated and
            stored first when there is none.
        :raises HypothesisRejected: If no generated answer is admitted.
        """
        return self.view(self.serve(query, instruction))

    def view(self, record_id: str) -> HypothesisView:
        """
        :return: The view of the stored hypothesis with the given record identifier,
            read once into the knowledge graph.
        """
        return self.graph.hypothesis(record_id)

    def fragment_cited(self, citation: Citation) -> DescriptionEntity:
        """
        :return: The stored fragment a citation names.
        :raises UnknownCitation: If its stored hypothesis has no such fragment.
        """
        return self.view(citation.record_id).fragment_cited(citation)
