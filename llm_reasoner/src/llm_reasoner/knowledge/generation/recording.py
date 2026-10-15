"""
Recording an admitted hypothesis — what was asked, of which model, under which ontology
versions, with what warnings — and storing it when there is a store.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import Optional

from llm_reasoner.knowledge.generation.dialogue import GenerationOutcome
from llm_reasoner.knowledge.generation.prompts import described_violation
from llm_reasoner.knowledge.generation.request import HypothesisRequest
from llm_reasoner.knowledge.representation.rdf import GraphWriter
from llm_reasoner.knowledge.storage.citation import Citation
from llm_reasoner.knowledge.storage.record import (
    ConsistencyVerdict,
    HypothesisRecord,
    current_module_versions,
    qualified_name,
)
from llm_reasoner.knowledge.storage.store import HypothesisStore


@dataclass
class HypothesisRecorder:
    """
    Records admitted hypotheses, and stores them when given a store.
    """

    client_identity: str
    """
    Which model generated the hypotheses, as its client reports it.
    """

    consistency_checked: bool
    """
    Whether the hypotheses were checked in the open world before being admitted.
    """

    store: Optional[HypothesisStore] = None
    """
    Where admitted hypotheses are stored; no storing when absent.
    """

    def record(
        self, outcome: GenerationOutcome, request: HypothesisRequest
    ) -> HypothesisRecord:
        """
        :return: The record of an admitted hypothesis, stored with it when there is a
            store.
        """
        hypothesis = outcome.hypothesis
        record = HypothesisRecord(
            instruction=request.instruction,
            root_type=qualified_name(request.root_type),
            fixed_values=request.written_fixed_values,
            client_identity=self.client_identity,
            exchanges=outcome.exchanges,
            warnings=[
                described_violation(warning, hypothesis.identifier_of)
                for warning in outcome.warnings
            ],
            consistency=(
                ConsistencyVerdict.CONSISTENT
                if self.consistency_checked
                else ConsistencyVerdict.NOT_CHECKED
            ),
            module_versions=current_module_versions(request.module_files),
        )
        if self.store is not None:
            namespace = Citation.namespace_of(record.record_id)
            self.store.put(record, GraphWriter(namespace=namespace).write(hypothesis))
        return record
