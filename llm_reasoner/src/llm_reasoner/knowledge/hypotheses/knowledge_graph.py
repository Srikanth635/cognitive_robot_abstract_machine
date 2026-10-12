"""
The knowledge component's own graph: the fragments of its stored hypotheses, linked by
their fields, apart from krrood's symbol graph, which is the world's.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from llm_reasoner.knowledge.hypotheses.view import Fragment, HypothesisView
from llm_reasoner.knowledge.storage.store import HypothesisStore


@dataclass
class KnowledgeGraph:
    """
    The stored hypotheses as objects, each read once, so a fragment is the same object
    wherever it is reached; the source of EQL's domains over knowledge.
    """

    store: HypothesisStore
    """
    Where the hypotheses are stored.
    """

    read_hypotheses: dict[str, HypothesisView] = field(default_factory=dict, init=False)
    """
    The hypotheses read so far, by record identifier.
    """

    def hypothesis(self, record_id: str) -> HypothesisView:
        """
        :return: The view of the stored hypothesis with the given record identifier,
            read from the store the first time it is asked for.
        """
        if record_id not in self.read_hypotheses:
            self.read_hypotheses[record_id] = HypothesisView.of(
                self.store.get(record_id)
            )
        return self.read_hypotheses[record_id]

    def hypotheses(self) -> list[HypothesisView]:
        """
        :return: The view of every stored hypothesis.
        """
        return [self.hypothesis(record.record_id) for record in self.store.records()]

    def instances_of(self, kind: type[Fragment]) -> list[Fragment]:
        """
        :param kind: A vocabulary class.
        :return: The fragments of every stored hypothesis that are instances of the
            class, a domain for an EQL variable over it.
        """
        return [
            fragment
            for hypothesis in self.hypotheses()
            for fragment in hypothesis.instances_of(kind)
        ]
