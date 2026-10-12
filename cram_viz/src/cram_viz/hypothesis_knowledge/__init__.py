"""
Real, EQL-queryable pouring knowledge for the Hypothesis Graph tab.

A stored hypothesis is read through ``llm_reasoner``'s
:class:`~llm_reasoner.hypotheses.view.HypothesisView`, projected through its
:class:`~llm_reasoner.hypotheses.graph.HypothesisGraphProjector`, and presented phase
first by :class:`.phase_hierarchy.PhaseHierarchyOverlay`.

The hypothesis comes either from the checked-in sample store (:func:`sample_view`) or
from a model generation (:mod:`.generation`). Both take the same path from here on, so
the tab, its EQL variables and its highlighting behave identically whichever produced
the hypothesis.

:func:`active` is the single source of truth for what is currently displayed and
queryable; :func:`activate` swaps in another hypothesis.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from llm_reasoner.knowledge.storage.store import HypothesisStore
from llm_reasoner.hypotheses.graph import (
    HypothesisGraph,
    HypothesisGraphPresentation,
    HypothesisGraphProjector,
)
from llm_reasoner.hypotheses.view import HypothesisView
from typing_extensions import Any, Dict, Optional

from cram_viz import paths
from cram_viz.hypothesis_knowledge.phase_hierarchy import PhaseHierarchyOverlay

SAMPLE_RECORD_ID = "pour_milk_sample"
"""
The record identifier of the checked-in sample hypothesis.
"""


def sample_view() -> HypothesisView:
    """
    :return: The checked-in sample hypothesis, a hand-authored structure for pouring
        milk from a carton into a bowl.
    """
    return HypothesisView.of(
        HypothesisStore(directory=paths.SAMPLE_HYPOTHESES_PATH).get(SAMPLE_RECORD_ID)
    )


def build_projection(view: HypothesisView) -> HypothesisGraph:
    """
    :return: The hypothesis projected with every link drawn directly, then regrouped
        phase first.
    """
    projection = HypothesisGraphProjector().project(
        view, HypothesisGraphPresentation.COMPLETE
    )
    return PhaseHierarchyOverlay().apply(projection, view)


# %% active knowledge


@dataclass
class HypothesisKnowledge:
    """
    One stored hypothesis and everything the viewer derives from it.

    Bundling these together keeps the graph payload, the EQL domains and the query-
    result-to-node mapping from ever describing different hypotheses.
    """

    view: HypothesisView
    """
    The stored hypothesis, as instances of the generated vocabulary.
    """

    projection: HypothesisGraph
    """
    The rendered graph of the hypothesis.
    """

    generated: bool = False
    """
    Whether a model produced this hypothesis, as opposed to the checked-in sample.
    """

    node_identifier_by_object_id: Dict[int, str] = field(
        default_factory=dict, init=False
    )
    """
    The rendered node identifier of each fragment, keyed by the fragment's identity.
    """

    def __post_init__(self) -> None:
        self.node_identifier_by_object_id = {
            id(node.source): node.identifier
            for node in self.projection.nodes
            if node.source is not None
        }

    @classmethod
    def from_view(
        cls, view: HypothesisView, generated: bool = False
    ) -> HypothesisKnowledge:
        """
        :return: The knowledge of a stored hypothesis, projected for display.
        """
        return cls(view=view, projection=build_projection(view), generated=generated)

    def node_identifier_for(self, fragment: Any) -> Optional[str]:
        """
        :return: The rendered node identifier of a fragment, or ``None`` if it is not
            part of this hypothesis.
        """
        return self.node_identifier_by_object_id.get(id(fragment))


_active: Optional[HypothesisKnowledge] = None


def active() -> Optional[HypothesisKnowledge]:
    """
    The knowledge currently displayed and queried, or ``None`` before anything has been
    generated.

    Nothing is active at startup: the tab shows what a model produced for an
    instruction, so an empty tab is the honest state until one has been asked for.
    """
    return _active


def activate(
    knowledge: Optional[HypothesisKnowledge],
) -> Optional[HypothesisKnowledge]:
    """
    Replace what the viewer displays and queries; ``None`` clears it.

    The caller is responsible for resetting :mod:`cram_viz.kb` afterwards, so the EQL
    domains are rebuilt against the new hypothesis.
    """
    global _active
    _active = knowledge
    return _active


def activate_sample() -> HypothesisKnowledge:
    """
    Activate the checked-in sample hypothesis.

    Useful for exercising the projection without spending a generation.
    """
    return activate(HypothesisKnowledge.from_view(sample_view()))


def node_identifier_for(fragment: Any) -> Optional[str]:
    """
    The active projection's node identifier for a knowledge fragment.
    """
    knowledge = active()
    return knowledge.node_identifier_for(fragment) if knowledge else None
