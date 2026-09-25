"""
Real, EQL-queryable pouring knowledge for the Hypothesis Graph tab.

A :class:`~llm_reasoner.knowledge.microworlds.flanagan.pouring.PouringPhaseStructure`
is projected live through ``llm_reasoner``'s own
:class:`~llm_reasoner.knowledge.hypothesis.graph.HypothesisGraphProjector`, then two
display-only representations are layered on top: the Minsky action frame
(:mod:`.action_frame`) and the image-schema profile (:mod:`.image_schema`).

The structure comes either from :mod:`.pouring_structure` (the checked-in hand-authored
sample, used at startup) or from a model generation (:mod:`.generation`). Both take the
same path from here on, so the tab, its EQL variables and its highlighting behave
identically whichever produced the structure.

:func:`active` is the single source of truth for what is currently displayed and
queryable; :func:`activate` swaps in a newly generated structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from llm_reasoner.knowledge.hypothesis.graph import (
    HypothesisGraph,
    HypothesisGraphPresentation,
    HypothesisGraphProjector,
)
from llm_reasoner.knowledge.hypothesis.provenance import Provenance
from llm_reasoner.knowledge.hypothesis.record import HypothesisRecord
from llm_reasoner.knowledge.microworlds.flanagan.pouring import PouringPhaseStructure
from llm_reasoner.knowledge.ontological.decorators import ConceptReference
from typing_extensions import Any, Dict, Optional

from cram_viz.hypothesis_knowledge.action_frame import (
    MinskyActionFrameOverlay,
    MinskyActionFrameTemplate,
)
from cram_viz.hypothesis_knowledge.image_schema import (
    ImageSchemaProfile,
    ImageSchemaProfileOverlay,
)
from cram_viz.hypothesis_knowledge.phase_hierarchy import PhaseHierarchyOverlay
from cram_viz.hypothesis_knowledge.pouring_structure import (
    build_pouring_phase_structure,
)

#: the instruction text shown at the center of the Hypothesis Graph tab when no
#: generated instruction has replaced it
DEFAULT_INSTRUCTION = "Pour cereal from the cereal box into the bowl"


def build_hypothesis_record(
    structure: PouringPhaseStructure,
    instruction: str = DEFAULT_INSTRUCTION,
) -> HypothesisRecord:
    """
    Wrap a hand-authored structure in a :class:`HypothesisRecord`, admitting every
    real fragment reachable from it into a fresh knowledge scope.

    A model-generated structure arrives inside a record already and does not need
    this.
    """
    return HypothesisRecord.from_root(
        structure,
        action_type=ConceptReference(
            iri="urn:cram_viz:PouringIntoBowl", label=instruction
        ),
        provenance=Provenance(generator="hand-authored", prompt_version="n/a"),
    )


def build_projection(
    record: HypothesisRecord,
    instruction: str = DEFAULT_INSTRUCTION,
    apply_overlays: bool = True,
) -> HypothesisGraph:
    """
    Project a record's real knowledge fragments, optionally layering the two
    display-only representations on top.

    The phase-structure branch is projected fully flat (``COMPLETE``): the phase
    hierarchy overlay owns the entire grouping decision itself rather than un-doing
    the projector's own default root grouping. The action frame is applied next; the
    image-schema profile last, so it can ground its entity-valued primitives in the
    frame's own entity nodes.
    """
    projection = HypothesisGraphProjector().project(
        record,
        instruction=instruction,
        presentation=HypothesisGraphPresentation.COMPLETE,
    )
    if not apply_overlays:
        return projection
    PhaseHierarchyOverlay().apply(projection)
    MinskyActionFrameOverlay(template=MinskyActionFrameTemplate.cereal_pouring()).apply(
        projection
    )
    ImageSchemaProfileOverlay(profile=ImageSchemaProfile.cereal_pouring()).apply(
        projection
    )
    return projection


# %% active knowledge
@dataclass
class HypothesisKnowledge:
    """
    One pouring structure and everything the viewer derives from it.

    Bundling these together keeps the graph payload, the EQL domains and the
    query-result-to-node mapping from ever describing different structures.
    """

    structure: PouringPhaseStructure
    """The real, EQL-queryable structure this knowledge is built from."""

    projection: HypothesisGraph
    """The rendered graph: the structure's fragments plus the display-only overlays."""

    instruction: str
    """The natural-language instruction shown at the centre of the graph."""

    generated: bool = False
    """Whether a model produced this structure, as opposed to the checked-in sample."""

    node_identifier_by_object_id: Dict[int, str] = field(default_factory=dict)
    """Fragment object identity -> its rendered node identifier, for exact highlighting."""

    def __post_init__(self) -> None:
        self.node_identifier_by_object_id = {
            id(node.source): node.identifier
            for node in self.projection.nodes
            if node.source is not None
        }

    @classmethod
    def from_structure(
        cls,
        structure: PouringPhaseStructure,
        instruction: str = DEFAULT_INSTRUCTION,
        generated: bool = False,
        record: Optional[HypothesisRecord] = None,
    ) -> HypothesisKnowledge:
        """Project a structure and bundle it with everything derived from it."""
        record = record or build_hypothesis_record(structure, instruction)
        return cls(
            structure=structure,
            projection=build_projection(record, instruction=instruction),
            instruction=instruction,
            generated=generated,
        )

    def node_identifier_for(self, fragment: Any) -> Optional[str]:
        """
        The rendered node identifier for a knowledge fragment, or ``None`` if it is
        not part of this projection.
        """
        return self.node_identifier_by_object_id.get(id(fragment))


_active: Optional[HypothesisKnowledge] = None


def active() -> Optional[HypothesisKnowledge]:
    """
    The knowledge currently displayed and queried, or ``None`` before anything has
    been generated.

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
    domains are rebuilt against the new structure.
    """
    global _active
    _active = knowledge
    return _active


def activate_sample() -> HypothesisKnowledge:
    """
    Activate the checked-in hand-authored structure.

    Useful for exercising the projection without spending a generation.
    """
    return activate(HypothesisKnowledge.from_structure(build_pouring_phase_structure()))


def node_identifier_for(fragment: Any) -> Optional[str]:
    """The active projection's node identifier for a knowledge fragment."""
    knowledge = active()
    return knowledge.node_identifier_for(fragment) if knowledge else None
