from __future__ import annotations

from llm_reasoner.knowledge.hypothesis.graph import (
    HypothesisGraphNode,
    HypothesisGraphNodeKind,
    HypothesisGraphProjector,
    HypothesisGraphVisualizer,
)
from llm_reasoner.knowledge.hypothesis.presentation import OntologyGraphPresentation
from llm_reasoner.knowledge.hypothesis.provenance import Provenance
from llm_reasoner.knowledge.hypothesis.record import HypothesisRecord
from llm_reasoner.knowledge.microworlds.flanagan.core import (
    ActionPhase,
    ActionPhaseStructure,
)
from llm_reasoner.knowledge.microworlds.flanagan.prehension import (
    ForceLoadingPhase,
    Liftoff,
    ReachPhase,
)
from llm_reasoner.knowledge.ontological.decorators import REGISTRY, ConceptReference
from llm_reasoner.knowledge.query import HypothesisView

# %% Presentation fixtures


def instruction_node() -> HypothesisGraphNode:
    """
    Return a central instruction node with natural-language content.
    """
    return HypothesisGraphNode(
        identifier="instruction:test",
        label="Pour cereal from the cereal box into the bowl",
        type_name="Instruction",
        kind=HypothesisGraphNodeKind.INSTRUCTION,
    )


def pouring_record() -> HypothesisRecord:
    """
    Return a small typed phase structure for presentation and query tests.
    """
    reach = ReachPhase(label="Reach toward the cereal box")
    load = ForceLoadingPhase(label="Increase grip and load force")
    structure = ActionPhaseStructure(
        label="Cereal-pouring phase structure",
        decomposes_task=ConceptReference(
            iri="soma:PouringInto",
            label="PouringInto",
        ),
        defines_phase=[reach, load],
    )
    return HypothesisRecord.from_root(
        structure,
        action_type=structure.decomposes_task,
        provenance=Provenance(
            generator="test",
            prompt_version="generation/v2",
        ),
    )


# %% Ontology names and descriptions


class TestOntologyDisplayMetadata:
    """
    Microworld classes own their canonical concise display concepts.
    """

    def test_force_loading_phase_declares_its_clear_display_name(self) -> None:
        metadata = REGISTRY.meta_for(ForceLoadingPhase)

        assert metadata is not None
        assert metadata.display_name == "ForceLoading"


class TestOntologyGraphLabels:
    """
    Graph labels derive from ontology classes rather than prose.
    """

    def test_instruction_remains_natural_language(self) -> None:
        presentation = OntologyGraphPresentation()

        assert presentation.label(instruction_node()) == (
            "Pour cereal from the cereal box into the bowl"
        )

    def test_phase_uses_declared_display_name_and_retains_description(self) -> None:
        phase = ForceLoadingPhase(label="Increase grip and load force")
        node = HypothesisGraphNode(
            identifier=phase.local_id,
            label=phase.label,
            type_name=type(phase).__name__,
            kind=HypothesisGraphNodeKind.FRAGMENT,
            source=phase,
        )
        presentation = OntologyGraphPresentation()

        assert presentation.label(node) == "ForceLoading"
        assert presentation.information(node) == [
            "description: Increase grip and load force",
            *node.graph_information(),
        ]

    def test_phase_structure_is_the_action_phases_branch(self) -> None:
        structure = ActionPhaseStructure(label="Cereal-pouring phase structure")
        node = HypothesisGraphNode(
            identifier=structure.local_id,
            label=structure.label,
            type_name=type(structure).__name__,
            kind=HypothesisGraphNodeKind.ROOT,
            source=structure,
        )

        assert OntologyGraphPresentation().label(node) == "ActionPhases"

    def test_event_uses_its_ontology_class_and_retains_description(self) -> None:
        event = Liftoff(label="Load exceeds the cereal box weight")
        node = HypothesisGraphNode(
            identifier=event.local_id,
            label=event.label,
            type_name=type(event).__name__,
            kind=HypothesisGraphNodeKind.FRAGMENT,
            source=event,
        )
        presentation = OntologyGraphPresentation()

        assert presentation.label(node) == "Liftoff"
        assert presentation.information(node)[0] == (
            "description: Load exceeds the cereal box weight"
        )


class TestOntologyGraphVisualizerFactory:
    """
    Ontology presentation is opt-in and preserves the existing factory.
    """

    def test_ontology_factory_uses_canonical_labels(self) -> None:
        record = pouring_record()
        projection = HypothesisGraphProjector().project(
            record,
            instruction="Pour cereal from the cereal box into the bowl",
        )

        visualizer = HypothesisGraphVisualizer.for_ontology_graph(projection)
        labels = {
            visualizer.node_label(index) for index in projection.graph.node_indices()
        }

        assert "Pour cereal from the cereal box into the bowl" in labels
        assert "ActionPhases" in labels
        assert "Reach" in labels
        assert "ForceLoading" in labels


# %% Knowledge-local polymorphic domains


class TestPolymorphicKnowledgeDomain:
    """
    Knowledge views expose subclass-inclusive domains for EQL callers.
    """

    def test_action_phase_domain_contains_concrete_phase_subclasses(self) -> None:
        view = HypothesisView(pouring_record())

        phases = view.domain(ActionPhase)

        assert [type(phase) for phase in phases] == [ReachPhase, ForceLoadingPhase]
