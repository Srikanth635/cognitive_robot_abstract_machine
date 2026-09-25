"""
Ontology-oriented labels and details for hypothesis graph nodes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from llm_reasoner.knowledge.ontological.decorators import (
    REGISTRY,
    OntologyRegistry,
)

# %% Ontology graph presentation


@dataclass(frozen=True)
class OntologyGraphPresentation:
    """
    Render concise ontology concepts while retaining generated prose.
    """

    registry: OntologyRegistry = field(default_factory=lambda: REGISTRY)
    """
    Ontology registry providing display metadata for projected classes.
    """

    def label(self, node: Any) -> str:
        """
        Return the concise concept label for a projected graph node.
        """
        if node.type_name == "Instruction":
            return node.label
        if node.source is None:
            return node.label
        for source_type in type(node.source).__mro__:
            metadata = self.registry.meta_for(source_type)
            if metadata is not None and metadata.display_name is not None:
                return metadata.display_name
        if node.type_name.endswith("EventType"):
            return node.type_name.removesuffix("Type")
        return node.type_name

    def information(self, node: Any) -> list[str]:
        """
        Return node details with generated prose preserved as description.
        """
        information = node.graph_information()
        if node.source is None or not node.label:
            return information
        return [f"description: {node.label}", *information]


__all__ = ["OntologyGraphPresentation"]
