"""
What the ontology says of kinds of object, for the classes of the code: a class put under
CRAM-AICOR's container has an opening, and so does every class that specialises it.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

from rdflib import OWL, RDF, RDFS, BNode, Graph, Literal, URIRef

from llm_reasoner.knowledge.ontologies import CRAM_AICOR_NAMESPACE
from llm_reasoner.knowledge.ontologies.frame_fields import (
    CODE_ONTOLOGY_DIRECTORY,
    CodeOntology,
)

OBJECT_KINDS_FILE = CODE_ONTOLOGY_DIRECTORY / "object-kinds.ttl"
"""
The file saying which classes of the code are containers and that a container has an
opening.
"""

OBJECT_KINDS_NAMESPACE = "http://www.ease-crc.org/ont/llm_reasoner/object-kinds#"
"""
The namespace the property of a container having an opening is named in.
"""

CONTAINER = URIRef(CRAM_AICOR_NAMESPACE + "Container")
"""
CRAM-AICOR's container.
"""

HAS_OPENING = URIRef(OBJECT_KINDS_NAMESPACE + "hasOpening")
"""
The property saying that a container has an opening.
"""


@dataclass
class ObjectKinds:
    """
    The kinds of object the ontology classifies, read for whether they have an opening.
    """

    graph: Graph
    """
    The file saying which classes are containers and that a container has an opening.
    """

    code: CodeOntology
    """
    The ontology of the code's classes, whose hierarchy a class's kind follows.
    """

    @classmethod
    @cache
    def shipped(cls) -> ObjectKinds:
        """
        :return: The kinds the package ships, read once.
        """
        return cls(
            graph=Graph().parse(OBJECT_KINDS_FILE, format="turtle"),
            code=CodeOntology.shipped(),
        )

    def has_opening(self, class_name: str) -> bool:
        """
        :param class_name: The module and class name of a class of the code.
        :return: Whether the ontology says its instances have an opening: the class or a
            class it specialises is put under the container, and a container has one.
            ``False`` for a class the code ontology lacks.
        """
        return self.container_has_opening() and any(
            (base, RDFS.subClassOf, CONTAINER) in self.graph
            for base in self.code.class_and_bases(
                URIRef(self.code.namespace + class_name)
            )
        )

    def container_has_opening(self) -> bool:
        """
        :return: Whether the file says that a container has an opening.
        """
        return any(
            isinstance(node, BNode)
            and (node, OWL.onProperty, HAS_OPENING) in self.graph
            and (node, OWL.hasValue, Literal(True)) in self.graph
            for node in self.graph.objects(CONTAINER, RDFS.subClassOf)
        )
