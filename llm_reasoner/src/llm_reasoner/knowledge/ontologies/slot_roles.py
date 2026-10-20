"""
Which manipulation role of the predictive manipulation microworld fills which slot of the
action core's frame, as the file linking the two families states it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache

from rdflib import RDFS, Graph, URIRef
from typing_extensions import Optional

from llm_reasoner.knowledge.ontologies import ONTOLOGIES_DIRECTORY
from llm_reasoner.knowledge.ontologies.action_perspectives import action_core
from llm_reasoner.knowledge.ontologies.predictive_manipulation import core
from llm_reasoner.knowledge.ontologies.vocabulary import Vocabulary

SLOT_ROLES_FILE = ONTOLOGIES_DIRECTORY / "slot-roles.ttl"
"""
The file putting each manipulation role under the frame slot it fills.
"""

SLOT_ROLES_NAMESPACE = "http://www.ease-crc.org/ont/llm_reasoner/slot-roles#"
"""
The namespace of the file, which declares no term of its own.
"""


@dataclass
class SlotRoles:
    """
    The manipulation roles the file puts under the slots of the action core's frame.
    """

    graph: Graph
    """
    The file linking the families.
    """

    vocabulary: Vocabulary = field(default_factory=Vocabulary)
    """
    Finds the generated class of each role the file names.
    """

    @classmethod
    @cache
    def shipped(cls) -> SlotRoles:
        """
        :return: The links the package ships, read once.
        """
        return cls(graph=Graph().parse(SLOT_ROLES_FILE, format="turtle"))

    def role_filling(
        self, slot: type[action_core.FrameSlot]
    ) -> Optional[type[core.ManipulationRole]]:
        """
        :param slot: A slot class of the action core.
        :return: The manipulation role the file puts directly under the slot, ``None``
            when it puts none there.
        """
        return next(
            (
                self.vocabulary.definition_named(str(role))
                for role in self.graph.subjects(RDFS.subClassOf, URIRef(slot.term.iri))
                if isinstance(role, URIRef)
            ),
            None,
        )
