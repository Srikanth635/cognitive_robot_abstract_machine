"""
A family of ontology modules sharing an open value partition: a hub declares it, and two
modules importing the hub, but not each other, add members to it.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from rdflib import Namespace

from llm_reasoner.knowledge.ontologies import OntologyModuleFile

# %% the family


class OpenPartitionModuleFile(OntologyModuleFile, StrEnum):
    """
    The Turtle file of each module, listed with every module after the modules it
    depends on.
    """

    HUB = "hub.ttl"
    """
    Corrections and the open value partition of their directions.
    """

    TARGET_CORRECTIONS = "target-corrections.ttl"
    """
    Corrections relative to a target, adding two directions.
    """

    SIDEWAYS_CORRECTIONS = "sideways-corrections.ttl"
    """
    Corrections to one side, adding one direction and closing it per correction.
    """

    @property
    def directory(self) -> Path:
        """
        The directory holding the family's Turtle files.
        """
        return Path(__file__).parent / "ontology_translation_cases" / "open_partitions"

    @property
    def namespace_root(self) -> str:
        """
        The IRI prefix of the family's modules.
        """
        return "http://www.ease-crc.org/ont/llm_reasoner/test/open-partitions/"

    @property
    def usable_modules(self) -> set[OntologyModuleFile]:
        """
        The module and the hub, as a perspective uses the action core.
        """
        return {OpenPartitionModuleFile.HUB, self}

    @property
    def vocabulary_package(self) -> str:
        """
        The package the tests import the family's generated modules from.
        """
        return "open_partition_vocabulary"

    @property
    def terms(self) -> Namespace:
        """
        The terms the module declares, by local name.
        """
        return Namespace(self.namespace)
