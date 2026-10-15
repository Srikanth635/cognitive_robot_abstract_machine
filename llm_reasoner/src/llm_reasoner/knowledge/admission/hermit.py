"""
Checking consistency with HermiT, run as a separate Java process.

HermiT ships with owlready2, which is only needed to locate it; the reasoner itself is
run through its command line, which reports an inconsistent ontology by name.
"""

from __future__ import annotations

import importlib.util
import subprocess
import tempfile
from dataclasses import dataclass, replace
from enum import StrEnum
from functools import cached_property
from pathlib import Path

from rdflib import OWL, Graph
from typing_extensions import Iterable, Optional, Sequence

from llm_reasoner.knowledge.admission.consistency import ConsistencyCheck
from llm_reasoner.exceptions import LLMReasonerError
from llm_reasoner.knowledge.ontologies import (
    KNOWLEDGE_ONTOLOGY_MODULES,
    OntologyModuleFile,
    PredictiveManipulationOntologyFile,
)
from llm_reasoner.knowledge.ontologies.linked import OntologyFiles

# %% constants


class HermitCommand(StrEnum):
    """
    The parts of the HermiT command line used.
    """

    MAIN_CLASS = "org.semanticweb.HermiT.cli.CommandLine"
    """
    The class running HermiT from the command line.
    """

    CHECK_CONSISTENCY = "-k"
    """
    The option checking that the ontology is consistent.
    """

    JAR = "HermiT.jar"
    """
    The file name of the HermiT archive.
    """

    INCONSISTENCY_REPORT = "InconsistentOntologyException"
    """
    What HermiT reports when the ontology is inconsistent.
    """


REASONER_PACKAGE = "owlready2"
"""
The Python package HermiT ships with.
"""

HERMIT_SUBDIRECTORY = "hermit"
"""
The directory of HermiT inside that package.
"""

# %% exceptions


@dataclass
class ReasonerFailed(LLMReasonerError):
    """
    HermiT stopped without judging the ontology, for instance on a parsing error.
    """

    output: str
    """
    What HermiT printed.
    """

    def __post_init__(self):
        super().__init__(f"HermiT failed without a verdict: {self.output[-2000:]}")


@dataclass
class HermitNotInstalled(LLMReasonerError):
    """
    HermiT was not found, because owlready2 is not installed.
    """

    def __post_init__(self):
        super().__init__(
            f"HermiT ships with {REASONER_PACKAGE}; install it to check consistency"
        )


# %% check


@dataclass
class HermitCheck(ConsistencyCheck):
    """
    Checks a hypothesis against background ontology files with HermiT.
    """

    background_files: Sequence[Path]
    """
    The ontology files the hypothesis is checked against, in Turtle or RDF/XML.
    """

    java_memory_megabytes: int = 4000
    """
    The heap size given to the Java process.
    """

    @classmethod
    def from_environment(cls) -> Optional[HermitCheck]:
        """
        :return: A check against the microworld and the linked ontologies the
            environment names, or ``None`` when it does not name all of them.
        """
        linked = OntologyFiles.linked_from_environment()
        if linked is None:
            return None
        return cls(
            background_files=[
                *(
                    module_file.path
                    for module_file in PredictiveManipulationOntologyFile
                ),
                *linked.paths,
            ]
        )

    def covering(self, module_files: Iterable[OntologyModuleFile]) -> HermitCheck:
        """
        :return: A check whose background also holds the files of the given modules it
            lacks, in the order the modules are registered.
        """
        wanted = set(module_files)
        missing = [
            module_file.path
            for module_file in KNOWLEDGE_ONTOLOGY_MODULES
            if module_file in wanted and module_file.path not in self.background_files
        ]
        return replace(self, background_files=[*self.background_files, *missing])

    @staticmethod
    def shipped_directory() -> Optional[Path]:
        """
        :return: The directory of the HermiT shipped with owlready2, or ``None`` when
            owlready2 is not installed.
        """
        specification = importlib.util.find_spec(REASONER_PACKAGE)
        if specification is None or specification.origin is None:
            return None
        return Path(specification.origin).parent / HERMIT_SUBDIRECTORY

    @cached_property
    def background(self) -> Graph:
        """
        Every background file in one graph, without its imports, which are resolved
        by listing the imported files themselves.
        """
        graph = OntologyFiles(paths=self.background_files).graph()
        graph.remove((None, OWL.imports, None))
        return graph

    def is_consistent(self, graph: Graph) -> bool:
        directory = self.shipped_directory()
        if directory is None:
            raise HermitNotInstalled()
        with tempfile.TemporaryDirectory() as scratch_directory:
            ontology = Path(scratch_directory) / "checked.nt"
            (self.background + graph).serialize(ontology, format="nt", encoding="utf-8")
            result = subprocess.run(
                [
                    "java",
                    f"-Xmx{self.java_memory_megabytes}M",
                    "-cp",
                    f"{directory}:{directory / HermitCommand.JAR}",
                    HermitCommand.MAIN_CLASS,
                    HermitCommand.CHECK_CONSISTENCY,
                    ontology.as_uri(),
                ],
                capture_output=True,
                text=True,
            )
        if result.returncode == 0:
            return True
        output = result.stdout + result.stderr
        if HermitCommand.INCONSISTENCY_REPORT in output:
            return False
        raise ReasonerFailed(output)
