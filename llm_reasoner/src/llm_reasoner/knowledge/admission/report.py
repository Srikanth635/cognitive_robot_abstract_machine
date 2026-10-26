"""
The knowledge's modules merged with the linked ontologies for a maintainer's reasoning,
and ``python -m llm_reasoner.knowledge.admission.report [--not-equated] [--leave-out
<file>...]``, which prints the classes of ours HermiT finds unsatisfiable in that merge.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

from rdflib import OWL, Graph
from typing_extensions import Optional, Sequence

from llm_reasoner.knowledge.admission.hermit import HermitCheck
from llm_reasoner.knowledge.ontologies import (
    KNOWLEDGE_ONTOLOGY_MODULES,
    ActionPerspectiveFile,
    PhysicalMeasurementFile,
    PredictiveManipulationOntologyFile,
)
from llm_reasoner.knowledge.ontologies.linked import AlignmentFile, OntologyFiles
from llm_reasoner.knowledge.ontologies.object_kinds import OBJECT_KINDS_FILE

# %% the merge


@dataclass(frozen=True)
class OntologyMerge:
    """
    Ontology files read as one ontology without their imports, with some statements left
    out and others added, for reasoning over them together.
    """

    files: tuple[Path, ...]
    """
    The files merged.
    """

    left_out: tuple[Path, ...] = ()
    """
    Files whose statements are left out of the merge, such as axioms of a linked
    ontology to be checked as if they were repaired.
    """

    added: tuple[Path, ...] = ()
    """
    Files in Turtle whose statements are added, such as hypotheses.
    """

    @classmethod
    def knowledge(
        cls,
        equated: bool = True,
        left_out: tuple[Path, ...] = (),
        added: tuple[Path, ...] = (),
    ) -> OntologyMerge:
        """
        :param equated: Whether CRAM-AICOR's copies of DUL's classes are equated with DUL
            by the alignment module.
        :return: The pinned linked ontologies with every knowledge module and the kinds
            of object.
        """
        return cls(
            files=(
                *OntologyFiles.pinned().paths,
                *(module_file.path for module_file in KNOWLEDGE_ONTOLOGY_MODULES),
                OBJECT_KINDS_FILE,
                *((AlignmentFile.DUL_ALIGNMENT,) if equated else ()),
            ),
            left_out=left_out,
            added=added,
        )

    @staticmethod
    def is_ours(iri: str) -> bool:
        """
        :return: Whether a term is in the namespaces of the knowledge's own modules.
        """
        return iri.startswith(
            os.path.commonprefix(
                [
                    family.namespace_root
                    for family in (
                        next(iter(ActionPerspectiveFile)),
                        next(iter(PredictiveManipulationOntologyFile)),
                        next(iter(PhysicalMeasurementFile)),
                    )
                ]
            )
        )

    def graph(self) -> Graph:
        """
        :return: The merged statements, without imports, those of the left-out files
            removed and those of the added files added.
        """
        graph = OntologyFiles(paths=self.files).graph()
        graph.remove((None, OWL.imports, None))
        for path in self.left_out:
            for statement in Graph().parse(path, format="turtle"):
                graph.remove(statement)
        for path in self.added:
            graph.parse(path, format="turtle")
        return graph


# %% the report


def main(arguments: Optional[Sequence[str]] = None) -> None:
    """
    Print the classes of ours HermiT finds unsatisfiable in the merge the command line
    asks for.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--not-equated", action="store_true")
    parser.add_argument("--leave-out", type=Path, nargs="*", default=[])
    parser.add_argument("--java-memory-megabytes", type=int, default=6000)
    parsed = parser.parse_args(arguments)
    merged = OntologyMerge.knowledge(
        equated=not parsed.not_equated, left_out=tuple(parsed.leave_out)
    ).graph()
    unsatisfiable = HermitCheck(
        background_files=[], java_memory_megabytes=parsed.java_memory_megabytes
    ).unsatisfiable_classes(merged)
    ours = [iri for iri in unsatisfiable if OntologyMerge.is_ours(iri)]
    print(f"{len(unsatisfiable)} unsatisfiable classes, {len(ours)} of ours:")
    for iri in ours:
        print(f"  {iri}")


if __name__ == "__main__":
    main()
