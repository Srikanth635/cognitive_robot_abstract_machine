"""
``python -m llm_reasoner.world.class_ontology <selection.json> <ontology.ttl>``: write
the ontology of the classes a selection file names.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from typing_extensions import List, Optional

from llm_reasoner.world.class_ontology.reading import ClassOntologyReader
from llm_reasoner.world.class_ontology.selection import ClassSelection
from llm_reasoner.world.class_ontology.writing import ClassOntologyWriter

logger = logging.getLogger(__spec__.name)
"""
This module's logger, named for the module even when ``python -m`` runs it.
"""


def main(argv: Optional[List[str]] = None) -> None:
    """
    Write the ontology a selection file asks for.

    :param argv: The command line after the program; the process's when none is given.
    """
    parser = argparse.ArgumentParser(
        prog="python -m llm_reasoner.world.class_ontology",
        description="Write the ontology of the classes a selection file names.",
    )
    parser.add_argument("selection", type=Path, help="the selection file (JSON)")
    parser.add_argument(
        "output", type=Path, help="where to write the ontology (Turtle)"
    )
    arguments = parser.parse_args(argv)
    selection = ClassSelection.from_file(arguments.selection)
    ontology = ClassOntologyReader(selection=selection).read()
    arguments.output.write_text(
        ClassOntologyWriter(ontology=ontology, namespace=selection.namespace).turtle()
    )
    logger.info(
        "%d classes and %d enumerations written to %s",
        len(ontology.classes),
        len(ontology.enumerations),
        arguments.output,
    )
    for skipped in ontology.skipped_modules:
        logger.warning(
            "skipped %s (%s: %s)", skipped.module, skipped.error_type, skipped.reason
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
