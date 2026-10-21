"""
Command-line entry point: evaluate a directory of documents and print a table.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from dotenv import load_dotenv
from typesafe_sdk import TypeSafeClient, TypeSafeError

from doc_evaluator.dimensions import EvaluationRubric
from doc_evaluator.documents import NoDocumentsFoundError, discover_documents
from doc_evaluator.evaluation import (
    DocumentEvaluation,
    DocumentEvaluationError,
    DocumentEvaluator,
)
from doc_evaluator.report import EvaluationReport

# %% argument parsing


def parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    """
    Parse command-line arguments for the document evaluator CLI.

    :param argv: command-line arguments, excluding the program name
    :return: the parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Evaluate a directory of documents on multiple dimensions using the TypeSafe API."
    )
    parser.add_argument(
        "directory", type=Path, help="directory of documents to evaluate (.txt, .md)"
    )
    return parser.parse_args(argv)


# %% main


def main(argv: Sequence[str] | None = None) -> int:
    """
    Evaluate every document in a directory and print the results as a table.

    :param argv: command-line arguments, excluding the program name; defaults to
        ``sys.argv[1:]``
    :return: process exit code: 0 if every document evaluated successfully, 1 otherwise
    """
    arguments = parse_arguments(argv)
    load_dotenv()

    try:
        documents = discover_documents(arguments.directory)
    except NoDocumentsFoundError as error:
        print(error, file=sys.stderr)
        return 1

    rubric = EvaluationRubric.research_report()
    evaluations: list[DocumentEvaluation] = []
    failures: list[DocumentEvaluationError] = []

    try:
        with TypeSafeClient() as client:
            evaluator = DocumentEvaluator(client=client, rubric=rubric)
            for document in documents:
                try:
                    evaluations.append(evaluator.evaluate(document))
                except DocumentEvaluationError as error:
                    failures.append(error)
    except TypeSafeError as error:
        print(f"Could not reach the TypeSafe API: {error}", file=sys.stderr)
        return 1

    print(
        EvaluationReport(
            evaluations=tuple(evaluations), failures=tuple(failures)
        ).render()
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
