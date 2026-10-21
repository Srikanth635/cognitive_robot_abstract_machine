"""
Tests for `TextTable` and `EvaluationReport`.
"""

from __future__ import annotations

from pathlib import Path

from typesafe_sdk import TypeSafeError

from doc_evaluator.dimensions import Dimension
from doc_evaluator.documents import Document
from doc_evaluator.evaluation import (
    DimensionScore,
    DocumentEvaluation,
    DocumentEvaluationError,
)
from doc_evaluator.report import EvaluationReport, TextTable

# %% TextTable


def test_render_pads_columns_to_the_widest_cell() -> None:
    table = TextTable(
        headers=("Name", "Score"), rows=(("first", "1.00"), ("second-longer", "10.00"))
    )

    lines = table.render().splitlines()

    assert lines[0] == "Name".ljust(13) + " | " + "Score".ljust(5)
    assert lines[1] == "-" * 13 + "-+-" + "-" * 5
    assert lines[2] == "first".ljust(13) + " | " + "1.00".ljust(5)
    assert lines[3] == "second-longer".ljust(13) + " | " + "10.00".ljust(5)


# %% EvaluationReport


def _evaluation_with_one_dimension(
    name: str, score: float, confidence: float
) -> DocumentEvaluation:
    dimension = Dimension(
        name="clarity",
        instructions="How clear is it?",
        criteria=("low", "medium", "high"),
    )
    return DocumentEvaluation(
        document=Document(path=Path(name), text=""),
        dimension_scores=(
            DimensionScore(dimension=dimension, score=score, confidence=confidence),
        ),
    )


def test_render_includes_one_row_per_document_and_one_column_per_dimension() -> None:
    report = EvaluationReport(
        evaluations=(
            _evaluation_with_one_dimension("a.txt", 1.5, 0.8),
            _evaluation_with_one_dimension("b.txt", 2.0, 0.6),
        ),
        failures=(),
    )

    rendered = report.render()

    assert "a.txt" in rendered
    assert "b.txt" in rendered
    assert "clarity" in rendered
    assert "1.50/2 (0.80)" in rendered
    assert "2.00/2 (0.60)" in rendered


def test_render_lists_failures_after_the_table() -> None:
    failure = DocumentEvaluationError(Path("bad.txt"), TypeSafeError("boom"))
    report = EvaluationReport(
        evaluations=(_evaluation_with_one_dimension("a.txt", 1.0, 1.0),),
        failures=(failure,),
    )

    rendered = report.render()

    table, failures_section = rendered.split("\n\n")

    assert "a.txt" in table
    assert failures_section == "Failed to evaluate:\n  bad.txt: boom"
