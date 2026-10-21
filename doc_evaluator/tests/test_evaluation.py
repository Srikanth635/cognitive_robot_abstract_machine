"""
Tests for `DocumentEvaluator`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typesafe_sdk import TypeSafeAPIError

from doc_evaluator.dimensions import EvaluationRubric
from doc_evaluator.documents import Document
from doc_evaluator.evaluation import DocumentEvaluationError, DocumentEvaluator

from .conftest import StubScoreAnswer, StubSystemOneResponse, StubTypeSafeClient

# %% evaluate


def test_evaluate_returns_a_score_per_dimension(
    two_dimension_rubric: EvaluationRubric, tmp_path: Path
) -> None:
    document = Document(path=tmp_path / "report.txt", text="body")
    response = StubSystemOneResponse(
        scores={
            "clarity": StubScoreAnswer(score=2.0, confidence=0.9),
            "rigor": StubScoreAnswer(score=1.0, confidence=0.5),
        }
    )
    evaluator = DocumentEvaluator(
        client=StubTypeSafeClient(response=response), rubric=two_dimension_rubric
    )

    evaluation = evaluator.evaluate(document)

    assert evaluation.document is document
    assert evaluation.dimension_scores[0].score == 2.0
    assert evaluation.dimension_scores[0].confidence == 0.9
    assert evaluation.dimension_scores[1].score == 1.0
    assert evaluation.dimension_scores[1].confidence == 0.5


def test_evaluate_sends_the_document_text_and_asks_one_question_per_dimension(
    two_dimension_rubric: EvaluationRubric, tmp_path: Path
) -> None:
    document = Document(path=tmp_path / "report.txt", text="body text")
    response = StubSystemOneResponse(
        scores={
            "clarity": StubScoreAnswer(score=0.0, confidence=1.0),
            "rigor": StubScoreAnswer(score=0.0, confidence=1.0),
        }
    )
    client = StubTypeSafeClient(response=response)
    evaluator = DocumentEvaluator(client=client, rubric=two_dimension_rubric)

    evaluator.evaluate(document)

    assert client.received_state["document"]["text"] == "body text"
    assert set(client.received_questions) == {"clarity", "rigor"}


def test_evaluate_wraps_api_errors_naming_the_failed_document(
    two_dimension_rubric: EvaluationRubric, tmp_path: Path
) -> None:
    document = Document(path=tmp_path / "report.txt", text="body")
    api_error = TypeSafeAPIError(
        status=500, body=None, headers={}, endpoint="POST /system_one"
    )
    evaluator = DocumentEvaluator(
        client=StubTypeSafeClient(error=api_error), rubric=two_dimension_rubric
    )

    with pytest.raises(DocumentEvaluationError) as excinfo:
        evaluator.evaluate(document)

    assert excinfo.value.document_path == document.path
    assert excinfo.value.cause is api_error
