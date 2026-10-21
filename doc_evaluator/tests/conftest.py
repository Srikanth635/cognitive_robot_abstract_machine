"""
Shared fixtures and test doubles for the document evaluator's tests.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from typesafe_sdk import TypeSafeError

from doc_evaluator.dimensions import Dimension, EvaluationRubric

# %% TypeSafe SDK stand-ins


@dataclass(frozen=True)
class StubScoreAnswer:
    """
    Stands in for a TypeSafe `ScoreAnswer`, exposing only the fields the evaluator
    reads.
    """

    score: float
    confidence: float


@dataclass(frozen=True)
class StubSystemOneResponse:
    """
    Stands in for a TypeSafe `SystemOneResponse`, exposing only the `scores` mapping the
    evaluator reads.
    """

    scores: Mapping[str, StubScoreAnswer]


@dataclass
class StubTypeSafeClient:
    """
    Stands in for `TypeSafeClient`: scripted to return a response or raise an error from
    `system_one`, and usable as a context manager.
    """

    response: StubSystemOneResponse | None = None
    error: TypeSafeError | None = None
    received_state: Any = None
    received_questions: Mapping[str, Any] | None = field(default=None)

    def system_one(
        self, state: Any, questions: Mapping[str, Any], **_options: Any
    ) -> StubSystemOneResponse:
        self.received_state = state
        self.received_questions = questions
        if self.error is not None:
            raise self.error
        return self.response

    def __enter__(self) -> StubTypeSafeClient:
        return self

    def __exit__(self, *exception_info: object) -> None:
        return None


# %% fixtures


@pytest.fixture
def two_dimension_rubric() -> EvaluationRubric:
    """
    A rubric with two simple three-level dimensions, for tests that don't need real
    rubric text.
    """
    return EvaluationRubric(
        dimensions=(
            Dimension(
                name="clarity",
                instructions="How clear is the document?",
                criteria=("low", "medium", "high"),
            ),
            Dimension(
                name="rigor",
                instructions="How rigorous is the document?",
                criteria=("low", "medium", "high"),
            ),
        )
    )


@pytest.fixture
def document_directory(tmp_path: Path) -> Path:
    """
    A temporary directory containing one .txt and one .md document.
    """
    (tmp_path / "first.txt").write_text("First document body.", encoding="utf-8")
    (tmp_path / "second.md").write_text("# Second document\n\nBody.", encoding="utf-8")
    return tmp_path
