"""
Scoring documents against a rubric with the TypeSafe API.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from typesafe_sdk import TypeSafeClient, TypeSafeError

from doc_evaluator.dimensions import Dimension, EvaluationRubric
from doc_evaluator.documents import Document

# %% results


@dataclass(frozen=True)
class DimensionScore:
    """
    A document's rating on a single :class:`Dimension`.
    """

    dimension: Dimension
    """
    The dimension this score was rated against.
    """

    score: float
    """Probability-weighted position on the dimension's rubric levels, from 0 up to
    ``len(dimension.criteria) - 1``."""

    confidence: float
    """
    Model confidence in the score, from 0 to 1.
    """

    @property
    def maximum_score(self) -> float:
        """
        The highest score achievable on this dimension's rubric.
        """
        return len(self.dimension.criteria) - 1


@dataclass(frozen=True)
class DocumentEvaluation:
    """
    The scores a document received across a rubric's dimensions.
    """

    document: Document
    """
    The document that was evaluated.
    """

    dimension_scores: tuple[DimensionScore, ...]
    """
    One score per dimension in the rubric, in rubric order.
    """


# %% errors


class DocumentEvaluationError(Exception):
    """
    Raised when the TypeSafe API fails to evaluate a document.
    """

    def __init__(self, document_path: Path, cause: TypeSafeError) -> None:
        super().__init__(f"Evaluation failed for {document_path}: {cause}")
        self.document_path = document_path
        """
        Path of the document that failed to evaluate.
        """
        self.cause = cause
        """
        The underlying TypeSafe SDK error.
        """


# %% DocumentEvaluator


@dataclass(frozen=True)
class DocumentEvaluator:
    """
    Scores documents against an :class:`EvaluationRubric` using the TypeSafe API.
    """

    client: TypeSafeClient
    """
    The TypeSafe API client used to evaluate documents.
    """

    rubric: EvaluationRubric
    """
    The dimensions every document is scored against.
    """

    def evaluate(self, document: Document) -> DocumentEvaluation:
        """
        Score a single document against every dimension in the rubric.

        :param document: the document to evaluate
        :raises DocumentEvaluationError: if the TypeSafe API call fails
        :return: the document's evaluation
        """
        questions = {
            dimension.name: dimension.to_score_question()
            for dimension in self.rubric.dimensions
        }
        state = {"document": {"title": document.path.name, "text": document.text}}
        try:
            response = self.client.system_one(state=state, questions=questions)
        except TypeSafeError as error:
            raise DocumentEvaluationError(document.path, error) from error

        dimension_scores = tuple(
            DimensionScore(
                dimension=dimension,
                score=response.scores[dimension.name].score,
                confidence=response.scores[dimension.name].confidence,
            )
            for dimension in self.rubric.dimensions
        )
        return DocumentEvaluation(document=document, dimension_scores=dimension_scores)
