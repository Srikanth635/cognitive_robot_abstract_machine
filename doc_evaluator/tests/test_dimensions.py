"""
Tests for `Dimension` and `EvaluationRubric`.
"""

from __future__ import annotations

from doc_evaluator.dimensions import Dimension, EvaluationRubric

# %% Dimension.to_score_question


def test_to_score_question_carries_instructions_and_criteria() -> None:
    dimension = Dimension(
        name="clarity",
        instructions="How clear is it?",
        criteria=("low", "medium", "high"),
    )

    question = dimension.to_score_question()

    assert question.type == "score"
    assert question.instructions == "How clear is it?"
    assert list(question.criteria) == ["low", "medium", "high"]


# %% EvaluationRubric.research_report


def test_research_report_rubric_has_clarity_and_rigor_dimensions() -> None:
    rubric = EvaluationRubric.research_report()

    names = tuple(dimension.name for dimension in rubric.dimensions)

    assert names == ("Clarity / readability", "Factual accuracy / rigor")


def test_research_report_rubric_dimensions_have_non_empty_ordered_criteria() -> None:
    rubric = EvaluationRubric.research_report()

    for dimension in rubric.dimensions:
        assert len(dimension.criteria) >= 2
        assert all(isinstance(level, str) and level for level in dimension.criteria)
        assert len(set(dimension.criteria)) == len(dimension.criteria)
