"""
Scoring rubrics: the dimensions documents are evaluated against.
"""

from __future__ import annotations

from dataclasses import dataclass

from typesafe_sdk import Score

# %% Dimension


@dataclass(frozen=True)
class Dimension:
    """
    A single ordered scale a document is rated against.
    """

    name: str
    """
    Short label for this dimension; used as the question key and as a report column
    header.
    """

    instructions: str
    """
    The judgment to make, describing what to look for in the document.
    """

    criteria: tuple[str, ...]
    """
    Ordered rubric levels, low to high; each describes a concrete situation, not a
    degree.
    """

    def to_score_question(self) -> Score:
        """
        Build the TypeSafe :class:`Score` question that asks this dimension's judgment.

        :return: a Score question with this dimension's instructions and rubric levels
        """
        return Score(instructions=self.instructions, criteria=list(self.criteria))


# %% EvaluationRubric


@dataclass(frozen=True)
class EvaluationRubric:
    """
    A set of dimensions every document in a batch is scored against.
    """

    dimensions: tuple[Dimension, ...]
    """
    The dimensions to score, in the order they should be reported.
    """

    @classmethod
    def research_report(cls) -> EvaluationRubric:
        """
        The rubric for evaluating research and technical reports.

        Scores clarity and evidential rigor: whether the document is clearly written,
        and whether its own claims are backed by evidence, citation, or explicit
        reasoning presented in the text itself.

        :return: a rubric with a clarity dimension and a rigor dimension
        """
        return cls(
            dimensions=(
                Dimension(
                    name="Clarity / readability",
                    instructions=(
                        "How clearly does the document communicate its content? Judge the "
                        "writing itself: organization, definition and consistent use of terms, "
                        "and whether a reader can follow the argument without rereading."
                    ),
                    criteria=(
                        "The writing is dense or disorganized: undefined jargon, unclear "
                        "antecedents, or a structure that forces rereading to follow the "
                        "argument.",
                        "The writing is mostly followable, but some passages are convoluted or "
                        "the argument assumes background the document never states.",
                        "The writing is clear and logically ordered: terms are defined or used "
                        "consistently, and each section's purpose is evident.",
                        "A reader unfamiliar with the topic could follow the argument on a "
                        "single read: no undefined terms, no forced backtracking, and the "
                        "structure mirrors the logic of the argument.",
                    ),
                ),
                Dimension(
                    name="Factual accuracy / rigor",
                    instructions=(
                        "How well does the document support its own substantive claims? Judge "
                        "whether claims are backed by citations, data, or explicit reasoning "
                        "presented in the document, and whether that reasoning is internally "
                        "consistent, not whether the claims are true against outside knowledge."
                    ),
                    criteria=(
                        "Substantive claims are unsupported or contradicted elsewhere in the "
                        "document, with no citation, data, or explicit reasoning offered for "
                        "them.",
                        "Some claims are backed by evidence or reasoning, but key assertions "
                        "rely on unstated assumptions or are asserted without support.",
                        "Most substantive claims are backed by citations, data, or explicit "
                        "reasoning, with at most minor unsupported assertions.",
                        "Every substantive claim is backed by citation, data, or explicit "
                        "derivation, and the reasoning is internally consistent throughout.",
                    ),
                ),
            )
        )
