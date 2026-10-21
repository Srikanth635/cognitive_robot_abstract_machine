"""
Rendering evaluation results as a console table.
"""

from __future__ import annotations

from dataclasses import dataclass

from doc_evaluator.evaluation import (
    DimensionScore,
    DocumentEvaluation,
    DocumentEvaluationError,
)

# %% TextTable


@dataclass(frozen=True)
class TextTable:
    """
    A left-aligned, whitespace-padded text table.
    """

    headers: tuple[str, ...]
    """
    Column headers.
    """

    rows: tuple[tuple[str, ...], ...]
    """
    Table body; each row must have as many cells as there are headers.
    """

    def render(self) -> str:
        """
        Render the table as aligned, newline-joined text.

        :return: the rendered table, one line per header row and body row
        """
        columns = (self.headers, *self.rows)
        widths = [
            max(len(row[index]) for row in columns)
            for index in range(len(self.headers))
        ]
        lines = [self._render_row(self.headers, widths), self._render_separator(widths)]
        lines.extend(self._render_row(row, widths) for row in self.rows)
        return "\n".join(lines)

    @staticmethod
    def _render_row(row: tuple[str, ...], widths: list[int]) -> str:
        return " | ".join(cell.ljust(width) for cell, width in zip(row, widths))

    @staticmethod
    def _render_separator(widths: list[int]) -> str:
        return "-+-".join("-" * width for width in widths)


# %% EvaluationReport


@dataclass(frozen=True)
class EvaluationReport:
    """
    A batch of document evaluations, ready to render as a table.
    """

    evaluations: tuple[DocumentEvaluation, ...]
    """
    Documents that were evaluated successfully.
    """

    failures: tuple[DocumentEvaluationError, ...]
    """
    Documents that failed to evaluate.
    """

    def render(self) -> str:
        """
        Render the successful evaluations as a table, followed by any failures.

        :return: the rendered report
        """
        sections = [self._render_table()] if self.evaluations else []
        if self.failures:
            sections.append(self._render_failures())
        return "\n\n".join(sections)

    def _render_table(self) -> str:
        dimension_names = tuple(
            score.dimension.name for score in self.evaluations[0].dimension_scores
        )
        headers = ("Document", *dimension_names)
        rows = tuple(
            (
                evaluation.document.path.name,
                *(self._render_cell(score) for score in evaluation.dimension_scores),
            )
            for evaluation in self.evaluations
        )
        return TextTable(headers=headers, rows=rows).render()

    @staticmethod
    def _render_cell(score: DimensionScore) -> str:
        return f"{score.score:.2f}/{score.maximum_score:.0f} ({score.confidence:.2f})"

    def _render_failures(self) -> str:
        lines = ["Failed to evaluate:"]
        lines.extend(
            f"  {failure.document_path}: {failure.cause}" for failure in self.failures
        )
        return "\n".join(lines)
