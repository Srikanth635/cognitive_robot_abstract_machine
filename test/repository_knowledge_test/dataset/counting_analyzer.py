"""
Observe which source files require parsing during refresh.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from repository_knowledge.contracts import SourceAnalyzer
from repository_knowledge.model import FileExtraction, SourceRecord

# %% extraction accounting


@dataclass
class CountingAnalyzer(SourceAnalyzer):
    """
    Retain the source paths processed by a delegated analyzer.
    """

    analyzer: SourceAnalyzer
    """The source analysis behavior being measured."""

    analyzed_paths: list[str] = field(default_factory=list)
    """
    The paths passed to the analyzer.
    """

    def extract(self, source: SourceRecord) -> FileExtraction:
        self.analyzed_paths.append(source.relative_path)
        return self.analyzer.extract(source)
