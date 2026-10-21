"""Compare retrieval interfaces against pinned, independent source expectations."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import time
import tomllib
from dataclasses import asdict, dataclass, field, fields
from enum import StrEnum
from pathlib import Path

from typing_extensions import TYPE_CHECKING

from repository_knowledge.adapters.git_source import GitArgument, GitSource
from repository_knowledge.adapters.serialization import RecordCodec
from repository_knowledge.adapters.sqlite_index import SQLiteIndex
from repository_knowledge.assistance import ContextRole, TaskContext
from repository_knowledge.exceptions import InvalidQuery
from repository_knowledge.interfaces.cli import Command, Option
from repository_knowledge.model import (
    QueryPage,
    RepositorySnapshot,
    Resolution,
    SourceLocation,
)
from repository_knowledge.operations.freshness import FreshnessCheck

if TYPE_CHECKING:
    from typing_extensions import Any

# %% evaluation vocabulary


class RetrievalMethod(StrEnum):
    """The agent-visible retrieval operation under comparison."""

    SOURCE = "source-search"
    STRUCTURE = "structural-query"
    CURATED = "structural-curated"


class EvaluationField(StrEnum):
    """Fields in the operator's evaluation task manifest."""

    SYMBOL = "symbol"
    SOURCE_PATH = "source_path"
    START_LINE = "start_line"
    REQUIRED_PATHS = "required_paths"
    REVISION = "revision"
    NAVIGATION = "navigation"


class EvaluationArgument(StrEnum):
    """Evaluation-specific process arguments and source selection."""

    SEARCH = "grep"
    NO_COLOR = "--no-color"
    LINE_NUMBER = "-n"
    EXTENDED_PATTERN = "-E"
    PATTERN = "-e"
    PATHS = "--"
    PYTHON_FILES = "*.py"
    EXCLUDE_GENERATED = ":(exclude)**/ormatic_interface.py"
    MODULE = "-m"
    APPLICATION = "repository_knowledge"
    TASKS = "--tasks"
    OUTPUT = "--output"


@dataclass(frozen=True)
class NavigationTask:
    """A source location selected independently of the queried index."""

    symbol: str
    """The qualified declaration requested by the agent."""
    source_path: str
    """The expected definition path, empty for an absent declaration."""
    start_line: int
    """The expected declaration line, zero for an absent declaration."""
    required_paths: tuple[str, ...] = ()
    """Additional independently selected evidence paths for recall scoring."""

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> NavigationTask:
        """Read a task from the evaluation manifest."""
        return cls(
            data[EvaluationField.SYMBOL],
            data[EvaluationField.SOURCE_PATH],
            data[EvaluationField.START_LINE],
            tuple(data.get(EvaluationField.REQUIRED_PATHS, [])),
        )


@dataclass
class RetrievalEvidence:
    """Verified evidence and visible uncertainty in one retrieval response."""

    locations: list[SourceLocation]
    """The returned definition locations."""
    cited_paths: set[str]
    """Paths actually cited in the returned evidence."""
    valid_citations: int
    """Citations consistent with the independent committed snapshot."""
    total_citations: int
    """All citations checked, including duplicates."""
    curated_items: int = 0
    """Included concepts, without any claim of authenticated review."""
    uncertain_items: int = 0
    """Returned items explicitly lacking resolved source relationships."""


@dataclass
class RetrievalObservation:
    """A measured retrieval result, separate from model task performance."""

    method: RetrievalMethod
    """The measured retrieval interface."""
    symbol: str
    """The task's requested declaration."""
    source_revision: str
    """The immutable source used by all compared methods."""
    seconds: float
    """Wall time through response receipt, including interpreter startup."""
    audit_seconds: float
    """Additional independent scoring time excluded from retrieval latency."""
    response_bytes: int
    """Uncompressed UTF-8 response bytes, not model tokens."""
    agent_tool_calls: int
    """Agent-visible requests, excluding the evaluator's source audit."""
    exit_status: int
    """The retrieval subprocess exit code."""
    definition_found: bool
    """Whether the returned evidence contains the expected path and line."""
    returned_locations: int
    """The number of returned definition locations."""
    required_paths_found: int
    """Independently required paths present in the response citations."""
    required_paths_total: int
    """The number of independently required paths."""
    valid_citations: int
    """Citations passing the independent source audit."""
    total_citations: int
    """Citations subjected to that audit."""
    curated_items: int
    """Concepts included by the selected local bundle."""
    uncertain_items: int
    """Returned items marked unresolved or uncertain."""
    agent_tokens: int | None = None
    """Unmeasured until an instrumented model task supplies token usage."""
    generated_task_success: bool | None = None
    """Unmeasured until generated work passes the task's independent checks."""


# %% independent source verification


@dataclass
class CitationAudit:
    """Validate returned evidence against committed blobs outside the index."""

    snapshot: RepositorySnapshot
    """The source captured independently for evaluation."""

    def contains_location(self, location: SourceLocation) -> bool:
        """Check that a citation's entire range exists in captured source."""
        return any(
            source.relative_path == location.relative_path
            and 1
            <= location.start_line
            <= location.end_line
            <= len(source.source_text.splitlines())
            for source in self.snapshot.sources
        )

    def check_query(self, result: QueryPage) -> RetrievalEvidence:
        """Check the snapshot identity and exact excerpt text of structural results."""
        valid = sum(
            result.snapshot_identifier == self.snapshot.manifest.snapshot_identifier
            and self.contains_location(excerpt.location)
            and any(
                source.relative_path == excerpt.location.relative_path
                and excerpt.source_text
                == "\n".join(
                    source.source_text.splitlines()[
                        excerpt.location.start_line - 1 : excerpt.location.end_line
                    ]
                )
                for source in self.snapshot.sources
            )
            for excerpt in result.excerpts
        )
        return RetrievalEvidence(
            [symbol.location for symbol in result.symbols],
            {excerpt.location.relative_path for excerpt in result.excerpts},
            valid,
            len(result.excerpts),
            uncertain_items=sum(
                reference.resolution != Resolution.RESOLVED
                for reference in result.references
            ),
        )

    def check_context(self, result: TaskContext) -> RetrievalEvidence:
        """Check every citation's source, snapshot, file digest and line extent."""
        citations = [citation for item in result.items for citation in item.citations]
        valid = sum(
            result.manifest.snapshot_identifier
            == self.snapshot.manifest.snapshot_identifier
            and citation.snapshot_identifier
            == self.snapshot.manifest.snapshot_identifier
            and citation.source_commit == self.snapshot.manifest.commit_identifier
            and self.contains_location(citation.location)
            and any(
                source.relative_path == citation.location.relative_path
                and source.content_digest == citation.content_digest
                for source in self.snapshot.sources
            )
            for citation in citations
        )
        return RetrievalEvidence(
            [
                citation.location
                for item in result.items
                if item.role == ContextRole.DECLARATION
                for citation in item.citations
            ],
            {citation.location.relative_path for citation in citations},
            valid,
            len(citations),
            sum(item.role == ContextRole.CONCEPT for item in result.items),
            sum(item.resolution != Resolution.RESOLVED for item in result.items),
        )

    def check_search(self, response: str) -> RetrievalEvidence:
        """Check the paths and line numbers returned by committed Git search."""
        locations = []
        valid_citations = 0
        for match in response.splitlines():
            revision, relative_path, line, text = match.split(":", 3)
            location = SourceLocation(relative_path, int(line), int(line))
            if revision != self.snapshot.manifest.commit_identifier:
                raise InvalidQuery("Source search returned another revision.")
            locations.append(location)
            valid_citations += self.contains_location(location) and any(
                source.relative_path == relative_path
                and source.source_text.splitlines()[location.start_line - 1] == text
                for source in self.snapshot.sources
            )
        return RetrievalEvidence(
            locations,
            {location.relative_path for location in locations},
            valid_citations,
            len(locations),
        )


# %% timed client requests


@dataclass
class RetrievalEvaluation:
    """Measure actual CLI requests against one pinned source revision."""

    repository_root: Path
    """The checkout supplying committed objects."""
    database_path: Path
    """The already built committed-source cache."""
    revision: str
    """The source revision, resolved once before any trial."""
    bundle_path: Path | None = None
    """The explicitly selected local draft bundle, if any."""
    timeout_seconds: float = 300.0
    """The deadline for a single retrieval process."""
    character_budget: int = 60000
    """The curated-context payload allowance."""
    audit: CitationAudit = field(init=False)
    """The independent source evidence used to score responses."""

    def __post_init__(self) -> None:
        """Pin and check source identity before measuring any retrieval."""
        source = GitSource(self.repository_root)
        self.revision = source.resolve_revision(self.revision)
        FreshnessCheck(SQLiteIndex(self.database_path), source).check(
            self.revision
        ).require_current()
        self.audit = CitationAudit(source.read_snapshot(self.revision))

    def prepare_arguments(
        self, task: NavigationTask, method: RetrievalMethod
    ) -> list[str]:
        """Select a committed-source retrieval command without supplying its answer path."""
        if method == RetrievalMethod.SOURCE:
            identifier = re.escape(task.symbol.rsplit(".", 1)[-1])
            return [
                GitArgument.PROGRAM,
                GitArgument.DIRECTORY,
                str(self.repository_root),
                EvaluationArgument.SEARCH,
                EvaluationArgument.NO_COLOR,
                EvaluationArgument.LINE_NUMBER,
                EvaluationArgument.EXTENDED_PATTERN,
                EvaluationArgument.PATTERN,
                rf"^[[:space:]]*(class|def|async def)[[:space:]]+{identifier}([[:space:]]|\(|:)",
                self.revision,
                EvaluationArgument.PATHS,
                EvaluationArgument.PYTHON_FILES,
                EvaluationArgument.EXCLUDE_GENERATED,
            ]
        arguments = [
            sys.executable,
            EvaluationArgument.MODULE,
            EvaluationArgument.APPLICATION,
            Option.REPOSITORY,
            str(self.repository_root),
            Option.DATABASE,
            str(self.database_path),
            Option.REVISION,
            self.revision,
        ]
        if method == RetrievalMethod.STRUCTURE:
            return [*arguments, Command.DESCRIBE_SYMBOL, task.symbol]
        arguments.extend(
            [
                Command.PREPARE_CONTEXT,
                task.symbol,
                Option.CHARACTERS,
                str(self.character_budget),
            ]
        )
        if self.bundle_path is not None:
            arguments.extend([Option.BUNDLE, str(self.bundle_path)])
        return arguments

    def measure(
        self, task: NavigationTask, method: RetrievalMethod
    ) -> RetrievalObservation:
        """Execute and independently score one navigation retrieval."""
        arguments = self.prepare_arguments(task, method)
        started = time.perf_counter()
        process = subprocess.run(
            arguments, capture_output=True, timeout=self.timeout_seconds
        )
        seconds = time.perf_counter() - started
        audit_started = time.perf_counter()
        evidence = RetrievalEvidence([], set(), 0, 0)
        if method == RetrievalMethod.SOURCE and process.returncode in (0, 1):
            evidence = self.audit.check_search(process.stdout.decode("utf-8"))
        elif process.returncode == 0:
            if method == RetrievalMethod.CURATED:
                evidence = self.audit.check_context(
                    RecordCodec().decode(process.stdout.decode("utf-8"), TaskContext)
                )
            else:
                evidence = self.audit.check_query(
                    RecordCodec().decode(process.stdout.decode("utf-8"), QueryPage)
                )
        required = set(task.required_paths) | (
            {task.source_path} if task.source_path else set()
        )
        return RetrievalObservation(
            method,
            task.symbol,
            self.revision,
            seconds,
            time.perf_counter() - audit_started,
            len(process.stdout),
            1,
            process.returncode,
            any(
                location.relative_path == task.source_path
                and location.start_line == task.start_line
                for location in evidence.locations
            ),
            len(evidence.locations),
            len(required & evidence.cited_paths),
            len(required),
            evidence.valid_citations,
            evidence.total_citations,
            evidence.curated_items,
            evidence.uncertain_items,
        )


# %% reproducible evaluation entry


def main() -> int:
    """Write one CSV observation per task and method without invoking a model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(Option.REPOSITORY, type=Path, required=True)
    parser.add_argument(Option.DATABASE, type=Path, required=True)
    parser.add_argument(EvaluationArgument.TASKS, type=Path, required=True)
    parser.add_argument(EvaluationArgument.OUTPUT, type=Path, required=True)
    parser.add_argument(Option.BUNDLE, type=Path)
    arguments = parser.parse_args()
    with arguments.tasks.open("rb") as stream:
        configuration = tomllib.load(stream)
    evaluator = RetrievalEvaluation(
        arguments.repository,
        arguments.database,
        configuration[EvaluationField.REVISION],
        arguments.bundle,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    failed = False
    with arguments.output.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[attribute.name for attribute in fields(RetrievalObservation)],
        )
        writer.writeheader()
        for record in configuration[EvaluationField.NAVIGATION]:
            task = NavigationTask.from_json(record)
            for method in RetrievalMethod:
                if not task.source_path and method == RetrievalMethod.CURATED:
                    continue
                observation = evaluator.measure(task, method)
                writer.writerow(asdict(observation))
                stream.flush()
                print(
                    f"{method}: {task.symbol}: {observation.seconds:.3f}s, definition={observation.definition_found}, citations={observation.valid_citations}/{observation.total_citations}",
                    flush=True,
                )
                failed |= (
                    observation.valid_citations != observation.total_citations
                    or observation.required_paths_found
                    != observation.required_paths_total
                    or (bool(task.source_path) != observation.definition_found)
                    or (not task.source_path and observation.returned_locations != 0)
                    or observation.exit_status
                    not in ((0, 1) if method == RetrievalMethod.SOURCE else (0,))
                )
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
