"""
Invalid source identities, query requests and index state.
"""

from __future__ import annotations

from dataclasses import dataclass

from krrood.exceptions import DataclassException

# %% source identity errors


@dataclass
class InvalidConceptPath(DataclassException):
    """
    Reject a concept path outside its bundle's document namespace.
    """

    relative_path: str
    """The rejected bundle-relative identity."""

    def error_message(self) -> str:
        return f"Invalid concept path: {self.relative_path}."

    def suggest_correction(self) -> str:
        return "Use a normalized relative Markdown path excluding index.md and log.md."


@dataclass
class SourceChangedDuringCapture(DataclassException):
    """
    Reject source that changed while a coherent snapshot was captured.
    """

    source_path: str
    """The affected source path or checkout description."""

    def error_message(self) -> str:
        return f"Source changed during capture: {self.source_path}."

    def suggest_correction(self) -> str:
        return "Let the edit or checkout operation finish and retry capture."


@dataclass
class SourceSnapshotMismatch(DataclassException):
    """
    Reject saved working-file evidence that no longer matches source.
    """

    indexed_snapshot: str
    """The identity represented by saved evidence."""

    requested_snapshot: str
    """
    The identity captured from current working files.
    """

    def error_message(self) -> str:
        return f"Saved working-file snapshot {self.indexed_snapshot} differs from {self.requested_snapshot}."

    def suggest_correction(self) -> str:
        return "Refresh the overlay before requesting current source evidence."


@dataclass
class SourceScopeMismatch(DataclassException):
    """
    Reject mixing a committed cache and a working-file cache.
    """

    indexed_scope: str
    """The source scope already stored at the cache path."""

    requested_scope: str
    """
    The source scope requested for the operation.
    """

    def error_message(self) -> str:
        return f"Cache scope {self.indexed_scope} differs from requested scope {self.requested_scope}."

    def suggest_correction(self) -> str:
        return "Use a separate database for each source scope."


@dataclass
class InvalidUntrackedPath(DataclassException):
    """
    Reject an inclusion path that escapes the selected repository.
    """

    relative_path: str
    """The rejected inclusion path."""

    def error_message(self) -> str:
        return f"Invalid untracked inclusion path: {self.relative_path}."

    def suggest_correction(self) -> str:
        return (
            "Select a repository-relative file or directory without parent traversal."
        )


@dataclass
class UnmergedSourceIndex(DataclassException):
    """
    Reject an index containing unresolved merge stages.
    """

    relative_path: str
    """The tracked path containing an unresolved merge."""

    def error_message(self) -> str:
        return f"Unresolved Git index stages exist for {self.relative_path}."

    def suggest_correction(self) -> str:
        return "Resolve the merge before capturing current-source evidence."


@dataclass
class InvalidSourceLocation(DataclassException):
    """
    Reject a location outside a snapshot or an invalid inclusive line range.
    """

    relative_path: str
    """The rejected repository-relative path."""

    start_line: int
    """
    The rejected first line.
    """

    end_line: int
    """The rejected final line."""

    def error_message(self) -> str:
        return f"Invalid source location: {self.relative_path}:{self.start_line}-{self.end_line}."

    def suggest_correction(self) -> str:
        return "Use a repository-relative POSIX path and a positive ordered line range."


@dataclass
class GitOperationFailed(DataclassException):
    """
    Report a source operation that Git could not complete.
    """

    command_arguments: tuple[str, ...]
    """The failed Git arguments."""

    diagnostic: str
    """
    Git's error output.
    """

    def error_message(self) -> str:
        return f"Git could not complete {self.command_arguments}: {self.diagnostic}"

    def suggest_correction(self) -> str:
        return "Check the repository path and select an existing commit."


@dataclass
class InvalidQuery(DataclassException):
    """
    Reject a query whose bounds cannot be honoured.
    """

    explanation: str
    """The violated query constraint."""

    def error_message(self) -> str:
        return self.explanation

    def suggest_correction(self) -> str:
        return "Use nonnegative offsets and positive bounded result limits."


@dataclass
class UnsupportedIndexFormat(DataclassException):
    """
    Reject persisted data that this implementation cannot interpret.
    """

    actual_version: int
    """The stored schema version."""

    def error_message(self) -> str:
        return f"Unsupported source-index format: {self.actual_version}."

    def suggest_correction(self) -> str:
        return "Rebuild the disposable index using this implementation."


@dataclass
class InvalidSerializedType(DataclassException):
    """
    Reject a serialized type outside the source-index data vocabulary.
    """

    serialized_type: str
    """The rejected type or collection identity."""

    def error_message(self) -> str:
        return f"Unrecognized index data type: {self.serialized_type}."

    def suggest_correction(self) -> str:
        return "Rebuild the index from trusted source with the configured interpreter."


@dataclass
class IndexUnavailable(DataclassException):
    """
    Report that no complete source index is available at the selected path.
    """

    database_path: str
    """The requested index location."""

    def error_message(self) -> str:
        return f"No completed source index exists at {self.database_path}."

    def suggest_correction(self) -> str:
        return "Build a committed-source index before running navigation queries."


@dataclass
class SourceRevisionMismatch(DataclassException):
    """
    Reject source evidence captured for a different requested commit.
    """

    indexed_commit: str
    """The source revision represented by saved evidence."""

    requested_commit: str
    """
    The revision selected by the consumer.
    """

    def error_message(self) -> str:
        return f"Indexed commit {self.indexed_commit} differs from requested commit {self.requested_commit}."

    def suggest_correction(self) -> str:
        return "Rebuild the index or explicitly select its indexed revision."
