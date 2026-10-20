"""
Parse and dispatch bounded committed-source navigation commands.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path

from typing_extensions import TYPE_CHECKING

from repository_knowledge.contracts import IndexWriter
from repository_knowledge.assistance import TaskProfile
from repository_knowledge.exceptions import InvalidQuery, SourceScopeMismatch
from repository_knowledge.model import (
    FreshnessReport,
    IndexManifest,
    IndexSelection,
    QueryPage,
    RefreshReport,
    SelectionKind,
    TraversalDirection,
    SourceScope,
)
from repository_knowledge.operations.freshness import FreshnessCheck
from repository_knowledge.operations.indexing import IndexBuilder
from repository_knowledge.operations.queries import IndexContents, Navigation
from repository_knowledge.operations.refresh import IndexRefresh

if TYPE_CHECKING:
    from typing_extensions import Any

# %% command contract


class Command(StrEnum):
    """
    The source-navigation operations exposed by the initial CLI.
    """

    INDEX = "index"
    REFRESH = "refresh"
    RECONCILE = "reconcile"
    FIND_SYMBOLS = "find-symbols"
    DESCRIBE_SYMBOL = "describe-symbol"
    FIND_REFERENCES = "find-references"
    FIND_RELATIONSHIPS = "find-relationships"
    CHECK_FRESHNESS = "check-freshness"
    FIND_EXAMPLES = "find-examples"
    FIND_TESTS = "find-tests"
    PREPARE_CONTEXT = "prepare-context"
    ASSESS_CHANGE = "assess-change"


class Option(StrEnum):
    """
    Named arguments accepted by the source-navigation interface.
    """

    REPOSITORY = "--repository"
    DATABASE = "--database"
    REVISION = "--revision"
    SCOPE = "--scope"
    INCLUDE_UNTRACKED = "--include-untracked"
    LIMIT = "--limit"
    OFFSET = "--offset"
    DEPTH = "--depth"
    DIRECTION = "--direction"
    PROFILE = "--profile"
    CHARACTERS = "--characters"
    BUNDLE = "--bundle"
    AGAINST_DATABASE = "--against-database"
    AGAINST_REVISION = "--against-revision"
    ORM = "--orm"


class ArgumentField(StrEnum):
    """
    Keys emitted by the argument parser.
    """

    COMMAND = "command"
    REPOSITORY = "repository_root"
    DATABASE = "database_path"
    REVISION = "revision"
    SCOPE = "source_scope"
    INCLUDE_UNTRACKED = "included_untracked_paths"
    QUERY = "query"
    SYMBOL = "symbol_identifier"
    LIMIT = "limit"
    OFFSET = "offset"
    DEPTH = "depth"
    DIRECTION = "direction"
    PROFILE = "profile"
    CHARACTERS = "character_budget"
    BUNDLE = "bundle_path"
    AGAINST_DATABASE = "baseline_database"
    AGAINST_REVISION = "baseline_revision"
    ORM = "orm_task"


class LayoutPath(StrEnum):
    """
    Locations contained in the repository-knowledge parent directory.
    """

    PACKAGE = "repository_knowledge"
    CACHE = ".cache"
    DATABASE = "index.sqlite3"
    OVERLAY_DATABASE = "working-tree.sqlite3"


class RevisionSelector(StrEnum):
    """
    The default committed source selected by a checkout.
    """

    CHECKOUT = "HEAD"


@dataclass(frozen=True)
class CommandArguments:
    """
    The interpreted request accepted by source-navigation commands.
    """

    command: Command
    """
    The requested navigation or indexing operation.
    """

    repository_root: Path
    """
    The checkout supplying committed source.
    """

    database_path: Path
    """
    The rebuildable index being used.
    """

    revision: str
    """
    The requested committed source selector.
    """

    query: str
    """
    Text used to find declarations.
    """

    symbol_identifier: str
    """
    The qualified declaration selected for navigation.
    """

    limit: int
    """
    The requested evidence page size.
    """

    offset: int
    """
    The requested evidence page offset.
    """

    depth: int
    """
    The requested relationship distance.
    """

    direction: TraversalDirection
    """
    Whether traversal follows incoming or outgoing relationships.
    """

    source_scope: SourceScope = SourceScope.COMMITTED
    """
    Whether source evidence represents committed objects or working files.
    """

    included_untracked_paths: tuple[str, ...] = ()
    """
    Explicit repository-relative untracked files or directories.
    """

    profile: TaskProfile = TaskProfile.CODE_EXTENSION
    """The task purpose controlling context priority."""
    character_budget: int = 16000
    """The maximum context payload characters, excluding JSON metadata."""
    bundle_path: Path | None = None
    """An optional knowledge bundle supplying curated context."""
    baseline_database: Path | None = None
    """The explicitly selected committed comparison index."""
    baseline_revision: str = ""
    """The revision required of the baseline comparison index."""
    orm_task: bool = False
    """Whether context should include ORM generator source and inputs."""

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> CommandArguments:
        """
        Interpret the parser's request shape at the interface boundary.
        """
        repository_root = data[ArgumentField.REPOSITORY].resolve()
        source_scope = SourceScope(data[ArgumentField.SCOPE])
        included_untracked_paths = tuple(data[ArgumentField.INCLUDE_UNTRACKED] or ())
        if source_scope == SourceScope.COMMITTED and included_untracked_paths:
            raise InvalidQuery(
                "Untracked source inclusion requires working-tree scope."
            )
        database_filename = (
            LayoutPath.DATABASE
            if source_scope == SourceScope.COMMITTED
            else LayoutPath.OVERLAY_DATABASE
        )
        database_path = (
            data[ArgumentField.DATABASE]
            or repository_root
            / LayoutPath.PACKAGE
            / LayoutPath.CACHE
            / database_filename
        )
        return cls(
            Command(data[ArgumentField.COMMAND]),
            repository_root,
            database_path.resolve(),
            data[ArgumentField.REVISION],
            data[ArgumentField.QUERY],
            data[ArgumentField.SYMBOL],
            data[ArgumentField.LIMIT],
            data[ArgumentField.OFFSET],
            data[ArgumentField.DEPTH],
            TraversalDirection(data[ArgumentField.DIRECTION]),
            source_scope,
            included_untracked_paths,
            TaskProfile(data[ArgumentField.PROFILE]),
            data[ArgumentField.CHARACTERS],
            data[ArgumentField.BUNDLE],
            data[ArgumentField.AGAINST_DATABASE],
            data[ArgumentField.AGAINST_REVISION],
            data[ArgumentField.ORM],
        )

    def select_evidence(self, maximum_depth: int) -> IndexSelection:
        """
        Translate the interface request into a source-evidence selection.
        """
        kinds = {
            Command.CHECK_FRESHNESS: SelectionKind.MANIFEST,
            Command.FIND_SYMBOLS: SelectionKind.SYMBOL_SEARCH,
            Command.DESCRIBE_SYMBOL: SelectionKind.SYMBOL_DETAILS,
            Command.FIND_REFERENCES: SelectionKind.REFERENCES,
            Command.FIND_RELATIONSHIPS: SelectionKind.RELATIONSHIPS,
        }
        return IndexSelection(
            kinds[self.command],
            self.symbol_identifier,
            self.query,
            self.direction,
            self.depth,
            maximum_depth,
        )


@dataclass
class CommandParser:
    """
    Define the explicit committed-source command interface.
    """

    default_limit: int = 20
    """
    The initial number of evidence items per query.
    """

    def parse(self, arguments: list[str]) -> CommandArguments:
        """
        Parse a command without loading repository or storage adapters.
        """
        parser = argparse.ArgumentParser(
            prog="cram-knowledge",
            description="Query committed source or an explicitly selected working-tree overlay.",
        )
        parser.add_argument(
            Option.REPOSITORY, dest=ArgumentField.REPOSITORY, type=Path, required=True
        )
        parser.add_argument(Option.DATABASE, dest=ArgumentField.DATABASE, type=Path)
        parser.add_argument(Option.REVISION, default=RevisionSelector.CHECKOUT)
        parser.add_argument(
            Option.SCOPE,
            dest=ArgumentField.SCOPE,
            choices=list(SourceScope),
            default=SourceScope.COMMITTED,
        )
        parser.add_argument(
            Option.INCLUDE_UNTRACKED,
            dest=ArgumentField.INCLUDE_UNTRACKED,
            action="append",
        )
        parser.set_defaults(
            query="",
            symbol_identifier="",
            limit=self.default_limit,
            offset=0,
            depth=1,
            direction=TraversalDirection.OUTGOING,
            profile=TaskProfile.CODE_EXTENSION,
            character_budget=16000,
            bundle_path=None,
            baseline_database=None,
            baseline_revision="",
            orm_task=False,
        )
        commands = parser.add_subparsers(dest=ArgumentField.COMMAND, required=True)
        for command in Command:
            command_parser = commands.add_parser(command)
            if command == Command.FIND_SYMBOLS:
                command_parser.add_argument(ArgumentField.QUERY)
            elif command in (
                Command.DESCRIBE_SYMBOL,
                Command.FIND_REFERENCES,
                Command.FIND_RELATIONSHIPS,
                Command.FIND_EXAMPLES,
                Command.FIND_TESTS,
                Command.PREPARE_CONTEXT,
                Command.ASSESS_CHANGE,
            ):
                command_parser.add_argument(ArgumentField.SYMBOL)
            if command in (
                Command.FIND_SYMBOLS,
                Command.DESCRIBE_SYMBOL,
                Command.FIND_REFERENCES,
                Command.FIND_RELATIONSHIPS,
                Command.FIND_EXAMPLES,
                Command.FIND_TESTS,
                Command.ASSESS_CHANGE,
            ):
                command_parser.add_argument(
                    Option.LIMIT, type=int, default=self.default_limit
                )
                command_parser.add_argument(Option.OFFSET, type=int, default=0)
            if command == Command.PREPARE_CONTEXT:
                command_parser.add_argument(
                    Option.ORM, dest=ArgumentField.ORM, action="store_true"
                )
                command_parser.add_argument(
                    Option.PROFILE,
                    choices=list(TaskProfile),
                    default=TaskProfile.CODE_EXTENSION,
                )
                command_parser.add_argument(
                    Option.CHARACTERS,
                    dest=ArgumentField.CHARACTERS,
                    type=int,
                    default=16000,
                )
                command_parser.add_argument(
                    Option.BUNDLE, dest=ArgumentField.BUNDLE, type=Path
                )
            if command == Command.ASSESS_CHANGE:
                command_parser.add_argument(
                    Option.AGAINST_DATABASE,
                    dest=ArgumentField.AGAINST_DATABASE,
                    type=Path,
                    required=True,
                )
                command_parser.add_argument(
                    Option.AGAINST_REVISION,
                    dest=ArgumentField.AGAINST_REVISION,
                    required=True,
                )
            if command == Command.FIND_RELATIONSHIPS:
                command_parser.add_argument(Option.DEPTH, type=int, default=1)
                command_parser.add_argument(
                    Option.DIRECTION,
                    choices=list(TraversalDirection),
                    default=TraversalDirection.OUTGOING,
                )
        return CommandArguments.from_json(vars(parser.parse_args(arguments)))


# %% operation dispatch


@dataclass
class CommandLine:
    """
    Execute CLI requests through the application operations.
    """

    index_builder: IndexBuilder
    """
    The committed source index builder.
    """

    index_writer: IndexWriter
    """
    The atomic index publication boundary.
    """

    navigation: Navigation
    """
    The bounded source query operations.
    """

    freshness: FreshnessCheck
    """
    The source revision compatibility check.
    """

    def execute(
        self, arguments: CommandArguments
    ) -> IndexManifest | QueryPage | FreshnessReport | RefreshReport:
        """
        Execute one request after enforcing its committed revision.
        """
        if arguments.source_scope != self.index_builder.source_reader.source_scope:
            raise SourceScopeMismatch(
                self.index_builder.source_reader.source_scope, arguments.source_scope
            )
        if arguments.command == Command.INDEX:
            index = self.index_builder.build(arguments.revision)
            freshness = FreshnessCheck(
                IndexContents(index), self.index_builder.source_reader
            )
            freshness.check(arguments.revision).require_current()
            self.index_writer.replace(index)
            freshness.check(arguments.revision).require_current()
            return index.manifest
        if arguments.command in (Command.REFRESH, Command.RECONCILE):
            return IndexRefresh(
                self.index_builder, self.freshness.index_reader, self.index_writer
            ).refresh(
                arguments.revision, reconcile=arguments.command == Command.RECONCILE
            )
        contents = IndexContents(
            self.freshness.index_reader.read(
                arguments.select_evidence(self.navigation.maximum_depth)
            )
        )
        report = replace(self.freshness, index_reader=contents).check(
            arguments.revision
        )
        if arguments.command == Command.CHECK_FRESHNESS:
            return report
        report.require_current()
        navigation = replace(self.navigation, index_reader=contents)
        result: QueryPage
        match arguments.command:
            case Command.FIND_SYMBOLS:
                result = navigation.find_symbols(
                    arguments.query, arguments.limit, arguments.offset
                )
            case Command.DESCRIBE_SYMBOL:
                result = navigation.describe_symbol(
                    arguments.symbol_identifier, arguments.limit, arguments.offset
                )
            case Command.FIND_REFERENCES:
                result = navigation.find_references(
                    arguments.symbol_identifier, arguments.limit, arguments.offset
                )
            case Command.FIND_RELATIONSHIPS:
                result = navigation.find_relationships(
                    arguments.symbol_identifier,
                    arguments.direction,
                    arguments.depth,
                    arguments.limit,
                    arguments.offset,
                )
            case _:
                raise InvalidQuery("Unsupported source navigation request.")
        if self.index_builder.source_reader.source_scope == SourceScope.WORKING_TREE:
            replace(self.freshness, index_reader=contents).check(
                arguments.revision
            ).require_current()
        return result
