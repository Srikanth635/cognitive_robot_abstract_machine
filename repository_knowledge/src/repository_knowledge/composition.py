"""
Construct source, storage and interface adapters for the local CLI.
"""

from __future__ import annotations

import sys

from repository_knowledge.adapters.git_source import GitSource
from repository_knowledge.adapters.documents import DocumentSource, SourceCollection
from repository_knowledge.adapters.python_source import PythonSource
from repository_knowledge.adapters.serialization import RecordCodec
from repository_knowledge.adapters.sqlite_index import SQLiteIndex
from repository_knowledge.adapters.working_tree import WorkingTreeSource
from repository_knowledge.contracts import SnapshotReader
from repository_knowledge.interfaces.cli import (
    Command,
    CommandLine,
    CommandParser,
    LayoutPath,
)
from repository_knowledge.interfaces.context import ContextApplication
from repository_knowledge.adapters.pytest_source import PytestSource
from repository_knowledge.model import FreshnessReport, SourceScope
from repository_knowledge.operations.freshness import FreshnessCheck
from repository_knowledge.operations.indexing import IndexBuilder
from repository_knowledge.operations.queries import Navigation
from repository_knowledge.adapters.okf import OkfDirectory, OkfCodec
from repository_knowledge.concepts import (
    BundleValidation,
    EvidenceState,
    KnowledgeBundle,
)
from repository_knowledge.exceptions import InvalidQuery
from repository_knowledge.interfaces.wiki import (
    WikiApplication,
    WikiCommand,
    WikiParser,
)
from repository_knowledge.model import IndexSelection, SelectionKind
from repository_knowledge.operations.queries import IndexContents

# %% local application composition


def main() -> int:
    """
    Execute one source-navigation request using the configured adapters.
    """
    arguments = CommandParser().parse(sys.argv[1:])
    source_reader: SnapshotReader = GitSource(arguments.repository_root)
    if arguments.source_scope == SourceScope.WORKING_TREE:
        source_reader = WorkingTreeSource(
            arguments.repository_root, arguments.included_untracked_paths
        )
    storage = SQLiteIndex(arguments.database_path)
    if arguments.command in (
        Command.FIND_EXAMPLES,
        Command.FIND_TESTS,
        Command.PREPARE_CONTEXT,
        Command.ASSESS_CHANGE,
    ):
        bundle = KnowledgeBundle([])
        if arguments.command == Command.PREPARE_CONTEXT:
            bundle_path = (
                arguments.bundle_path
                or arguments.repository_root / LayoutPath.PACKAGE / "wiki"
            )
            if arguments.bundle_path is not None or bundle_path.is_dir():
                bundle = OkfDirectory(bundle_path).read_bundle()
        baseline_reader = (
            SQLiteIndex(arguments.baseline_database.resolve())
            if arguments.baseline_database is not None
            else None
        )
        result = ContextApplication(
            storage,
            source_reader,
            PytestSource(),
            baseline_reader,
            GitSource(arguments.repository_root),
        ).execute(arguments, bundle)
        print(RecordCodec().encode(result))
        return 0
    application = CommandLine(
        IndexBuilder(
            source_reader, SourceCollection([PythonSource(), DocumentSource()])
        ),
        storage,
        Navigation(storage),
        FreshnessCheck(storage, source_reader),
    )
    result = application.execute(arguments)
    print(RecordCodec().encode(result))
    if isinstance(result, FreshnessReport) and (
        not result.matches_requested_revision or not result.matches_requested_source
    ):
        return 2
    return 0


def wiki_main() -> int:
    """
    Inspect a local OKF proposal against an explicitly selected source index.
    """
    arguments = WikiParser().parse(sys.argv[1:])
    if (
        arguments.source_scope == SourceScope.COMMITTED
        and arguments.included_untracked_paths
    ):
        raise InvalidQuery("Untracked source inclusion requires working-tree scope.")
    bundle = OkfDirectory(arguments.bundle_path).read_bundle()
    if arguments.command == WikiCommand.EXPORT:
        if arguments.destination is None:
            raise InvalidQuery("Wiki export requires a new destination directory.")
        OkfDirectory(arguments.destination).write_bundle(bundle)
        print(RecordCodec().encode(bundle))
        return 0
    source_reader: SnapshotReader = GitSource(arguments.repository_root)
    if arguments.source_scope == SourceScope.WORKING_TREE:
        source_reader = WorkingTreeSource(
            arguments.repository_root, arguments.included_untracked_paths
        )
    index = SQLiteIndex(arguments.database_path).read(
        IndexSelection(SelectionKind.KNOWLEDGE)
    )
    freshness = FreshnessCheck(IndexContents(index), source_reader)
    freshness.check(arguments.revision).require_current()
    if arguments.command == WikiCommand.MOVE:
        bundle = KnowledgeBundle(
            [
                (
                    OkfCodec().prepare_move(concept, arguments.replacement)
                    if concept.relative_path == arguments.query
                    else concept
                )
                for concept in bundle.concepts
            ],
            bundle.navigation,
            bundle.navigation_issues,
        )
    result = WikiApplication().execute(arguments, bundle, index)
    if arguments.source_scope == SourceScope.WORKING_TREE:
        freshness.check(arguments.revision).require_current()
    if arguments.command == WikiCommand.MOVE:
        if arguments.destination is None:
            raise InvalidQuery("A move proposal requires a new destination directory.")
        OkfDirectory(arguments.destination).write_bundle(result)
        print(RecordCodec().encode(result))
        return 0
    print(RecordCodec().encode(result))
    validation = result if isinstance(result, BundleValidation) else result.validation
    if validation.navigation_issues or any(
        item.format_issues
        or item.policy_issues
        or item.evidence_state != EvidenceState.CURRENT
        for item in validation.concepts
    ):
        return 2
    return 0
