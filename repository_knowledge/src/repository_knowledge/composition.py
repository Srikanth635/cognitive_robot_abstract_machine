"""
Construct source, storage and interface adapters for the local CLI.
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
from enum import StrEnum
from dataclasses import dataclass, replace

from repository_knowledge.interfaces.agents import RequestApplication, ApplicationResult
from repository_knowledge.assistance import TaskProfile
from repository_knowledge.adapters.plan_sources import CoraplexPlanningSources
from repository_knowledge.interfaces.clients import ClientConfiguration, ClientParser
from repository_knowledge.adapters.canonical_process import PublicationProcess

from repository_knowledge.adapters.git_source import GitSource, GitCheckout, GitArgument
from repository_knowledge.adapters.committed_knowledge import CommittedKnowledge
from repository_knowledge.adapters.digests import CanonicalDigest
from repository_knowledge.adapters.github_artifacts import (
    GitHubArtifacts,
    PublicationFilename,
)
from repository_knowledge.adapters.github_identity import GitHubTransport, GitHubReviews
from repository_knowledge.adapters.publication_policy import PolicyDocument
from repository_knowledge.adapters.trusted_proposal import (
    TrustedChecker,
    TrustedProposal,
)
from repository_knowledge.adapters.exceptions import InvalidIdentityEvidence
from repository_knowledge.interfaces.publishing import (
    PublicationParser,
    PublicationCommand,
)
from repository_knowledge.operations.publication import (
    KnowledgePublication,
    PublicationConsumer,
)
from repository_knowledge.operations.exceptions import PublicationDenied
from repository_knowledge.publication import (
    ConsumptionState,
    TrustIssue,
    PublicationMode,
)
from repository_knowledge.adapters.documents import DocumentSource, SourceCollection
from repository_knowledge.adapters.python_source import PythonSource
from repository_knowledge.adapters.serialization import RecordCodec
from repository_knowledge.adapters.sqlite_index import SQLiteIndex
from repository_knowledge.adapters.working_tree import WorkingTreeSource
from repository_knowledge.contracts import SnapshotReader, CanonicalReader
from repository_knowledge.clients import ContextKnowledge
from repository_knowledge.operations.canonical_context import CanonicalContext
from repository_knowledge.interfaces.cli import (
    Command,
    CommandLine,
    CommandArguments,
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


class PublicationEnvironment(StrEnum):
    """
    The independently provisioned provider credential available to operators.
    """

    IDENTITY_TOKEN = "KNOWLEDGE_IDENTITY_TOKEN"


@dataclass
class LocalApplication(RequestApplication):
    """
    Compose local source adapters for one coherent application request.
    """

    canonical_reader: CanonicalReader | None = None
    """
    An explicitly selected authenticated publication boundary.
    """

    def execute(self, arguments: CommandArguments) -> ApplicationResult:
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
            knowledge = self._read_knowledge(arguments, source_reader)
            if knowledge.canonical_status is not None:
                arguments = replace(
                    arguments, revision=knowledge.canonical_status.source_commit
                )
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
                context_references=(
                    CoraplexPlanningSources().select()
                    if arguments.command == Command.PREPARE_CONTEXT
                    and arguments.profile == TaskProfile.PLAN_GENERATION
                    else []
                ),
            ).execute(arguments, knowledge.bundle)
            if arguments.command == Command.PREPARE_CONTEXT:
                result = replace(result, canonical_status=knowledge.canonical_status)
            return result
        application = CommandLine(
            IndexBuilder(
                source_reader, SourceCollection([PythonSource(), DocumentSource()])
            ),
            storage,
            Navigation(storage),
            FreshnessCheck(storage, source_reader),
        )
        result = application.execute(arguments)
        return result

    def _read_knowledge(
        self, arguments: CommandArguments, source_reader: SnapshotReader
    ) -> ContextKnowledge:
        """
        Select local drafts or independently admitted canonical context.
        """
        if arguments.command != Command.PREPARE_CONTEXT:
            return ContextKnowledge(KnowledgeBundle([]))
        if self.canonical_reader is not None:
            return CanonicalContext(self.canonical_reader).read(
                source_reader.resolve_revision(arguments.revision),
                arguments.source_scope,
            )
        bundle_path = (
            arguments.bundle_path
            or arguments.repository_root / LayoutPath.PACKAGE / "wiki"
        )
        if arguments.bundle_path is not None or bundle_path.is_dir():
            return ContextKnowledge(OkfDirectory(bundle_path).read_bundle())
        return ContextKnowledge(KnowledgeBundle([]))


def main() -> int:
    """
    Execute one source-navigation request using the configured adapters.
    """
    configuration = ClientParser().parse(sys.argv[1:])
    result = ClientApplication(configuration).compose().execute(configuration.arguments)
    print(RecordCodec().encode(result))
    if isinstance(result, FreshnessReport) and (
        not result.matches_requested_revision or not result.matches_requested_source
    ):
        return 2
    return 0


@dataclass
class ClientApplication:
    """
    Compose optional canonical authority independently of query arguments.
    """

    configuration: ClientConfiguration
    """
    The source and authority choices made by the operator.
    """

    def compose(self) -> LocalApplication:
        """
        Keep artifact authentication in its clean separately pinned checkout.
        """
        canonical = self.configuration.canonical
        if canonical is None:
            return LocalApplication()
        return LocalApplication(
            PublicationProcess(
                self.configuration.arguments.repository_root,
                canonical.trusted_repository,
                canonical.checker_revision,
                canonical.artifact_identifier,
                Path(sys.executable),
            )
        )


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


# %% independently trusted publication composition


def publication_main() -> int:
    """
    Evaluate source proposals or authenticate complete canonical publications.
    """
    arguments = PublicationParser().parse(sys.argv[1:])
    checker = TrustedChecker(arguments.trusted_repository, arguments.checker_revision)
    checker.require_current()
    if not arguments.policy_path.is_relative_to(arguments.trusted_repository):
        raise InvalidIdentityEvidence("policy is outside the trusted checkout")
    policy = PolicyDocument(arguments.policy_path).read()
    transport = GitHubTransport(
        os.environ.get(PublicationEnvironment.IDENTITY_TOKEN, "")
    )
    reviews = GitHubReviews(transport)
    digest = CanonicalDigest()
    codec = RecordCodec()
    if (
        arguments.command in (PublicationCommand.PUBLISH, PublicationCommand.CONSUME)
        and policy.mode == PublicationMode.INDEPENDENT_REVIEW
        and not policy.reviewers
    ):
        raise PublicationDenied([TrustIssue.MISSING_REVIEW])
    if arguments.command == PublicationCommand.FINGERPRINT:
        index = IndexBuilder(
            GitSource(arguments.repository_root),
            SourceCollection([PythonSource(), DocumentSource()]),
        ).build(arguments.revision)
        print(codec.encode(digest.fingerprint(index)))
        return 0
    if arguments.command == PublicationCommand.CONSUME:
        downloaded = GitHubArtifacts(transport).read_publication(
            policy, arguments.artifact_identifier, arguments.checker_revision
        )
        result = PublicationConsumer(
            policy, reviews, digest, arguments.checker_revision, reviews
        ).consume(downloaded.publication, downloaded.receipt, arguments.revision)
        print(codec.encode(result))
        return 0 if result.state == ConsumptionState.ACCEPTED else 2
    base_revision = arguments.base_revision
    if arguments.command == PublicationCommand.PUBLISH:
        base_revision = (
            GitCheckout(arguments.repository_root)
            .run(GitArgument.REVISION, arguments.revision + "^1")
            .decode("ascii")
            .strip()
        )
    checked = TrustedProposal(arguments.repository_root, checker, policy).verify(
        arguments.revision, base_revision
    )
    if arguments.command == PublicationCommand.CHECK:
        print(codec.encode(checked.result))
        return 2 if checked.result.report.issues else 0
    if checked.result.report.issues:
        raise PublicationDenied(checked.result.report.issues)
    publication = KnowledgePublication(
        policy, CommittedKnowledge(), reviews, digest, reviews
    ).publish(checked.index, arguments.request_number, arguments.checker_revision)
    destination = arguments.output_directory
    if destination is None or destination.exists():
        raise InvalidIdentityEvidence("publication output must be a new directory")
    destination.mkdir(parents=True)
    (destination / PublicationFilename.PAYLOAD).write_text(
        codec.encode(publication), encoding="utf-8", newline="\n"
    )
    print(codec.encode(publication.binding))
    return 0
