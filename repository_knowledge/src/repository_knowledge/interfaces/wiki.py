"""
Expose curated knowledge through bounded, source-aware requests.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from typing_extensions import Any

from repository_knowledge.concepts import BundleValidation, ConceptPage, KnowledgeBundle
from repository_knowledge.interfaces.cli import Option, LayoutPath, RevisionSelector
from repository_knowledge.model import IndexData, SourceScope
from repository_knowledge.operations.knowledge import (
    KnowledgeQueries,
    KnowledgeValidation,
    ConceptLifecycle,
)

# %% wiki request contract


class WikiCommand(StrEnum):
    """
    The local curated-knowledge operations available to agents.
    """

    VALIDATE = "validate"
    FIND = "find"
    SHOW = "show"
    EXPORT = "export"
    MOVE = "move"


class WikiField(StrEnum):
    """
    Keys supplied by the wiki argument parser.
    """

    COMMAND = "command"
    REPOSITORY = "repository"
    DATABASE = "database"
    REVISION = "revision"
    SCOPE = "scope"
    INCLUDE_UNTRACKED = "include_untracked"
    BUNDLE = "bundle"
    QUERY = "query"
    LIMIT = "limit"
    OFFSET = "offset"
    DESTINATION = "destination"
    REPLACEMENT = "replacement"


class WikiOption(StrEnum):
    """
    Additional flags and paths belonging to curated bundles.
    """

    BUNDLE = "--bundle"
    DIRECTORY = "wiki"


@dataclass(frozen=True)
class WikiArguments:
    """
    A curated-knowledge request with an explicit source scope.
    """

    command: WikiCommand
    """
    The requested wiki operation.
    """

    repository_root: Path
    """
    The source repository selected for evidence checks.
    """

    bundle_path: Path
    """
    The local curated bundle being inspected.
    """

    database_path: Path
    """
    The completed source index used for validation.
    """

    revision: str
    """
    The requested source revision.
    """

    source_scope: SourceScope
    """
    Whether evidence includes current working files.
    """

    included_untracked_paths: tuple[str, ...]
    """
    The working-file inclusion policy used by the saved index.
    """

    query: str
    """
    The discovery text or exact concept path.
    """

    limit: int
    """
    The maximum returned concept count.
    """

    offset: int
    """
    The first match returned in this page.
    """

    destination: Path | None
    """
    The new directory selected for an explicit export.
    """

    replacement: str = ""
    """
    The new concept identity proposed by a relocation.
    """

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> WikiArguments:
        """
        Normalize the external parser shape into a typed request.
        """
        root = data[WikiField.REPOSITORY].resolve()
        scope = SourceScope(data[WikiField.SCOPE])
        filename = (
            LayoutPath.DATABASE
            if scope == SourceScope.COMMITTED
            else LayoutPath.OVERLAY_DATABASE
        )
        return cls(
            WikiCommand(data[WikiField.COMMAND]),
            root,
            (
                data[WikiField.BUNDLE]
                or root / LayoutPath.PACKAGE / WikiOption.DIRECTORY
            ).resolve(),
            (
                data[WikiField.DATABASE]
                or root / LayoutPath.PACKAGE / LayoutPath.CACHE / filename
            ).resolve(),
            data[WikiField.REVISION],
            scope,
            tuple(data[WikiField.INCLUDE_UNTRACKED] or ()),
            data[WikiField.QUERY],
            data[WikiField.LIMIT],
            data[WikiField.OFFSET],
            data[WikiField.DESTINATION],
            data[WikiField.REPLACEMENT],
        )


@dataclass
class WikiParser:
    """
    Parse local wiki commands without initializing source adapters.
    """

    default_limit: int = 20
    """
    The initial number of concepts requested per query page.
    """

    def parse(self, arguments: list[str]) -> WikiArguments:
        """
        Read a request with explicit repository and optional bundle boundaries.
        """
        parser = argparse.ArgumentParser(prog="cram-wiki")
        parser.add_argument(Option.REPOSITORY, type=Path, required=True)
        parser.add_argument(Option.DATABASE, type=Path)
        parser.add_argument(Option.REVISION, default=RevisionSelector.CHECKOUT)
        parser.add_argument(
            Option.SCOPE, choices=list(SourceScope), default=SourceScope.COMMITTED
        )
        parser.add_argument(Option.INCLUDE_UNTRACKED, action="append")
        parser.add_argument(WikiOption.BUNDLE, type=Path)
        parser.set_defaults(
            query="",
            limit=self.default_limit,
            offset=0,
            destination=None,
            replacement="",
        )
        commands = parser.add_subparsers(dest=WikiField.COMMAND, required=True)
        for command in WikiCommand:
            subparser = commands.add_parser(command)
            if command in (WikiCommand.FIND, WikiCommand.SHOW):
                subparser.add_argument(WikiField.QUERY)
                subparser.add_argument(
                    Option.LIMIT, type=int, default=self.default_limit
                )
                subparser.add_argument(Option.OFFSET, type=int, default=0)
            if command == WikiCommand.MOVE:
                subparser.add_argument(WikiField.QUERY)
                subparser.add_argument(WikiField.REPLACEMENT)
            if command in (WikiCommand.EXPORT, WikiCommand.MOVE):
                subparser.add_argument(WikiField.DESTINATION, type=Path)
        return WikiArguments.from_json(vars(parser.parse_args(arguments)))


# %% wiki operations


@dataclass
class WikiApplication:
    """
    Dispatch source-bound wiki requests without accessing storage directly.
    """

    def execute(
        self, arguments: WikiArguments, bundle: KnowledgeBundle, index: IndexData
    ) -> BundleValidation | ConceptPage | KnowledgeBundle:
        """
        Validate or discover curated content from a captured source graph.
        """
        if arguments.command == WikiCommand.EXPORT:
            return bundle
        if arguments.command == WikiCommand.MOVE:
            return ConceptLifecycle().relocate(
                bundle, arguments.query, arguments.replacement
            )
        if arguments.command == WikiCommand.VALIDATE:
            return KnowledgeValidation().validate(bundle, index)
        if arguments.command == WikiCommand.SHOW:
            bundle_paths = {concept.relative_path for concept in bundle.concepts}
            if arguments.query not in bundle_paths:
                return KnowledgeQueries().find(
                    KnowledgeBundle([], bundle.navigation),
                    index,
                    arguments.query,
                    arguments.limit,
                    arguments.offset,
                )
        return KnowledgeQueries().find(
            bundle, index, arguments.query, arguments.limit, arguments.offset
        )
