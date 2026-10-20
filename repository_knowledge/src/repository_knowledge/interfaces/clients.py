"""
Operator-selected client configuration, separate from agent query inputs.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, replace
from pathlib import Path

from repository_knowledge.exceptions import InvalidQuery
from repository_knowledge.interfaces.cli import (
    Command,
    CommandArguments,
    CommandParser,
    Option,
)
from repository_knowledge.interfaces.publishing import (
    PublicationField,
    PublicationOption,
)
from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import Any

# %% separately configured publication authority


@dataclass(frozen=True)
class CanonicalSelection:
    """
    An artifact and the externally pinned checker that may authenticate it.
    """

    trusted_repository: Path
    """
    The clean checkout supplying the trusted consumer and policy.
    """

    checker_revision: str
    """
    The full immutable trusted checker commit.
    """

    artifact_identifier: int
    """
    The immutable provider artifact selected by the operator.
    """

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> CanonicalSelection | None:
        """
        Require a complete authority selection whenever any pin is supplied.
        """
        trusted = payload[PublicationField.TRUSTED_REPOSITORY]
        checker = payload[PublicationField.CHECKER_REVISION]
        artifact = payload[PublicationField.ARTIFACT]
        if trusted is None and checker is None and artifact is None:
            return None
        if (
            trusted is None
            or checker is None
            or artifact is None
            or artifact < 1
            or re.fullmatch(r"[0-9a-f]{40}", checker) is None
        ):
            raise InvalidQuery(
                "Canonical context requires a trusted checkout, full checker commit and positive artifact ID."
            )
        return cls(trusted.resolve(), checker, artifact)


@dataclass(frozen=True)
class ClientConfiguration:
    """
    The ordinary source selectors and optional independent authority.
    """

    arguments: CommandArguments
    """
    The operator-selected source, cache and baseline.
    """

    canonical: CanonicalSelection | None
    """
    The authenticated artifact configuration, when requested.
    """


@dataclass
class ClientParser:
    """
    Parse client configuration without adding authority to tool inputs.
    """

    def parse(self, arguments: list[str], server: bool = False) -> ClientConfiguration:
        """
        Separate authority flags from the existing command request contract.
        """
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument(PublicationOption.TRUSTED_REPOSITORY, type=Path)
        parser.add_argument(PublicationOption.CHECKER_REVISION)
        parser.add_argument(PublicationOption.ARTIFACT, type=int)
        if server:
            parser.add_argument(Option.AGAINST_DATABASE, type=Path)
            parser.add_argument(Option.AGAINST_REVISION, default="")
        authority, remaining = parser.parse_known_args(arguments)
        source = CommandParser().parse(
            [*remaining, Command.CHECK_FRESHNESS] if server else remaining
        )
        if server:
            source = replace(
                source,
                baseline_database=authority.against_database,
                baseline_revision=authority.against_revision,
            )
        return ClientConfiguration(
            source, CanonicalSelection.from_json(vars(authority))
        )
