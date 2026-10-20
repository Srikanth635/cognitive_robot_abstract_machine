"""
Parse explicit immutable source, checker and artifact identities.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing_extensions import TYPE_CHECKING

from repository_knowledge.exceptions import InvalidQuery
from repository_knowledge.interfaces.cli import Option

if TYPE_CHECKING:
    from typing_extensions import Any

# %% publication command vocabulary


class PublicationCommand(StrEnum):
    """
    The distinct trust operations exposed to repository operators.
    """

    CHECK = "check"
    PUBLISH = "publish"
    CONSUME = "consume"
    FINGERPRINT = "fingerprint"


class PublicationEntry(StrEnum):
    """
    The module subcommand selecting publication operations.
    """

    PUBLICATION = "publication"


class PublicationOption(StrEnum):
    """
    Explicit authority and artifact options for the trust interface.
    """

    TRUSTED_REPOSITORY = "--trusted-repository"
    CHECKER_REVISION = "--checker-revision"
    POLICY = "--policy"
    BASE_REVISION = "--base-revision"
    REQUEST = "--request"
    OUTPUT = "--output"
    ARTIFACT = "--artifact-id"


class PublicationField(StrEnum):
    """
    The external argument-parser request shape.
    """

    COMMAND = "command"
    REPOSITORY = "repository"
    TRUSTED_REPOSITORY = "trusted_repository"
    CHECKER_REVISION = "checker_revision"
    POLICY = "policy"
    REVISION = "revision"
    BASE_REVISION = "base_revision"
    REQUEST = "request"
    OUTPUT = "output"
    ARTIFACT = "artifact_id"


@dataclass(frozen=True)
class PublicationArguments:
    """
    A source request with a separately selected immutable trust root.
    """

    command: PublicationCommand
    """
    The requested trust operation.
    """

    repository_root: Path
    """
    The repository containing proposal/source Git objects.
    """

    trusted_repository: Path
    """
    The independently selected checker checkout.
    """

    checker_revision: str
    """
    The externally pinned immutable checker revision.
    """

    policy_path: Path
    """
    The trusted policy document.
    """

    revision: str
    """
    The exact proposed, published or consumed source commit.
    """

    base_revision: str
    """
    The exact committed proposal baseline, when checking admission.
    """

    request_number: int
    """
    The merged pull request being independently verified.
    """

    artifact_identifier: int
    """
    The immutable artifact selected for consumption or historical restoration.
    """

    output_directory: Path | None
    """
    A new directory receiving the canonical payload.
    """

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> PublicationArguments:
        """
        Validate external identities before they reach Git or the identity provider.
        """
        trusted = data[PublicationField.TRUSTED_REPOSITORY].resolve()
        command = PublicationCommand(data[PublicationField.COMMAND])
        revisions = [
            data[PublicationField.CHECKER_REVISION],
            data[PublicationField.REVISION],
        ]
        if command == PublicationCommand.CHECK:
            revisions.append(data[PublicationField.BASE_REVISION])
        if any(
            re.fullmatch(r"[0-9a-f]{40}", revision) is None for revision in revisions
        ):
            raise InvalidQuery(
                "Publication identities must be full immutable Git commit identifiers."
            )
        if (
            command == PublicationCommand.PUBLISH
            and data[PublicationField.REQUEST] < 0
            or command == PublicationCommand.CONSUME
            and data[PublicationField.ARTIFACT] < 1
        ):
            raise InvalidQuery(
                "Request numbers cannot be negative; artifact identities must be positive."
            )
        return cls(
            command,
            data[PublicationField.REPOSITORY].resolve(),
            trusted,
            data[PublicationField.CHECKER_REVISION],
            (
                data[PublicationField.POLICY]
                or trusted / "repository_knowledge/policy/publication.toml"
            ).resolve(),
            data[PublicationField.REVISION],
            data[PublicationField.BASE_REVISION],
            data[PublicationField.REQUEST],
            data[PublicationField.ARTIFACT],
            data[PublicationField.OUTPUT],
        )


@dataclass
class PublicationParser:
    """
    Keep authority configuration separate from mutable proposal inputs.
    """

    def parse(self, arguments: list[str]) -> PublicationArguments:
        """
        Read a publication request without loading network or repository adapters.
        """
        parser = argparse.ArgumentParser(prog="cram-publication")
        parser.add_argument(Option.REPOSITORY, type=Path, required=True)
        parser.add_argument(
            PublicationOption.TRUSTED_REPOSITORY, type=Path, required=True
        )
        parser.add_argument(PublicationOption.CHECKER_REVISION, required=True)
        parser.add_argument(PublicationOption.POLICY, type=Path)
        parser.set_defaults(base_revision="", request=0, artifact_id=0, output=None)
        commands = parser.add_subparsers(dest=PublicationField.COMMAND, required=True)
        for command in PublicationCommand:
            subparser = commands.add_parser(command)
            subparser.add_argument(Option.REVISION, required=True)
            if command == PublicationCommand.CHECK:
                subparser.add_argument(PublicationOption.BASE_REVISION, required=True)
            elif command == PublicationCommand.PUBLISH:
                subparser.add_argument(PublicationOption.REQUEST, type=int, default=0)
                subparser.add_argument(
                    PublicationOption.OUTPUT, type=Path, required=True
                )
            elif command == PublicationCommand.CONSUME:
                subparser.add_argument(
                    PublicationOption.ARTIFACT, type=int, required=True
                )
        return PublicationArguments.from_json(vars(parser.parse_args(arguments)))
