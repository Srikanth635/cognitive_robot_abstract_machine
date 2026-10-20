"""
Consume canonical context using the externally pinned checker process.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from repository_knowledge.adapters.exceptions import InvalidIdentityEvidence
from repository_knowledge.adapters.serialization import RecordCodec
from repository_knowledge.contracts import CanonicalReader
from repository_knowledge.publication import ConsumptionResult, ConsumptionState

# %% executable custody across client versions


class ConsumerArgument(StrEnum):
    """
    The pinned consumer's existing executable contract.
    """

    MODULE_FLAG = "-m"
    MODULE = "repository_knowledge"
    PUBLICATION = "publication"
    REPOSITORY = "--repository"
    TRUSTED = "--trusted-repository"
    CHECKER = "--checker-revision"
    CONSUME = "consume"
    REVISION = "--revision"
    ARTIFACT = "--artifact-id"


class ConsumerEnvironment(StrEnum):
    """
    Python and Git settings controlling checker source imports.
    """

    IMPORT_PATH = "PYTHONPATH"
    NO_BYTECODE = "PYTHONDONTWRITEBYTECODE"
    NO_USER_SITE = "PYTHONNOUSERSITE"
    GIT_LOCKS = "GIT_OPTIONAL_LOCKS"


@dataclass
class PublicationProcess(CanonicalReader):
    """
    Authenticate an artifact using independently selected code and policy.
    """

    repository_root: Path
    """
    The source checkout associated with client queries.
    """

    trusted_repository: Path
    """
    The clean pinned checker checkout.
    """

    checker_revision: str
    """
    The immutable checker and policy identity.
    """

    artifact_identifier: int
    """
    The provider's immutable artifact number.
    """

    interpreter: Path
    """
    The configured CRAM interpreter running the checker.
    """

    timeout_seconds: float = 60.0
    """
    The network and verification deadline for canonical context.
    """

    def read(self, revision: str) -> ConsumptionResult:
        if self.artifact_identifier < 1 or any(
            re.fullmatch(r"[0-9a-f]{40}", value) is None
            for value in (revision, self.checker_revision)
        ):
            raise InvalidIdentityEvidence(
                "Canonical consumption requires immutable identities."
            )
        environment = os.environ.copy()
        environment[ConsumerEnvironment.IMPORT_PATH] = os.pathsep.join(
            str(self.trusted_repository / directory)
            for directory in ("repository_knowledge/src", "krrood/src")
        )
        environment[ConsumerEnvironment.NO_BYTECODE] = "1"
        environment[ConsumerEnvironment.NO_USER_SITE] = "1"
        environment[ConsumerEnvironment.GIT_LOCKS] = "0"
        arguments = [
            str(self.interpreter),
            ConsumerArgument.MODULE_FLAG,
            ConsumerArgument.MODULE,
            ConsumerArgument.PUBLICATION,
            ConsumerArgument.REPOSITORY,
            str(self.repository_root),
            ConsumerArgument.TRUSTED,
            str(self.trusted_repository),
            ConsumerArgument.CHECKER,
            self.checker_revision,
            ConsumerArgument.CONSUME,
            ConsumerArgument.REVISION,
            revision,
            ConsumerArgument.ARTIFACT,
            str(self.artifact_identifier),
        ]
        result = subprocess.run(
            arguments,
            cwd=self.trusted_repository,
            env=environment,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            stdin=subprocess.DEVNULL,
        )
        if result.returncode not in (0, 2):
            raise InvalidIdentityEvidence(
                "The trusted canonical consumer failed; retain source navigation."
            )
        consumed = RecordCodec().decode(result.stdout, ConsumptionResult)
        if (consumed.state == ConsumptionState.ACCEPTED) != (result.returncode == 0):
            raise InvalidIdentityEvidence(
                "The canonical consumer verdict disagrees with its process status."
            )
        return consumed
