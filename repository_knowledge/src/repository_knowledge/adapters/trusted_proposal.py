"""
Evaluate proposal Git objects using a clean, independently pinned checker.
"""

from __future__ import annotations

import os
import subprocess
import sys
import krrood
from dataclasses import dataclass, replace
from pathlib import Path
from typing_extensions import TYPE_CHECKING

from repository_knowledge.adapters.committed_knowledge import CommittedKnowledge
from repository_knowledge.adapters.digests import CanonicalDigest
from repository_knowledge.adapters.documents import DocumentSource, SourceCollection
from repository_knowledge.adapters.exceptions import InvalidIdentityEvidence
from repository_knowledge.adapters.git_source import GitCheckout, GitSource, TreeEntry
from repository_knowledge.adapters.python_source import PythonSource
from repository_knowledge.adapters.serialization import RecordCodec
from repository_knowledge.concepts import KnowledgeBundle
from repository_knowledge.model import IndexData
from repository_knowledge.operations.admission import ProposalAdmission
from repository_knowledge.operations.indexing import IndexBuilder
from repository_knowledge.publication import AdmissionResult, ExtractionFingerprint

if TYPE_CHECKING:
    from repository_knowledge.publication import PublicationPolicy

# %% immutable checker custody


@dataclass
class TrustedChecker:
    """
    Verify the source checkout supplying executable checker code and policy.
    """

    repository_root: Path
    """The independently selected checker checkout."""
    revision: str
    """
    The trusted immutable checker revision.
    """

    def require_current(self) -> None:
        """
        Refuse a changed checkout or code loaded from another repository path.
        """
        actual_root = Path(__file__).resolve().parents[4]
        checkout = GitCheckout(self.repository_root)
        current = checkout.run("rev-parse", "HEAD").decode("ascii").strip()
        changed = checkout.run("status", "--porcelain=v1", "--untracked-files=all")
        dependency = (
            Path(krrood.__file__).resolve() if krrood.__file__ is not None else None
        )
        if (
            self.repository_root.resolve() != actual_root
            or current != self.revision
            or changed.strip()
            or dependency is None
            or not dependency.is_relative_to(
                self.repository_root.resolve() / "krrood/src"
            )
        ):
            raise InvalidIdentityEvidence(
                "checker checkout differs from its immutable trust root"
            )

    def governance_changes(
        self, proposal_root: Path, proposal_revision: str, policy: PublicationPolicy
    ) -> list[str]:
        """
        Compare protected file modes and object identities against trusted code.
        """
        trusted = self.protected_entries(self.repository_root, self.revision, policy)
        proposed = self.protected_entries(proposal_root, proposal_revision, policy)
        return sorted(
            path
            for path in trusted.keys() | proposed.keys()
            if trusted.get(path) != proposed.get(path)
        )

    @staticmethod
    def protected_entries(
        root: Path, revision: str, policy: PublicationPolicy
    ) -> dict[str, TreeEntry]:
        """
        Read protected tree metadata without opening or executing proposal contents.
        """
        listing = GitCheckout(root).run("ls-tree", "-r", "-z", "--full-tree", revision)
        entries = [
            TreeEntry.from_listing(record) for record in listing.split(b"\0") if record
        ]
        return {
            entry.relative_path: entry
            for entry in entries
            if any(
                entry.relative_path.startswith(prefix)
                for prefix in policy.protected_paths
            )
        }


# %% reproducible source admission


@dataclass
class CheckedProposal:
    """
    A captured source projection and its bounded admission result.
    """

    index: IndexData
    """
    The coherent source facts available to the publisher.
    """

    result: AdmissionResult
    """
    The complete deterministic admission outcome.
    """


@dataclass
class TrustedProposal:
    """
    Build proposal evidence without importing its code or trusting its cache.
    """

    proposal_root: Path
    """
    The repository holding untrusted proposal objects.
    """

    checker: TrustedChecker
    """
    The clean, independently pinned checker boundary.
    """
    policy: PublicationPolicy
    """
    The trusted policy, never loaded from proposal code.
    """

    verification_hash_seed: str = "1"
    """
    The explicit hash seed of the independent verification interpreter.
    """

    def verify(self, revision: str, base_revision: str) -> CheckedProposal:
        """
        Compare clean extractions in separate interpreters before source admission.
        """
        self.checker.require_current()
        builder = IndexBuilder(
            GitSource(self.proposal_root),
            SourceCollection([PythonSource(), DocumentSource()]),
        )
        complete = builder.build(revision)
        digester = CanonicalDigest()
        first = digester.fingerprint(complete)
        index = replace(complete, references=[], analyses=[])
        del complete
        second = self.independent_fingerprint(revision)
        previous_snapshot = GitSource(self.proposal_root).read_snapshot(base_revision)
        previous = IndexData(
            previous_snapshot.manifest,
            previous_snapshot.sources,
            [],
            [],
            previous_snapshot.diagnostics,
        )
        reader = CommittedKnowledge()
        before = (
            reader.read_bundle(previous, self.policy.bundle_directory)
            if any(
                source.relative_path.startswith(self.policy.bundle_directory + "/")
                for source in previous.sources
            )
            else KnowledgeBundle([])
        )
        proposed = reader.read_bundle(index, self.policy.bundle_directory)
        governance = self.checker.governance_changes(
            self.proposal_root, revision, self.policy
        )
        first_digest = digester.digest(first)
        report = ProposalAdmission(digester).check(
            before, proposed, index, governance, first_digest, digester.digest(second)
        )
        result = AdmissionResult(
            index.manifest,
            digester.digest(self.policy),
            first_digest,
            governance,
            report,
        )
        return CheckedProposal(index, result)

    def independent_fingerprint(self, revision: str) -> ExtractionFingerprint:
        """
        Read a complete fingerprint from a fresh trusted interpreter.
        """
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = self.verification_hash_seed
        command = [
            sys.executable,
            "-m",
            "repository_knowledge",
            "publication",
            "--repository",
            str(self.proposal_root),
            "--trusted-repository",
            str(self.checker.repository_root),
            "--checker-revision",
            self.checker.revision,
            "fingerprint",
            "--revision",
            revision,
        ]
        result = subprocess.run(
            command,
            cwd=self.checker.repository_root,
            env=environment,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise InvalidIdentityEvidence("independent trusted extraction failed")
        return RecordCodec().decode(result.stdout, ExtractionFingerprint)
