"""Read exact committed blobs and derive workspace import roots."""

from __future__ import annotations

import hashlib
import io
import platform
import subprocess
import tokenize
import tomllib
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path, PurePosixPath

from typing_extensions import TYPE_CHECKING

from repository_knowledge.contracts import SnapshotReader
from repository_knowledge.exceptions import GitOperationFailed
from repository_knowledge.model import (
    Diagnostic,
    DiagnosticKind,
    IndexManifest,
    IndexVersion,
    RepositorySnapshot,
    SourceKind,
    SourceRecord,
    SourceScope,
)

if TYPE_CHECKING:
    from typing_extensions import Any

# %% Git and manifest vocabulary


class GitArgument(StrEnum):
    """Git commands and options required by committed-source reads."""

    PROGRAM = "git"
    DIRECTORY = "-C"
    REVISION = "rev-parse"
    VERIFY = "--verify"
    END_OPTIONS = "--end-of-options"
    TREE = "ls-tree"
    RECURSIVE = "-r"
    NULL_DELIMITED = "-z"
    OBJECTS = "cat-file"
    BATCH = "--batch"
    HISTORY = "rev-list"
    HISTORY_ROOTS = "--max-parents=0"
    FILES = "ls-files"
    STAGE = "--stage"
    OTHERS = "--others"
    EXCLUDE_STANDARD = "--exclude-standard"
    GIT_DIRECTORY = "--absolute-git-dir"
    CHECKOUT = "HEAD"


class ProjectField(StrEnum):
    """External project metadata keys interpreted by source-root discovery."""

    PROJECT = "project"
    NAME = "name"
    TOOL = "tool"
    SETUPTOOLS = "setuptools"
    PACKAGE_DIRECTORY = "package-dir"
    PACKAGES = "packages"
    FIND = "find"
    WHERE = "where"


class SourceSuffix(StrEnum):
    """Readable source formats in the initial inventory."""

    PYTHON = ".py"
    MARKDOWN = ".md"
    RESTRUCTURED_TEXT = ".rst"
    TOML = ".toml"
    YAML = ".yaml"
    SHORT_YAML = ".yml"
    JSON = ".json"
    XML = ".xml"
    SQL = ".sql"
    JINJA = ".jinja"
    JINJA_SHORT = ".j2"


class ExcludedDirectory(StrEnum):
    """Directories whose local or third-party state is outside source indexing."""

    ENVIRONMENT = ".venv"
    CACHE = ".cache"
    PYTHON_CACHE = "__pycache__"
    NODE_DEPENDENCIES = "node_modules"
    VENDORED = "vendor"
    BUILD = "build"
    DISTRIBUTION = "dist"


class SourceFilename(StrEnum):
    """Filenames with defined inventory semantics."""

    PROJECT = "pyproject.toml"
    GENERATED_MAPPING = "ormatic_interface.py"
    PACKAGE = "__init__.py"
    SOURCE_DIRECTORY = "src"


class GitMode(StrEnum):
    """Git tree modes that cannot supply ordinary source files."""

    SYMBOLIC_LINK = "120000"
    SUBMODULE = "160000"
    DIRECTORY = "040000"
    REGULAR = "100644"


@dataclass(frozen=True)
class TreeEntry:
    """An object and path supplied by a committed Git tree."""

    relative_path: str
    """The tracked POSIX path."""

    object_identifier: str
    """The committed blob or referenced Git object."""

    file_mode: str
    """The Git file mode controlling how contents may be interpreted."""

    @classmethod
    def from_listing(cls, listing: bytes) -> TreeEntry:
        """Parse one null-delimited Git tree record."""
        metadata, relative_path = listing.split(b"\t", 1)
        file_mode, object_type, object_identifier = metadata.decode("ascii").split(" ")
        return cls(
            relative_path.decode("utf-8", errors="surrogateescape"),
            object_identifier,
            file_mode,
        )


@dataclass(frozen=True)
class SourceRoot:
    """A source directory's owning distribution and import prefix."""

    relative_directory: str
    """The repository-relative directory containing importable modules."""

    distribution_name: str
    """The distribution declaring this source root."""

    module_prefix: str = ""
    """A package prefix explicitly mapped to the directory."""


@dataclass(frozen=True)
class ModuleIdentity:
    """The import identity and project owning a Python source file."""

    module_name: str
    """The source's canonical import name."""

    distribution_name: str
    """The project declaring the matching source root."""

    is_package: bool
    """Whether the source declares a package initializer."""


@dataclass(frozen=True)
class ProjectMetadata:
    """The project fields needed to determine Python source roots."""

    distribution_name: str
    """The declared project distribution."""

    source_directories: list[SourceRoot]
    """The project's explicitly configured source directories."""

    @classmethod
    def from_json(cls, data: dict[str, Any], project_directory: str) -> ProjectMetadata:
        """Read setuptools source-root declarations from project metadata."""
        project = data.get(ProjectField.PROJECT, {})
        distribution_name = project.get(ProjectField.NAME, project_directory)
        configuration = data.get(ProjectField.TOOL, {}).get(ProjectField.SETUPTOOLS, {})
        roots = [
            SourceRoot(
                str(PurePosixPath(project_directory, directory)),
                distribution_name,
                prefix,
            )
            for prefix, directory in configuration.get(
                ProjectField.PACKAGE_DIRECTORY, {}
            ).items()
        ]
        packages = configuration.get(ProjectField.PACKAGES, {})
        if isinstance(packages, dict):
            roots.extend(
                SourceRoot(
                    str(PurePosixPath(project_directory, directory)), distribution_name
                )
                for directory in packages.get(ProjectField.FIND, {}).get(
                    ProjectField.WHERE, []
                )
            )
        return cls(distribution_name, roots)


# %% Git checkout access


@dataclass
class GitCheckout:
    """Read Git metadata and objects from a selected checkout."""

    repository_root: Path
    """The checkout against which Git commands execute."""

    def run(self, *arguments: str, input_data: bytes | None = None) -> bytes:
        """Return Git output or raise a structured source-operation failure."""
        result = subprocess.run(
            [
                GitArgument.PROGRAM,
                GitArgument.DIRECTORY,
                str(self.repository_root),
                *arguments,
            ],
            input=input_data,
            capture_output=True,
        )
        if result.returncode != 0:
            raise GitOperationFailed(
                tuple(arguments), result.stderr.decode("utf-8", errors="replace")
            )
        return result.stdout


# %% committed source inventory


@dataclass
class GitSource(SnapshotReader):
    """Read one repository's committed source independently of working files."""

    repository_root: Path
    """The checkout supplying Git objects."""

    @property
    def source_scope(self) -> SourceScope:
        return SourceScope.COMMITTED

    checkout: GitCheckout = field(init=False)
    """The Git command boundary for this checkout."""

    def __post_init__(self) -> None:
        """Bind Git reads to the selected checkout."""
        self.checkout = GitCheckout(self.repository_root)

    def resolve_revision(self, revision: str) -> str:
        return (
            self.checkout.run(
                GitArgument.REVISION,
                GitArgument.VERIFY,
                GitArgument.END_OPTIONS,
                f"{revision}^{{commit}}",
            )
            .decode("ascii")
            .strip()
        )

    def read_snapshot(self, revision: str) -> RepositorySnapshot:
        commit_identifier = self.resolve_revision(revision)
        tree_identifier = (
            self.checkout.run(GitArgument.REVISION, f"{commit_identifier}^{{tree}}")
            .decode("ascii")
            .strip()
        )
        history_roots = (
            self.checkout.run(
                GitArgument.HISTORY, GitArgument.HISTORY_ROOTS, commit_identifier
            )
            .decode("ascii")
            .splitlines()
        )
        entries = [
            TreeEntry.from_listing(listing)
            for listing in self.checkout.run(
                GitArgument.TREE,
                GitArgument.RECURSIVE,
                GitArgument.NULL_DELIMITED,
                commit_identifier,
            ).split(b"\0")
            if listing
        ]
        inventory = SourceInventory()
        readable = [
            entry
            for entry in entries
            if inventory.classify(entry).source_kind
            not in (SourceKind.EXCLUDED, SourceKind.UNSUPPORTED)
        ]
        contents = inventory.capture(entries, self._read_blobs(readable))
        records = contents.sources
        diagnostics = contents.diagnostics
        source_digest = contents.source_digest
        repository_identifier = hashlib.sha256(
            "\n".join(sorted(history_roots)).encode("ascii")
        ).hexdigest()
        snapshot_identifier = hashlib.sha256(
            "\n".join(
                (
                    repository_identifier,
                    commit_identifier,
                    tree_identifier,
                    source_digest,
                    platform.python_version(),
                    str(IndexVersion.SOURCE_ANALYSIS),
                )
            ).encode("utf-8")
        ).hexdigest()
        manifest = IndexManifest(
            repository_identifier,
            commit_identifier,
            tree_identifier,
            snapshot_identifier,
            source_digest,
            platform.python_version(),
            len(records),
        )
        return RepositorySnapshot(manifest, records, diagnostics)

    def _read_blobs(self, entries: list[TreeEntry]) -> dict[str, bytes]:
        """Read selected blobs together without opening excluded contents."""
        if not entries:
            return {}
        output = self.checkout.run(
            GitArgument.OBJECTS,
            GitArgument.BATCH,
            input_data="".join(
                f"{entry.object_identifier}\n" for entry in entries
            ).encode("ascii"),
        )
        stream = io.BytesIO(output)
        blobs: dict[str, bytes] = {}
        for entry in entries:
            object_identifier, object_type, object_size = (
                stream.readline().decode("ascii").strip().split(" ")
            )
            blobs[entry.relative_path] = stream.read(int(object_size))
            stream.read(1)
        return blobs


@dataclass(frozen=True)
class InventoryDecision:
    """The selected inventory category and any associated limitation."""

    source_kind: SourceKind
    """The supported analysis category or exclusion."""

    diagnostic: Diagnostic | None = None
    """The reason full source analysis is unavailable."""


@dataclass(frozen=True)
class InventoryEntry:
    """A committed tree entry and the policy governing its contents."""

    tree_entry: TreeEntry
    """The committed object and repository path."""

    decision: InventoryDecision
    """The source category and any exclusion evidence."""


# %% source inventory policy


@dataclass
class SourceContents:
    """Source records and exclusions captured under one inventory policy."""

    sources: list[SourceRecord]
    """The selected source inventory and supported contents."""

    diagnostics: list[Diagnostic]
    """The policy exclusions and unsupported source entries."""

    source_digest: str
    """The identity of selected paths, contents and classification."""


@dataclass
class SourceInventory:
    """Interpret file inventory consistently across committed and working source."""

    def capture(
        self, entries: list[TreeEntry], blobs: dict[str, bytes]
    ) -> SourceContents:
        """Create source records from already captured, supported contents."""
        source_roots = self.discover_roots(entries, blobs)
        records: list[SourceRecord] = []
        diagnostics: list[Diagnostic] = []
        for entry in entries:
            decision = self.classify(entry)
            contents = blobs.get(entry.relative_path, b"")
            source_text = contents.decode("utf-8-sig", errors="replace")
            if decision.source_kind == SourceKind.PYTHON:
                encoding, consumed_lines = tokenize.detect_encoding(
                    io.BytesIO(contents).readline
                )
                source_text = contents.decode(encoding, errors="replace")
            source_text = source_text.replace("\r\n", "\n").replace("\r", "\n")
            identity = self.identify_module(entry.relative_path, source_roots)
            digest = (
                hashlib.sha256(contents).hexdigest()
                if entry.relative_path in blobs
                else entry.file_mode
            )
            records.append(
                SourceRecord(
                    entry.relative_path,
                    entry.object_identifier,
                    digest,
                    decision.source_kind,
                    source_text,
                    identity.module_name,
                    identity.distribution_name,
                    identity.is_package,
                )
            )
            if decision.diagnostic is not None:
                diagnostics.append(decision.diagnostic)
        digest = hashlib.sha256(
            "\n".join(
                f"{record.relative_path}\0{record.content_digest}\0{record.source_kind}"
                for record in records
            ).encode("utf-8", errors="surrogateescape")
        ).hexdigest()
        return SourceContents(records, diagnostics, digest)

    def discover_roots(
        self, entries: list[TreeEntry], blobs: dict[str, bytes]
    ) -> list[SourceRoot]:
        """Discover project roots from committed metadata and existing source paths."""
        paths = [entry.relative_path for entry in entries]
        roots: list[SourceRoot] = []
        for entry in entries:
            path = PurePosixPath(entry.relative_path)
            if (
                path.name != SourceFilename.PROJECT
                or len(path.parts) > 2
                or entry.relative_path not in blobs
            ):
                continue
            project_directory = "" if str(path.parent) == "." else str(path.parent)
            metadata = ProjectMetadata.from_json(
                tomllib.loads(blobs[entry.relative_path].decode("utf-8-sig")),
                project_directory,
            )
            if metadata.source_directories:
                roots.extend(metadata.source_directories)
                continue
            conventional = str(
                PurePosixPath(project_directory, SourceFilename.SOURCE_DIRECTORY)
            )
            source_directory = (
                conventional
                if any(candidate.startswith(conventional + "/") for candidate in paths)
                else project_directory
            )
            roots.append(SourceRoot(source_directory, metadata.distribution_name))
        return sorted(
            roots, key=lambda root: len(root.relative_directory), reverse=True
        )

    @staticmethod
    def identify_module(relative_path: str, roots: list[SourceRoot]) -> ModuleIdentity:
        """Associate a Python file with the longest matching source root."""
        path = PurePosixPath(relative_path)
        if path.suffix != SourceSuffix.PYTHON:
            return ModuleIdentity("", "", False)
        for root in roots:
            directory = (
                "" if root.relative_directory == "." else root.relative_directory
            )
            prefix = directory + "/" if directory else ""
            if not relative_path.startswith(prefix):
                continue
            module_path = PurePosixPath(relative_path[len(prefix) :])
            parts = list(module_path.with_suffix("").parts)
            is_package = module_path.name == SourceFilename.PACKAGE
            if is_package:
                parts = parts[:-1]
            if root.module_prefix:
                parts = root.module_prefix.split(".") + parts
            return ModuleIdentity(".".join(parts), root.distribution_name, is_package)
        return ModuleIdentity(
            ".".join(path.with_suffix("").parts),
            "",
            path.name == SourceFilename.PACKAGE,
        )

    @staticmethod
    def classify(entry: TreeEntry) -> InventoryDecision:
        """Determine whether a tracked entry can supply source evidence."""
        path = PurePosixPath(entry.relative_path)
        if entry.file_mode == GitMode.DIRECTORY:
            return InventoryDecision(
                SourceKind.EXCLUDED,
                Diagnostic(
                    entry.relative_path,
                    DiagnosticKind.UNSUPPORTED,
                    "A source path is a directory rather than a regular file.",
                ),
            )
        if path.name == SourceFilename.GENERATED_MAPPING:
            return InventoryDecision(
                SourceKind.EXCLUDED,
                Diagnostic(
                    entry.relative_path,
                    DiagnosticKind.GENERATED,
                    "Generated ORM mappings are excluded before blob access.",
                ),
            )
        if entry.file_mode in (GitMode.SYMBOLIC_LINK, GitMode.SUBMODULE):
            reason = (
                DiagnosticKind.SYMBOLIC_LINK
                if entry.file_mode == GitMode.SYMBOLIC_LINK
                else DiagnosticKind.SUBMODULE
            )
            return InventoryDecision(
                SourceKind.EXCLUDED,
                Diagnostic(
                    entry.relative_path,
                    reason,
                    "Referenced objects are not followed as ordinary source.",
                ),
            )
        if any(part in tuple(ExcludedDirectory) for part in path.parts):
            return InventoryDecision(
                SourceKind.EXCLUDED,
                Diagnostic(
                    entry.relative_path,
                    DiagnosticKind.VENDORED,
                    "Local state or vendored source is outside the inventory policy.",
                ),
            )
        suffix = path.suffix.lower()
        if suffix == SourceSuffix.PYTHON:
            return InventoryDecision(SourceKind.PYTHON)
        if suffix in (SourceSuffix.MARKDOWN, SourceSuffix.RESTRUCTURED_TEXT):
            return InventoryDecision(SourceKind.DOCUMENT)
        if suffix in (
            SourceSuffix.TOML,
            SourceSuffix.YAML,
            SourceSuffix.SHORT_YAML,
            SourceSuffix.JSON,
            SourceSuffix.XML,
        ):
            return InventoryDecision(SourceKind.CONFIGURATION)
        if suffix in (SourceSuffix.SQL, SourceSuffix.JINJA, SourceSuffix.JINJA_SHORT):
            return InventoryDecision(SourceKind.TEMPLATE)
        return InventoryDecision(
            SourceKind.UNSUPPORTED,
            Diagnostic(
                entry.relative_path,
                DiagnosticKind.UNSUPPORTED,
                "The initial source index does not interpret this file format.",
            ),
        )
