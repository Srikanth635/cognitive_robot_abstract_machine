"""
Capture current working files without confusing them with committed objects.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath, PureWindowsPath
from enum import IntEnum

from repository_knowledge.adapters.git_source import (
    GitArgument,
    GitMode,
    GitSource,
    SourceContents,
    SourceInventory,
    TreeEntry,
)
from repository_knowledge.contracts import SnapshotReader
from repository_knowledge.exceptions import (
    InvalidUntrackedPath,
    SourceChangedDuringCapture,
    UnmergedSourceIndex,
)
from repository_knowledge.model import RepositorySnapshot, SourceKind, SourceScope

# %% checkout capture records


class IndexStage(IntEnum):
    """
    The resolved Git index state accepted by source capture.
    """

    RESOLVED = 0


@dataclass(frozen=True)
class FileState:
    """
    Filesystem identity observed around a source-byte read.
    """

    device: int
    """
    The device containing the source.
    """

    inode: int
    """
    The source's filesystem object identifier.
    """

    byte_count: int
    """
    The observed file extent.
    """

    modification_time: int
    """
    The file's nanosecond modification timestamp.
    """

    status_or_creation_time: int
    """
    The platform's nanosecond ctime timestamp.
    """

    @classmethod
    def capture(cls, path: Path) -> FileState:
        """
        Read metadata without following a final symbolic link.
        """
        metadata = path.stat(follow_symlinks=False)
        return cls(
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        )


@dataclass
class WorkingCapture:
    """
    One observed checkout state and its captured source bytes.
    """

    checkout_commit: str
    """
    The HEAD observed during capture.
    """

    index_digest: str
    """
    The digest of staged index entries.
    """

    contents: SourceContents
    """
    The captured source inventory and contents.
    """


# %% current working-file snapshots


@dataclass
class WorkingTreeSource(SnapshotReader):
    """
    Capture a checkout's current tracked and explicitly selected source files.
    """

    repository_root: Path
    """
    The checkout supplying current source contents.
    """

    included_untracked_paths: tuple[str, ...] = ()
    """
    Repository-relative untracked files or directories selected for capture.
    """

    committed_source: GitSource = field(init=False)
    """
    The committed baseline and shared Git access boundary.
    """

    inventory: SourceInventory = field(default_factory=SourceInventory)
    """
    The source classification and import-root policy.
    """

    def __post_init__(self) -> None:
        """
        Validate untracked selections and bind the canonical checkout.
        """
        self.repository_root = self.repository_root.resolve()
        selections: list[str] = []
        for selection in self.included_untracked_paths:
            path = PurePosixPath(selection)
            if (
                not selection
                or path.is_absolute()
                or PureWindowsPath(selection).drive
                or ".." in path.parts
                or "\\" in selection
            ):
                raise InvalidUntrackedPath(selection)
            selections.append(str(path))
        self.included_untracked_paths = tuple(sorted(set(selections)))
        self.committed_source = GitSource(self.repository_root)

    @property
    def source_scope(self) -> SourceScope:
        return SourceScope.WORKING_TREE

    def resolve_revision(self, revision: str) -> str:
        return self.committed_source.resolve_revision(revision)

    def read_snapshot(self, revision: str) -> RepositorySnapshot:
        base = self.committed_source.read_snapshot(revision)
        before = self.capture_files()
        after = self.capture_files()
        if (
            before != after
            or self.resolve_revision(revision) != base.manifest.commit_identifier
        ):
            raise SourceChangedDuringCapture(str(self.repository_root))
        git_directory = (
            self.committed_source.checkout.run(
                GitArgument.REVISION, GitArgument.GIT_DIRECTORY
            )
            .decode("utf-8", errors="surrogateescape")
            .strip()
        )
        worktree_identifier = hashlib.sha256(
            "\n".join(
                (str(self.repository_root), str(Path(git_directory).resolve()))
            ).encode("utf-8", errors="surrogateescape")
        ).hexdigest()
        snapshot_identifier = hashlib.sha256(
            "\n".join(
                (
                    base.manifest.snapshot_identifier,
                    before.contents.source_digest,
                    worktree_identifier,
                    before.index_digest,
                    before.checkout_commit,
                    *self.included_untracked_paths,
                )
            ).encode("utf-8", errors="surrogateescape")
        ).hexdigest()
        manifest = replace(
            base.manifest,
            snapshot_identifier=snapshot_identifier,
            source_digest=before.contents.source_digest,
            source_count=len(before.contents.sources),
            source_scope=self.source_scope,
            worktree_identifier=worktree_identifier,
            index_digest=before.index_digest,
            checkout_commit=before.checkout_commit,
            included_untracked_paths=self.included_untracked_paths,
        )
        return RepositorySnapshot(
            manifest, before.contents.sources, before.contents.diagnostics
        )

    def capture_files(self) -> WorkingCapture:
        """
        Read one observable index, HEAD and working-file inventory.
        """
        checkout = self.committed_source.checkout
        checkout_commit = self.resolve_revision(GitArgument.CHECKOUT)
        index_listing = checkout.run(
            GitArgument.FILES, GitArgument.STAGE, GitArgument.NULL_DELIMITED
        )
        entries = self.read_entries(index_listing)
        contents = self.read_contents(entries)
        if (
            checkout.run(
                GitArgument.FILES, GitArgument.STAGE, GitArgument.NULL_DELIMITED
            )
            != index_listing
            or self.resolve_revision(GitArgument.CHECKOUT) != checkout_commit
        ):
            raise SourceChangedDuringCapture(str(self.repository_root))
        return WorkingCapture(
            checkout_commit,
            hashlib.sha256(index_listing).hexdigest(),
            contents,
        )

    def read_entries(self, index_listing: bytes) -> list[TreeEntry]:
        """
        Select tracked and explicitly included untracked index paths.
        """
        checkout = self.committed_source.checkout
        entries_by_path: dict[str, TreeEntry] = {}
        for listing in index_listing.split(b"\0"):
            if not listing:
                continue
            metadata, path_bytes = listing.split(b"\t", 1)
            mode, object_identifier, stage = metadata.decode("ascii").split(" ")
            relative_path = path_bytes.decode("utf-8", errors="surrogateescape")
            if int(stage) != IndexStage.RESOLVED:
                raise UnmergedSourceIndex(relative_path)
            entries_by_path[relative_path] = TreeEntry(relative_path, "", mode)
        if self.included_untracked_paths:
            untracked = checkout.run(
                GitArgument.FILES,
                GitArgument.OTHERS,
                GitArgument.EXCLUDE_STANDARD,
                GitArgument.NULL_DELIMITED,
            )
            for path_bytes in untracked.split(b"\0"):
                if not path_bytes:
                    continue
                relative_path = path_bytes.decode("utf-8", errors="surrogateescape")
                if any(
                    selection == "."
                    or relative_path == selection
                    or relative_path.startswith(selection + "/")
                    for selection in self.included_untracked_paths
                ):
                    entries_by_path[relative_path] = TreeEntry(
                        relative_path, "", GitMode.REGULAR
                    )
        return [entry for relative_path, entry in sorted(entries_by_path.items())]

    def read_contents(self, tree_entries: list[TreeEntry]) -> SourceContents:
        """
        Read supported working files while retaining policy exclusions.
        """
        entries: list[TreeEntry] = []
        blobs: dict[str, bytes] = {}
        for entry in tree_entries:
            relative_path = entry.relative_path
            path = self.repository_root / relative_path
            ancestors = [path, *path.parents]
            if any(
                ancestor.is_symlink() or ancestor.is_junction()
                for ancestor in ancestors
                if ancestor != self.repository_root
                and ancestor.is_relative_to(self.repository_root)
            ):
                entries.append(replace(entry, file_mode=GitMode.SYMBOLIC_LINK))
                continue
            if not path.exists():
                continue
            if not path.is_file() and entry.file_mode != GitMode.SUBMODULE:
                entries.append(replace(entry, file_mode=GitMode.DIRECTORY))
                continue
            entries.append(entry)
            if self.inventory.classify(entry).source_kind in (
                SourceKind.EXCLUDED,
                SourceKind.UNSUPPORTED,
            ):
                continue
            before = FileState.capture(path)
            contents = path.read_bytes()
            if FileState.capture(path) != before:
                raise SourceChangedDuringCapture(relative_path)
            blobs[relative_path] = contents
        return self.inventory.capture(entries, blobs)
