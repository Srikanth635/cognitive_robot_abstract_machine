"""
Decode canonical knowledge only from a captured committed inventory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing_extensions import TYPE_CHECKING

from repository_knowledge.adapters.exceptions import InvalidKnowledgeDocument
from repository_knowledge.adapters.okf import OkfCodec, ReservedDocument
from repository_knowledge.concepts import (
    KnowledgeBundle,
    KnowledgeFilename,
    KnowledgeIssue,
)
from repository_knowledge.contracts import KnowledgeReader
from repository_knowledge.model import SourceKind

if TYPE_CHECKING:
    from repository_knowledge.model import IndexData

# %% committed document boundary


@dataclass
class CommittedKnowledge(KnowledgeReader):
    """
    Keep publication independent of later filesystem edits or external bundles.
    """

    def read_bundle(self, index: IndexData, bundle_directory: str) -> KnowledgeBundle:
        root = PurePosixPath(bundle_directory)
        if (
            root.is_absolute()
            or ".." in root.parts
            or "\\" in bundle_directory
            or not root.parts
        ):
            raise InvalidKnowledgeDocument(bundle_directory)
        selected = [
            source
            for source in index.sources
            if source.relative_path.startswith(root.as_posix() + "/")
        ]
        if not selected:
            raise InvalidKnowledgeDocument(bundle_directory)
        concepts = []
        navigation = {}
        issues = {}
        for source in sorted(selected, key=lambda item: item.relative_path):
            path = PurePosixPath(source.relative_path)
            if (
                path.suffix != KnowledgeFilename.EXTENSION
                or source.source_kind != SourceKind.DOCUMENT
            ):
                raise InvalidKnowledgeDocument(source.relative_path)
            relative = path.relative_to(root).as_posix()
            if path.name not in (KnowledgeFilename.INDEX, KnowledgeFilename.LOG):
                concepts.append(OkfCodec().decode(relative, source.source_text))
                continue
            navigation[relative] = source.source_text
            if not ReservedDocument(relative, source.source_text).is_conformant():
                issues[relative] = [KnowledgeIssue.INVALID_NAVIGATION]
        return KnowledgeBundle(concepts, navigation, issues)
