"""Exchange OKF Markdown while preserving portable metadata extensions."""

from __future__ import annotations

import posixpath
import calendar
import re
from dataclasses import dataclass, replace
from datetime import date, datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath
from urllib.parse import quote, urlsplit

import yaml
from typing_extensions import Any

from repository_knowledge.concepts import (
    Concept,
    ConceptEvidence,
    ConceptProvenance,
    KnowledgeBundle,
    Lifecycle,
    KnowledgeIssue,
    KnowledgeFilename,
)
from repository_knowledge.adapters.documents import DocumentSource
from repository_knowledge.adapters.exceptions import InvalidKnowledgeDocument
from repository_knowledge.model import SourceLocation, SourceRecord, SourceKind

# %% external metadata vocabulary


class OkfField(StrEnum):
    """Official metadata and the explicitly namespaced CRAM extension."""

    TYPE = "type"
    TITLE = "title"
    DESCRIPTION = "description"
    STATUS = "status"
    STALE_AFTER = "stale_after"
    CRAM = "cram"


class ReservedField(StrEnum):
    """Metadata allowed exclusively on the root directory listing."""

    VERSION = "okf_version"


class ProvenanceField(StrEnum):
    """Repository extension keys, independent of official OKF fields."""

    COMMIT = "source_commit"
    EVIDENCE = "evidence"
    DEPENDS_ON = "depends_on"
    REPLACEMENT = "replacement"
    SOURCE = "source_id"
    PATH = "path"
    START = "start_line"
    END = "end_line"
    DIGEST = "content_digest"
    SYMBOL = "symbol"


@dataclass
class PortableValues:
    """Normalize safe YAML scalars for portable metadata consumers."""

    maximum_depth: int = 32
    """The maximum nested metadata depth accepted for exchange."""

    def normalize(self, value: Any, ancestors: frozenset[int] = frozenset()) -> Any:
        """Retain mapping/list/scalar values and spell timestamps as ISO text."""
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        if len(ancestors) > self.maximum_depth or id(value) in ancestors:
            raise InvalidKnowledgeDocument(type(value).__name__)
        parents = ancestors | {id(value)}
        if isinstance(value, dict):
            return {
                str(key): self.normalize(item, parents) for key, item in value.items()
            }
        if isinstance(value, list):
            return [self.normalize(item, parents) for item in value]
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        raise InvalidKnowledgeDocument(type(value).__name__)


@dataclass(frozen=True)
class EvidenceMetadata:
    """The external evidence entry shape owned by the CRAM extension."""

    source_identifier: str
    """The source footnote label."""
    location: SourceLocation
    """The repository-relative source span."""
    content_digest: str
    """The original source byte digest."""
    symbol_identifier: str
    """The optional qualified source declaration."""

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> EvidenceMetadata:
        """Interpret one namespaced evidence entry without executing source."""
        return cls(
            str(data[ProvenanceField.SOURCE]),
            SourceLocation(
                str(data[ProvenanceField.PATH]),
                int(data[ProvenanceField.START]),
                int(data[ProvenanceField.END]),
            ),
            str(data[ProvenanceField.DIGEST]),
            str(data.get(ProvenanceField.SYMBOL, "")),
        )


@dataclass
class ProvenanceMetadata:
    """Translate the documented extension into format-independent evidence."""

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> ConceptProvenance:
        """Read source lineage and preserve additional extension metadata."""
        entries = [
            EvidenceMetadata.from_json(item)
            for item in data.get(ProvenanceField.EVIDENCE, [])
        ]
        return ConceptProvenance(
            str(data.get(ProvenanceField.COMMIT, "")),
            [
                ConceptEvidence(
                    item.source_identifier,
                    item.location,
                    item.content_digest,
                    item.symbol_identifier,
                )
                for item in entries
            ],
            list(data.get(ProvenanceField.DEPENDS_ON, [])),
            str(data.get(ProvenanceField.REPLACEMENT, "")),
            {
                key: value
                for key, value in data.items()
                if key
                not in (
                    ProvenanceField.COMMIT,
                    ProvenanceField.EVIDENCE,
                    ProvenanceField.DEPENDS_ON,
                    ProvenanceField.REPLACEMENT,
                )
            },
        )


# %% document interchange


@dataclass
class OkfCodec:
    """Read permissive OKF concepts and export supported metadata losslessly."""

    def decode(self, relative_path: str, text: str) -> Concept:
        """Read a UTF-8 concept with a delimited metadata mapping."""
        lines = text.splitlines(keepends=True)
        delimiters = [
            offset for offset, line in enumerate(lines) if line.strip() == "---"
        ]
        if not delimiters or delimiters[0] != 0 or len(delimiters) < 2:
            raise InvalidKnowledgeDocument(relative_path)
        end = delimiters[1]
        metadata = PortableValues().normalize(yaml.safe_load("".join(lines[1:end])))
        if not isinstance(metadata, dict):
            raise InvalidKnowledgeDocument(relative_path)
        header = ConceptMetadata.from_json(metadata)
        body = "".join(lines[end + 1 :])
        source = metadata.get(OkfField.CRAM)
        if source is not None and not isinstance(source, dict):
            raise InvalidKnowledgeDocument(relative_path)
        return Concept(
            relative_path,
            header.concept_type,
            header.title,
            header.description,
            body,
            header.lifecycle,
            header.stale_after,
            ProvenanceMetadata.from_json(source) if source is not None else None,
            {key: value for key, value in metadata.items() if key not in OkfField},
            self.read_links(relative_path, body),
            list(dict.fromkeys(re.findall(r"\[\^([^\]]+)\](?!:)", body))),
        )

    def encode(self, concept: Concept) -> str:
        """Export a concept without manufacturing generation or review events."""
        metadata = dict(concept.metadata)
        metadata[OkfField.TYPE.value] = str(concept.concept_type)
        if concept.title:
            metadata[OkfField.TITLE.value] = concept.title
        if concept.description:
            metadata[OkfField.DESCRIPTION.value] = concept.description
        metadata[OkfField.STATUS.value] = str(concept.lifecycle)
        if concept.stale_after:
            metadata[OkfField.STALE_AFTER.value] = concept.stale_after
        if concept.provenance is not None:
            provenance = concept.provenance
            extension = dict(provenance.extensions)
            extension.update(
                {
                    ProvenanceField.COMMIT.value: provenance.source_commit,
                    ProvenanceField.DEPENDS_ON.value: provenance.depends_on,
                    ProvenanceField.REPLACEMENT.value: provenance.replacement,
                    ProvenanceField.EVIDENCE.value: [
                        {
                            ProvenanceField.SOURCE.value: item.source_identifier,
                            ProvenanceField.PATH.value: item.location.relative_path,
                            ProvenanceField.START.value: item.location.start_line,
                            ProvenanceField.END.value: item.location.end_line,
                            ProvenanceField.DIGEST.value: item.content_digest,
                            ProvenanceField.SYMBOL.value: item.symbol_identifier,
                        }
                        for item in provenance.evidence
                    ],
                }
            )
            metadata[OkfField.CRAM.value] = extension
        return (
            "---\n"
            + yaml.safe_dump(metadata, allow_unicode=True, sort_keys=False)
            + "---\n"
            + concept.body
        )

    def prepare_move(self, concept: Concept, replacement_path: str) -> Concept:
        """Preserve supported body-link destinations across a directory change."""
        extraction = DocumentSource().extract(
            SourceRecord(
                concept.relative_path, "", "", SourceKind.DOCUMENT, concept.body
            )
        )
        body = concept.body
        for reference in extraction.references:
            target = reference.target_source_path
            if target == concept.relative_path:
                target = replacement_path
            original = urlsplit(reference.target_expression)
            locator = "/" + quote(target, safe="/")
            if original.query:
                locator += f"?{original.query}"
            if original.fragment:
                locator += f"#{original.fragment}"
            body = body.replace(f"({reference.target_expression})", f"({locator})")
        return replace(
            concept, body=body, links=self.read_links(concept.relative_path, body)
        )

    @staticmethod
    def read_links(relative_path: str, body: str) -> list[str]:
        """Read supported local links using the shared source-document analyzer."""
        extraction = DocumentSource().extract(
            SourceRecord(relative_path, "", "", SourceKind.DOCUMENT, body)
        )
        return [
            (
                posixpath.join(reference.target_source_path, KnowledgeFilename.INDEX)
                if urlsplit(reference.target_expression).path.endswith("/")
                else reference.target_source_path
            )
            for reference in extraction.references
        ]


@dataclass(frozen=True)
class ConceptMetadata:
    """The official scalar fields interpreted at the exchange boundary."""

    concept_type: str
    """The mandatory nonempty producer-defined classification."""
    title: str
    """The optional discovery title."""
    description: str
    """The optional discovery summary."""
    lifecycle: str
    """The producer's lifecycle declaration."""
    stale_after: str
    """The optional expiry timestamp spelling."""

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> ConceptMetadata:
        """Read supported scalar metadata without requiring optional fields."""
        concept_type = data.get(OkfField.TYPE, "")
        return cls(
            concept_type if isinstance(concept_type, str) else "",
            str(data.get(OkfField.TITLE, "")),
            str(data.get(OkfField.DESCRIPTION, "")),
            str(data.get(OkfField.STATUS, Lifecycle.STABLE)),
            str(data.get(OkfField.STALE_AFTER, "")),
        )


# %% bundle filesystem boundary


@dataclass
class OkfDirectory:
    """Import or export a local bundle independently of source indexing."""

    directory: Path
    """The dedicated root of the knowledge bundle."""

    def read_bundle(self) -> KnowledgeBundle:
        """Read concepts and reserved documents without following filesystem links."""
        root = self.directory.resolve()
        if not root.is_dir():
            raise InvalidKnowledgeDocument(str(self.directory))
        concepts: list[Concept] = []
        navigation: dict[str, str] = {}
        navigation_issues: dict[str, list[KnowledgeIssue]] = {}
        for path in sorted(root.rglob("*.md")):
            if not path.resolve().is_relative_to(root) or path.is_symlink():
                raise InvalidKnowledgeDocument(str(path))
            relative = path.relative_to(root).as_posix()
            text = path.read_text(encoding="utf-8")
            if path.name in (KnowledgeFilename.INDEX, KnowledgeFilename.LOG):
                navigation[relative] = text
                if not ReservedDocument(relative, text).is_conformant():
                    navigation_issues[relative] = [KnowledgeIssue.INVALID_NAVIGATION]
            else:
                concepts.append(OkfCodec().decode(relative, text))
        return KnowledgeBundle(concepts, navigation, navigation_issues)

    def write_bundle(self, bundle: KnowledgeBundle) -> None:
        """Write a complete proposal only into a new directory."""
        if self.directory.exists():
            raise InvalidKnowledgeDocument(str(self.directory))
        root = self.directory.resolve()
        documents = {
            concept.relative_path: OkfCodec().encode(concept)
            for concept in bundle.concepts
        }
        for relative in bundle.navigation:
            if PurePosixPath(relative).name not in (
                KnowledgeFilename.INDEX,
                KnowledgeFilename.LOG,
            ):
                raise InvalidKnowledgeDocument(relative)
        documents.update(bundle.navigation)
        for relative in documents:
            path = root / relative
            if not path.resolve().is_relative_to(root):
                raise InvalidKnowledgeDocument(relative)
        for relative, text in documents.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8", newline="\n")


@dataclass(frozen=True)
class ReservedDocument:
    """Check reserved document structure without imposing concept requirements."""

    relative_path: str
    """The index or update-history path within the bundle."""
    text: str
    """The original reserved document content."""

    def is_conformant(self) -> bool:
        """Check frontmatter placement and dated update-history headings."""
        lines = self.text.splitlines()
        if not lines:
            return False
        if lines[0] == "---":
            if self.relative_path != KnowledgeFilename.INDEX:
                return False
            delimiters = [offset for offset, line in enumerate(lines) if line == "---"]
            if len(delimiters) < 2:
                return False
            metadata = yaml.safe_load("\n".join(lines[1 : delimiters[1]]))
            if not isinstance(metadata, dict) or set(metadata) != {
                ReservedField.VERSION
            }:
                return False
        if PurePosixPath(self.relative_path).name != KnowledgeFilename.LOG:
            return any(line.startswith("# ") for line in lines)
        headings = [line[3:].strip() for line in lines if line.startswith("## ")]
        for heading in headings:
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", heading) is None:
                return False
            year, month, day = map(int, heading.split("-"))
            if year < 1 or not 1 <= month <= 12:
                return False
            if day < 1 or day > calendar.monthrange(year, month)[1]:
                return False
        return bool(headings)
