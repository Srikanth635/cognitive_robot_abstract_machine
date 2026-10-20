"""
Extract document headings and explicit links as source evidence.
"""

from __future__ import annotations

import posixpath
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath
from urllib.parse import unquote, urlsplit

from repository_knowledge.contracts import SourceAnalyzer
from repository_knowledge.model import (
    Diagnostic,
    DiagnosticKind,
    FileExtraction,
    ReferenceRecord,
    RelationshipKind,
    SourceKind,
    SourceLocation,
    SourceRecord,
    SymbolKind,
    SymbolRecord,
)

# %% document syntax


class DocumentPattern(StrEnum):
    """
    The heading and link syntax supported by the initial document index.
    """

    MARKDOWN_HEADING = r"^(#{1,6})\s+(.+?)\s*#*\s*$"
    MARKDOWN_LINK = r"\[[^\]]+\]\(([^)]+)\)"
    BACKTICK_FENCE = "```"
    TILDE_FENCE = "~~~"
    RST_SUFFIX = ".rst"


@dataclass(frozen=True)
class Heading:
    """
    A source heading with its exact one-based declaration line.
    """

    title: str
    """
    The title written in the document.
    """

    start_line: int
    """
    The first line belonging to the section.
    """


# %% source document extraction


@dataclass
class DocumentSource(SourceAnalyzer):
    """
    Expose Markdown/RST sections and explicit local source links.
    """

    def extract(self, source: SourceRecord) -> FileExtraction:
        if source.source_kind != SourceKind.DOCUMENT:
            return FileExtraction()
        lines = source.source_text.splitlines()
        extraction = FileExtraction()
        extraction.symbols.append(
            SymbolRecord(
                source.relative_path,
                PurePosixPath(source.relative_path).name,
                "",
                SymbolKind.DOCUMENT,
                SourceLocation(source.relative_path, 1, max(1, len(lines))),
            )
        )
        headings = self._headings(source, lines)
        for offset, heading in enumerate(headings):
            end_line = (
                headings[offset + 1].start_line - 1
                if offset + 1 < len(headings)
                else len(lines)
            )
            extraction.symbols.append(
                SymbolRecord(
                    f"{source.relative_path}#L{heading.start_line}",
                    heading.title,
                    "",
                    SymbolKind.DOCUMENT_SECTION,
                    SourceLocation(
                        source.relative_path,
                        heading.start_line,
                        max(heading.start_line, end_line),
                    ),
                    documentation=heading.title,
                )
            )
        if PurePosixPath(source.relative_path).suffix != DocumentPattern.RST_SUFFIX:
            self._links(source, lines, headings, extraction)
        return extraction

    @staticmethod
    def _headings(source: SourceRecord, lines: list[str]) -> list[Heading]:
        """
        Read supported section declarations without interpreting document code.
        """
        headings: list[Heading] = []
        fenced = False
        for offset, line in enumerate(lines):
            if line.lstrip().startswith(
                (DocumentPattern.BACKTICK_FENCE, DocumentPattern.TILDE_FENCE)
            ):
                fenced = not fenced
                continue
            if fenced:
                continue
            if PurePosixPath(source.relative_path).suffix == DocumentPattern.RST_SUFFIX:
                if offset + 1 >= len(lines) or not line.strip():
                    continue
                underline = lines[offset + 1].strip()
                if (
                    underline
                    and len(underline) >= len(line.strip())
                    and len(set(underline)) == 1
                    and underline[0] in "=-~^#*+"
                ):
                    headings.append(Heading(line.strip(), offset + 1))
                continue
            match = re.match(DocumentPattern.MARKDOWN_HEADING, line)
            if match is not None:
                headings.append(Heading(match.group(2), offset + 1))
        return headings

    @staticmethod
    def _links(
        source: SourceRecord,
        lines: list[str],
        headings: list[Heading],
        extraction: FileExtraction,
    ) -> None:
        """
        Retain local document links without following external resources.
        """
        fenced = False
        for offset, line in enumerate(lines):
            if line.lstrip().startswith(
                (DocumentPattern.BACKTICK_FENCE, DocumentPattern.TILDE_FENCE)
            ):
                fenced = not fenced
                continue
            if fenced:
                continue
            owner = source.relative_path
            for heading in headings:
                if heading.start_line <= offset + 1:
                    owner = f"{source.relative_path}#L{heading.start_line}"
            for match in re.finditer(DocumentPattern.MARKDOWN_LINK, line):
                target = match.group(1).strip("<>")
                parsed = urlsplit(target)
                if parsed.scheme or parsed.netloc:
                    continue
                path = unquote(parsed.path)
                if not path:
                    path = PurePosixPath(source.relative_path).name
                relative = posixpath.normpath(
                    path.lstrip("/")
                    if path.startswith("/")
                    else posixpath.join(
                        str(PurePosixPath(source.relative_path).parent), path
                    )
                )
                if relative == ".." or relative.startswith("../"):
                    extraction.diagnostics.append(
                        Diagnostic(
                            source.relative_path,
                            DiagnosticKind.UNRESOLVED,
                            "A document link points outside the repository snapshot.",
                        )
                    )
                    continue
                extraction.references.append(
                    ReferenceRecord(
                        owner,
                        target,
                        RelationshipKind.DOCUMENTS,
                        SourceLocation(source.relative_path, offset + 1, offset + 1),
                        target_source_path=relative,
                        target_fragment=parsed.fragment,
                    )
                )


# %% analysis composition


@dataclass
class SourceCollection(SourceAnalyzer):
    """
    Combine independent source analyzers into one extraction contract.
    """

    analyzers: list[SourceAnalyzer]
    """
    The supported structural source analyzers.
    """

    def extract(self, source: SourceRecord) -> FileExtraction:
        result = FileExtraction()
        for analyzer in self.analyzers:
            extraction = analyzer.extract(source)
            result.symbols.extend(extraction.symbols)
            result.references.extend(extraction.references)
            result.bindings.extend(extraction.bindings)
            result.diagnostics.extend(extraction.diagnostics)
        return result
