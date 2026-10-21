"""
Tests for `Document` loading and directory discovery.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from doc_evaluator.documents import (
    Document,
    NoDocumentsFoundError,
    UnsupportedDocumentFormatError,
    discover_documents,
)

# %% Document.load


def test_load_reads_text_content(tmp_path: Path) -> None:
    path = tmp_path / "report.txt"
    path.write_text("The system achieved 99% accuracy.", encoding="utf-8")

    document = Document.load(path)

    assert document.text == "The system achieved 99% accuracy."
    assert document.path == path


def test_load_rejects_unsupported_extension(tmp_path: Path) -> None:
    path = tmp_path / "report.pdf"
    path.write_bytes(b"%PDF-1.4")

    with pytest.raises(UnsupportedDocumentFormatError):
        Document.load(path)


# %% discover_documents


def test_discover_documents_finds_recognized_files_sorted_by_path(
    document_directory: Path,
) -> None:
    documents = discover_documents(document_directory)

    assert tuple(document.path.name for document in documents) == (
        "first.txt",
        "second.md",
    )


def test_discover_documents_ignores_unsupported_files(document_directory: Path) -> None:
    (document_directory / "notes.pdf").write_bytes(b"%PDF-1.4")

    documents = discover_documents(document_directory)

    assert all(document.path.suffix != ".pdf" for document in documents)


def test_discover_documents_raises_when_directory_has_no_recognized_files(
    tmp_path: Path,
) -> None:
    with pytest.raises(NoDocumentsFoundError):
        discover_documents(tmp_path)
