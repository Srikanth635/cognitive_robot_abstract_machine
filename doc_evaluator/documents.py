"""
Loading documents from a directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

# %% recognized formats


class DocumentFormat(StrEnum):
    """
    A file extension this evaluator knows how to read as plain text.
    """

    TEXT = ".txt"
    MARKDOWN = ".md"


# %% errors


class UnsupportedDocumentFormatError(Exception):
    """
    Raised when a file's extension is not a recognized :class:`DocumentFormat`.
    """

    def __init__(self, path: Path) -> None:
        recognized = [document_format.value for document_format in DocumentFormat]
        super().__init__(
            f"{path} has an unsupported extension {path.suffix!r}; expected one of {recognized}"
        )
        self.path = path
        """
        The file whose extension was not recognized.
        """


class NoDocumentsFoundError(Exception):
    """
    Raised when a directory contains no files in a recognized :class:`DocumentFormat`.
    """

    def __init__(self, directory: Path) -> None:
        recognized = [document_format.value for document_format in DocumentFormat]
        super().__init__(
            f"No documents with a recognized extension {recognized} were found in {directory}"
        )
        self.directory = directory
        """
        The directory that contained no recognized documents.
        """


# %% Document


@dataclass(frozen=True)
class Document:
    """
    A single document to evaluate.
    """

    path: Path
    """
    Filesystem path the document was read from.
    """

    text: str
    """
    The document's full text content.
    """

    @classmethod
    def load(cls, path: Path) -> Document:
        """
        Read a document from disk.

        :param path: path to a file in a recognized :class:`DocumentFormat`
        :raises UnsupportedDocumentFormatError: if the file's extension is not
            recognized
        :return: the loaded document
        """
        if path.suffix not in set(DocumentFormat):
            raise UnsupportedDocumentFormatError(path)
        return cls(path=path, text=path.read_text(encoding="utf-8"))


# %% directory discovery


def discover_documents(directory: Path) -> tuple[Document, ...]:
    """
    Load every recognized document directly inside a directory.

    :param directory: directory to scan; subdirectories are not descended into
    :raises NoDocumentsFoundError: if no recognized documents are found
    :return: the loaded documents, sorted by path
    """
    paths = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix in set(DocumentFormat)
    )
    if not paths:
        raise NoDocumentsFoundError(directory)
    return tuple(Document.load(path) for path in paths)
