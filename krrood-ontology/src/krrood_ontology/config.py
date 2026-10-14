"""Validated inputs for the ontology front end."""

from __future__ import annotations

import keyword
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class RDFFormat(str, Enum):
    """RDF syntaxes supported by the first phase."""

    XML = "xml"
    TURTLE = "turtle"
    NTRIPLES = "nt"
    N3 = "n3"


@dataclass(frozen=True)
class OntologyInput:
    """A local ontology file and its optional explicit RDF syntax."""

    path: Path
    format: RDFFormat | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path).resolve())
        if not self.path.is_file():
            raise FileNotFoundError(self.path)


@dataclass(frozen=True)
class BootstrapOptions:
    """The destination and import name of a new vocabulary package."""

    output_directory: Path
    package_name: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "output_directory", Path(self.output_directory).resolve()
        )
        if not self.package_name.isidentifier() or keyword.iskeyword(self.package_name):
            raise ValueError("package_name must be a single valid Python identifier")


@dataclass(frozen=True)
class LocalImport:
    """An explicit ontology import IRI bound to a supplied local RDF file."""

    iri: str
    source: OntologyInput


@dataclass(frozen=True)
class CompilationOptions(BootstrapOptions):
    """Structural compilation settings with explicit semantic-omission consent."""

    structural_only: bool = False
    local_imports: tuple[LocalImport, ...] = ()
    binding_policy: str = "explicit"
    scoped_semantics: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.binding_policy not in {"explicit", "local-usage"}:
            raise ValueError("binding_policy must be explicit or local-usage")
        if self.scoped_semantics and self.binding_policy != "local-usage":
            raise ValueError("scoped_semantics requires binding_policy=local-usage")
        if self.binding_policy == "local-usage" and not self.structural_only:
            raise ValueError("local-usage currently requires structural_only=True")
