"""Ontology loading and structural dataclass generation using KRROOD."""

from krrood_ontology.bootstrap import BootstrapResult, OntologyBootstrap
from krrood_ontology.compiler import CompilationResult, OntologyCompiler
from krrood_ontology.config import (
    BootstrapOptions,
    CompilationOptions,
    LocalImport,
    OntologyInput,
    RDFFormat,
)
from krrood_ontology.individuals import IndividualLoader
from krrood_ontology.inspection import OntologyInspector, OntologyReport
from krrood_ontology.model import OntologyTranslationError
from krrood_ontology.ontology import OntologyDocument
from krrood_ontology.runtime import IndividualLoadingError, LoadedIndividuals

__all__ = [
    "LocalImport",
    "IndividualLoader",
    "IndividualLoadingError",
    "LoadedIndividuals",
    "BootstrapOptions",
    "BootstrapResult",
    "CompilationOptions",
    "CompilationResult",
    "OntologyCompiler",
    "OntologyTranslationError",
    "OntologyBootstrap",
    "OntologyDocument",
    "OntologyInput",
    "OntologyInspector",
    "OntologyReport",
    "RDFFormat",
]
