"""Resolve only explicitly supplied local ontology imports."""

from dataclasses import dataclass

from rdflib import OWL, Graph, URIRef

from krrood_ontology.config import LocalImport, OntologyInput
from krrood_ontology.model import CompilationDiagnostic, OntologyTranslationError
from krrood_ontology.ontology import OntologyDocument


class ImportResolutionError(OntologyTranslationError):
    """An import cannot be resolved through the supplied local file map."""

    def __init__(self, message: str, code: str = "import_resolution") -> None:
        super().__init__((CompilationDiagnostic(code, message, str(OWL.imports)),))


@dataclass(frozen=True)
class ResolvedOntology:
    """An asserted local import closure and its constituent source documents."""

    document: OntologyDocument
    sources: tuple[OntologyDocument, ...]
    imported_iris: tuple[str, ...]


@dataclass
class LocalImportResolver:
    """Merge local RDF graphs without fetching any import IRI."""

    def resolve(
        self,
        source: OntologyInput,
        mappings: tuple[LocalImport, ...],
        structural_only: bool,
    ) -> ResolvedOntology:
        root = OntologyDocument.load(source)
        bindings = {item.iri: item.source for item in mappings}
        if len(bindings) != len(mappings):
            raise ImportResolutionError("Duplicate local import IRI mappings")
        graph = Graph()
        pending = [root]
        documents = {}
        imported = set()
        while pending:
            document = pending.pop(0)
            if document.source.path in documents:
                continue
            documents[document.source.path] = document
            graph += document.graph
            for target in sorted(document.graph.objects(None, OWL.imports), key=str):
                if not isinstance(target, URIRef):
                    raise ImportResolutionError("owl:imports requires a named IRI")
                iri = str(target)
                if iri in imported:
                    continue
                imported.add(iri)
                if iri not in bindings:
                    if structural_only and not mappings:
                        continue
                    raise ImportResolutionError(
                        f"No supplied local file for import {iri}; use --import IRI=PATH",
                        "semantic_feature_deferred",
                    )
                pending.append(OntologyDocument.load(bindings[iri]))
        unused = set(bindings) - imported
        if unused:
            raise ImportResolutionError(
                f"Unused local import mappings: {', '.join(sorted(unused))}"
            )
        resolved = bool(mappings)
        if resolved:
            graph.remove((None, OWL.imports, None))
        merged = OntologyDocument(root.source, graph, root.sha256, root.format)
        return ResolvedOntology(
            merged,
            tuple(documents.values()),
            tuple(sorted(imported)) if resolved else (),
        )
