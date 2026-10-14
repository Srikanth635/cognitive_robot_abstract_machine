"""Preserve RDF source or project checked mapped values with explicit inference policy."""

from dataclasses import dataclass
from pathlib import Path

from rdflib import Graph, Literal, URIRef

from krrood_ontology.constraints import ConstraintTranslationError
from krrood_ontology.individuals import IndividualPlanner
from krrood_ontology.relation_runtime import ModelRelations
from krrood_ontology.runtime import GeneratedModel, LoadedIndividuals
from krrood_ontology.values import value_key


class IndividualExportError(ValueError):
    """An object registry or field value cannot be projected without silent loss."""


@dataclass(frozen=True)
class ExportResult:
    graph: Graph
    mode: str
    added_triples: int = 0
    removed_triples: int = 0
    omitted_inferred_facts: int = 0


@dataclass
class IndividualExporter:
    model: GeneratedModel

    def export(
        self, loaded: LoadedIndividuals, *, mode="source", include_inferred=False
    ):
        """Source copies asserted RDF; current updates mapped assertions in that copy."""
        if mode not in {"source", "current"} or type(include_inferred) is not bool:
            raise ValueError(
                "mode must be source or current; include_inferred must be a boolean"
            )
        if mode == "source" and include_inferred:
            raise ValueError("include_inferred applies only to mode=current")
        graph = Graph()
        for prefix, namespace in loaded.asserted_graph.namespaces():
            graph.bind(prefix, namespace)
        graph += loaded.asserted_graph
        if mode == "source":
            return ExportResult(graph, mode)
        relations = ModelRelations(self.model, {})
        objects = relations._objects(loaded)
        identities = {id(v): k for k, v in objects.items()}
        flags = {
            (f.subject_iri, f.property_iri, f.object_iri): f.inferred
            for f in relations.snapshot(loaded)
        }
        omitted = 0
        planner = IndividualPlanner(self.model, identity_policy=loaded.identity_policy)
        plans, _, _ = planner.plan_graph(loaded.asserted_graph)
        expected = {p.iri: p for p in plans}
        if objects.keys() != expected.keys():
            raise IndividualExportError(
                "Current export requires unchanged registry membership; adding or deleting individuals needs an explicit lifecycle policy"
            )
        if any(
            type(objects[iri]) is not self.model.classes[p.class_iri]
            for iri, p in expected.items()
        ):
            raise IndividualExportError(
                "Current export does not change asserted types or reclassify objects"
            )
        for iri, obj in sorted(objects.items()):
            for prop_iri in self.model.properties:
                mapping = self.model.mapping_for(
                    prop_iri, type(obj).__ontology_iri__, require_applicable=True
                )
                if mapping is None:
                    continue
                subject, predicate = URIRef(iri), URIRef(prop_iri)
                original = tuple(loaded.asserted_graph.objects(subject, predicate))
                projected = set()
                for value in tuple(getattr(obj, mapping.name)):
                    if mapping.kind == "object":
                        target_iri = identities.get(id(value))
                        if target_iri is None:
                            raise IndividualExportError(
                                "Object target is outside the supplied loaded registry"
                            )
                        term = URIRef(target_iri)
                        inferred = flags.get((iri, prop_iri, target_iri), False)
                        if inferred and term not in original and not include_inferred:
                            omitted += 1
                            continue
                        projected.add(term)
                    else:
                        try:
                            key = value_key(value)
                            preserved = {
                                term
                                for term in original
                                if isinstance(term, Literal)
                                and value_key(term.toPython()) == key
                            }
                            if preserved:
                                projected.update(preserved)
                                continue
                            term = Literal(
                                value,
                                datatype=(
                                    URIRef(mapping.range_iri)
                                    if mapping.range_iri
                                    else None
                                ),
                            )
                            converted = planner.literal(term, mapping, iri)
                            if value_key(converted) != key:
                                raise IndividualExportError(
                                    "Literal conversion changes the supplied value"
                                )
                            projected.add(term)
                        except (ConstraintTranslationError, TypeError) as error:
                            raise IndividualExportError(str(error)) from error
                graph.remove((subject, predicate, None))
                for term in projected:
                    graph.add((subject, predicate, term))
        # Reuse the loader's full preflight, including domains, references and identity lists.
        # No objects or native relations are created during export.
        planner.plan_graph(graph)
        original = set(loaded.asserted_graph)
        current = set(graph)
        return ExportResult(
            graph, mode, len(current - original), len(original - current), omitted
        )

    def graph(self, loaded, **options):
        return self.export(loaded, **options).graph

    def write(
        self, loaded, destination, *, mode="source", include_inferred=False, format=None
    ):
        """Write only after export preflight succeeds; never overwrite an existing file."""
        result = self.export(loaded, mode=mode, include_inferred=include_inferred)
        destination = Path(destination)
        if destination.exists():
            raise FileExistsError(destination)
        content = result.graph.serialize(format=format or "turtle", encoding="utf-8")
        with destination.open("xb") as stream:
            stream.write(content)
        return result
