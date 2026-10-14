"""Report observed RDF predicates and their compiler/runtime treatment."""

from collections import Counter
from dataclasses import dataclass

from rdflib import OWL, RDF, RDFS

from krrood_ontology.model import ModelDefinition
from krrood_ontology.ontology import OntologyDocument


@dataclass
class FeatureReporter:
    """Expose bounded coverage without claiming full OWL conformance."""

    def report(
        self, document: OntologyDocument, model: ModelDefinition
    ) -> dict[str, object]:
        counts = Counter(str(predicate) for _, predicate, _ in document.graph)
        structural = {
            str(term) for term in (RDF.type, RDFS.subClassOf, RDFS.domain, RDFS.range)
        }
        semantic = {str(term) for term in (OWL.inverseOf, RDFS.subPropertyOf)}
        omitted = {
            item.iri
            for item in model.diagnostics
            if item.code == "semantic_feature_deferred"
        }
        properties = {item.iri for item in model.properties}
        expressions = {
            str(term)
            for term in (
                OWL.equivalentClass,
                OWL.intersectionOf,
                OWL.unionOf,
                OWL.oneOf,
                OWL.onProperty,
                OWL.hasValue,
                OWL.someValuesFrom,
            )
        }
        annotations = {str(RDFS.label), str(RDFS.comment)} | {
            str(subject)
            for subject in document.graph.subjects(RDF.type, OWL.AnnotationProperty)
        }
        predicates = []
        for iri, count in sorted(counts.items()):
            if model.scoped_semantics and iri in {
                str(OWL.equivalentClass),
                str(OWL.unionOf),
                str(OWL.intersectionOf),
                str(OWL.oneOf),
            }:
                treatment = "scoped_positive_definition_mapping_and_retained_axioms"
            elif iri in omitted:
                treatment = "omitted_semantics"
            elif iri in structural:
                treatment = (
                    "named_inheritance_and_retained_anonymous_axioms"
                    if model.binding_policy == "local-usage"
                    and iri == str(RDFS.subClassOf)
                    else "structural_mapping_and_runtime_types"
                )
            elif iri in semantic:
                treatment = "native_relation_mapping"
            elif iri in expressions:
                treatment = "positive_expression_query_mapping"
            elif iri in properties:
                treatment = "runtime_assertion_preflight"
            elif iri in annotations:
                treatment = "runtime_annotation_or_source_metadata"
            else:
                treatment = "retained_source_only"
            predicates.append(
                {"iri": iri, "triple_count": count, "treatment": treatment}
            )
        type_terms = []
        for iri, count in sorted(
            Counter(
                str(term) for term in document.graph.objects(None, RDF.type)
            ).items()
        ):
            treatment = (
                "omitted_semantics"
                if iri in omitted
                else (
                    "runtime_type_preflight"
                    if iri in {item.iri for item in model.classes}
                    else (
                        "native_relation_mapping"
                        if iri
                        in {str(OWL.TransitiveProperty), str(OWL.SymmetricProperty)}
                        else "declaration_or_source_metadata"
                    )
                )
            )
            if model.scoped_semantics and iri == str(OWL.ReflexiveProperty):
                treatment = "explicit_scoped_native_self_edges"
            elif model.scoped_semantics and iri == str(OWL.AsymmetricProperty):
                treatment = "scoped_conflict_validation"
            type_terms.append(
                {"iri": iri, "assertion_count": count, "treatment": treatment}
            )
        return {
            "profile": "named_classes_explicit_types_additive_relations_positive_queries",
            "complete_owl_conformance": False,
            "scoped_validation": {
                "available": True,
                "automatic_enforcement": False,
                "profile": "mapped_fields_explicit_structural_types",
                "default_complete": False,
                "default_identity_policy": "owl",
                "unsupported_axioms": "reported_by_validator_coverage",
            },
            "binding_policy": model.binding_policy,
            "scoped_semantics": model.scoped_semantics,
            "property_iris": len(model.properties),
            "field_associations": (
                len(model.property_bindings)
                if model.binding_policy == "local-usage"
                else len(model.properties)
            ),
            "unbound_property_iris": (
                sorted(
                    {p.iri for p in model.properties}
                    - {p.iri for p in model.property_bindings}
                )
                if model.binding_policy == "local-usage"
                else []
            ),
            "preserved_axiom_records": len(model.schema_axioms),
            "predicates": predicates,
            "rdf_type_terms": type_terms,
            "diagnostics": [
                {"code": item.code, "iri": item.iri, "message": item.message}
                for item in model.diagnostics
            ],
        }
