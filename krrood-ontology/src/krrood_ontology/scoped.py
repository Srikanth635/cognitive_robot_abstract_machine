"""Opt-in positive definitions and shared native relation associations."""

from dataclasses import replace
from graphlib import CycleError, TopologicalSorter

from rdflib import OWL, RDF, RDFS, BNode, Graph, URIRef

from krrood_ontology.constraints import ConstraintParser, ConstraintTranslationError
from krrood_ontology.model import (
    CompilationDiagnostic,
    ExpressionDefinition,
    ExpressionKind,
    OntologyTranslationError,
    PropertyKind,
)
from krrood_ontology.semantics import RelationSemantics


class ScopedSemantics:
    """Project storage for native descriptors; preserve asserted signatures separately."""

    def translate(self, document, model):
        graph = document.graph
        classes = {c.iri: c for c in model.classes}
        equivalences = Graph()
        anonymous = {}
        for left, right in graph.subject_objects(OWL.equivalentClass):
            if isinstance(left, URIRef) and isinstance(right, URIRef):
                if str(left) not in classes or str(right) not in classes:
                    self.fail(
                        "definition_class",
                        "Equivalent classes must be registered",
                        str(left),
                    )
                equivalences.add((left, OWL.equivalentClass, right))
                equivalences.add((right, OWL.equivalentClass, left))
            elif isinstance(left, URIRef) and isinstance(right, BNode):
                anonymous.setdefault(str(left), set()).add(right)
            elif isinstance(right, URIRef) and isinstance(left, BNode):
                anonymous.setdefault(str(right), set()).add(left)
            else:
                self.fail(
                    "definition_owner",
                    "Anonymous-to-anonymous definitions are unsupported",
                )
        aliases = {}
        groups = {}
        for iri in classes:
            component = tuple(
                sorted(
                    map(
                        str,
                        equivalences.transitive_objects(
                            URIRef(iri), OWL.equivalentClass
                        ),
                    )
                )
            )
            groups[iri] = component[0]
            aliases[iri] = tuple(v for v in component if v != iri)
        parser = ConstraintParser(graph, model)
        definitions = {}
        for iri in sorted(classes):
            nodes = set().union(
                *(anonymous.get(v, set()) for v in (iri,) + aliases[iri])
            )
            if len(nodes) > 1:
                self.fail(
                    "multiple_definitions",
                    "A named equivalence group supports one anonymous definition",
                    iri,
                )
            if nodes:
                try:
                    expression = parser.parse(next(iter(nodes)))
                except ConstraintTranslationError as error:
                    self.fail("positive_definition", str(error), iri)
                self.positive(expression)
                definitions[iri] = expression
        dependencies = {groups[iri]: set() for iri in classes}
        for concept in model.classes:
            for parent in concept.asserted_parent_iris:
                if parent in groups and groups[parent] != groups[concept.iri]:
                    dependencies[groups[parent]].add(groups[concept.iri])
        for iri, expression in definitions.items():
            dependencies[groups[iri]].update(
                groups[v] for v in self.references(expression) if v in groups
            )
        try:
            tuple(TopologicalSorter(dependencies).static_order())
        except CycleError:
            self.fail(
                "definition_cycle", "Recursive positive definitions are unsupported"
            )
        affected = set()
        reflexive = tuple(
            sorted(map(str, graph.subjects(RDF.type, OWL.ReflexiveProperty)))
        )
        for kind in (
            OWL.TransitiveProperty,
            OWL.SymmetricProperty,
            OWL.ReflexiveProperty,
            OWL.AsymmetricProperty,
        ):
            affected.update(map(str, graph.subjects(RDF.type, kind)))
        for predicate in (OWL.inverseOf, RDFS.subPropertyOf):
            for a, b in graph.subject_objects(predicate):
                affected.update((str(a), str(b)))
        registered = {p.iri: p for p in model.properties}
        for iri in affected:
            if iri not in registered or registered[iri].kind != PropertyKind.OBJECT:
                self.fail(
                    "semantic_property_kind",
                    "Active relation requires a declared object property",
                    iri,
                )
        roots = tuple(
            replace(
                registered[iri],
                domain_iri="",
                domain_name="_OntologyEntity",
                range_type="_OntologyEntity",
                container="set",
                constructor=False,
            )
            for iri in sorted(affected)
        )
        # KRROOD owns inverse/transitive/superproperty execution and its signature checks.
        # A shared association supplies a homogeneous storage signature across named types.
        native_properties = tuple(
            (
                replace(
                    p,
                    domain_iri="",
                    range_iri="",
                    domain_name="_OntologyEntity",
                    range_type="_OntologyEntity",
                    container="set",
                    constructor=False,
                )
                if p.iri in affected
                else p
            )
            for p in model.properties
        )
        native = RelationSemantics().translate(
            document, replace(model, classes=(), properties=native_properties)
        )
        root_by_iri = {p.iri: p for p in roots}
        bindings = [p for p in model.property_bindings if p.iri not in affected]
        projected = []
        for concept in model.classes:
            own = {p.iri: p for p in concept.properties if p.iri not in affected}
            ancestors = set(
                map(str, graph.transitive_objects(URIRef(concept.iri), RDFS.subClassOf))
            )
            inherited = {p.iri for p in bindings if p.domain_iri in ancestors}
            if concept.iri in definitions:
                for prop_iri in self.expression_properties(definitions[concept.iri]):
                    if prop_iri in affected or prop_iri in inherited:
                        continue
                    copied = replace(
                        registered[prop_iri],
                        domain_iri=concept.iri,
                        domain_name=concept.name,
                    )
                    own[prop_iri] = copied
                    bindings.append(copied)
                    inherited.add(prop_iri)
            for other in aliases[concept.iri]:
                other_ancestors = set(
                    map(str, graph.transitive_objects(URIRef(other), RDFS.subClassOf))
                )
                for prop in tuple(
                    p
                    for p in model.property_bindings
                    if p.domain_iri in other_ancestors
                ):
                    if prop.iri in affected or prop.iri in own or prop.iri in inherited:
                        continue
                    # Do not duplicate fields already supplied by an asserted ancestor.
                    existing = next(
                        (
                            b
                            for b in bindings
                            if b.iri == prop.iri and b.domain_iri == concept.iri
                        ),
                        None,
                    )
                    if existing is None:
                        copied = replace(
                            prop, domain_iri=concept.iri, domain_name=concept.name
                        )
                        own[prop.iri] = copied
                        bindings.append(copied)
                        inherited.add(prop.iri)
            projected.append(replace(concept, properties=tuple(own.values())))
        globals_ = tuple(
            replace(p, container="set", constructor=False) if p.iri in affected else p
            for p in model.properties
        )
        handled = {
            str(v)
            for v in (
                OWL.equivalentClass,
                OWL.unionOf,
                OWL.intersectionOf,
                OWL.oneOf,
                OWL.inverseOf,
                OWL.TransitiveProperty,
                OWL.SymmetricProperty,
                OWL.ReflexiveProperty,
                OWL.AsymmetricProperty,
                RDFS.subPropertyOf,
            )
        }
        diagnostics = tuple(
            (
                replace(
                    d,
                    code="scoped_semantic_profile",
                    message="Handled by the bounded scoped profile; source axioms are retained and full OWL reasoning is not claimed",
                )
                if d.iri in handled and d.code == "semantic_feature_deferred"
                else d
            )
            for d in model.diagnostics
        )
        return replace(
            model,
            diagnostics=diagnostics,
            classes=tuple(projected),
            properties=globals_,
            root_properties=roots,
            property_bindings=tuple(bindings) + roots,
            scoped_semantics=True,
            descriptors=tuple(
                replace(
                    d,
                    property=root_by_iri.get(
                        d.property.iri, registered[d.property.iri]
                    ),
                )
                for d in native.descriptors
            ),
            class_aliases=tuple((k, v) for k, v in sorted(aliases.items()) if v),
            expressions=tuple(
                ExpressionDefinition(k, v) for k, v in sorted(definitions.items())
            ),
            reflexive_properties=reflexive,
        )

    def positive(self, node):
        if node.kind not in {
            ExpressionKind.CLASS,
            ExpressionKind.UNION,
            ExpressionKind.INTERSECTION,
            ExpressionKind.ENUMERATION,
            ExpressionKind.HAS_VALUE,
            ExpressionKind.SOME,
        }:
            self.fail(
                "positive_definition",
                "Definition is outside the positive object-witness profile",
            )
        for child in node.operands:
            self.positive(child)

    def expression_properties(self, node):
        result = {node.property_iri} if node.property_iri else set()
        for child in node.operands:
            result.update(self.expression_properties(child))
        return result

    def references(self, node):
        result = {node.iri} if node.kind == ExpressionKind.CLASS else set()
        for child in node.operands:
            result.update(self.references(child))
        return result

    @staticmethod
    def fail(code, message, iri=None):
        raise OntologyTranslationError((CompilationDiagnostic(code, message, iri),))
