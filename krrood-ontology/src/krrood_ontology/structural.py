"""Generic structural policies; native KRROOD descriptors implement field behavior."""

from dataclasses import dataclass, replace
from graphlib import CycleError, TopologicalSorter

from rdflib import OWL, RDF, RDFS, BNode, URIRef

from krrood_ontology.model import (
    ClassDefinition,
    CompilationDiagnostic,
    ModelDefinition,
    PropertyDefinition,
    PropertyKind,
)
from krrood_ontology.translation import OntologyTranslator


@dataclass
class StructuralTranslator(OntologyTranslator):
    """Preserve OWL axioms while projecting tested named inheritance and local fields."""

    def translate(self, document, report, structural_only=True):
        graph = document.graph
        names = {iri: self._name(iri, True) for iri in report.classes}
        if not names:
            self._fail("no_classes", "No explicitly declared named classes were found")
        self._check_class_names(names)
        parents = {}
        for subject in graph.subjects(RDFS.subClassOf):
            if not isinstance(subject, URIRef) or str(subject) not in names:
                self._fail(
                    "undeclared_class",
                    "Subclass subjects require declared named classes",
                    str(subject),
                )
        for iri in names:
            if any(
                not isinstance(v, (URIRef, BNode))
                for v in graph.objects(URIRef(iri), RDFS.subClassOf)
            ):
                self._fail(
                    "unsupported_parent",
                    "Superclass must be a class IRI or anonymous expression",
                    iri,
                )
            edges = tuple(
                sorted(
                    str(v)
                    for v in graph.objects(URIRef(iri), RDFS.subClassOf)
                    if isinstance(v, URIRef) and v != OWL.Thing
                )
            )
            if any(v not in names for v in edges):
                self._fail(
                    "unsupported_parent",
                    "Named superclass requires a declared class",
                    iri,
                )
            parents[iri] = edges
        try:
            order = tuple(TopologicalSorter(parents).static_order())
        except CycleError:
            self._fail(
                "inheritance_cycle",
                "Cyclic subclass declarations cannot become Python inheritance",
            )
        rank = {iri: i for i, iri in enumerate(order)}
        # Python itself checks C3. Plain temporary types avoid KRROOD class/instance registration.
        probes = {}
        bases_by_iri = {}
        for iri in order:
            edges = parents[iri]
            reduced = [
                v
                for v in edges
                if not any(
                    v != other and issubclass(probes[other], probes[v])
                    for other in edges
                )
            ]
            bases = tuple(sorted(reduced, key=rank.get, reverse=True))
            try:
                probes[iri] = type(
                    names[iri], tuple(probes[v] for v in bases) or (object,), {}
                )
            except TypeError as error:
                self._fail(
                    "inheritance_mro",
                    f"Python cannot represent these named parents: {error}",
                    iri,
                )
            bases_by_iri[iri] = bases
        if set(report.object_properties) & set(report.datatype_properties):
            self._fail(
                "property_kind_conflict",
                "Property has both object and datatype declarations",
            )
        declared = set(
            report.object_properties
            + report.datatype_properties
            + report.annotation_properties
        )
        for term in (
            set(graph.subjects(RDFS.domain))
            | set(graph.subjects(RDFS.range))
            | set(graph.subjects(RDF.type, RDF.Property))
        ):
            if str(term) not in declared:
                self._fail(
                    "undeclared_property",
                    "Property signatures require an explicit declaration",
                    str(term),
                )
        # Preserve source blank-node identity without whole-graph canonicalization.
        # The compiler serializes this same graph into source/closure.ttl.
        axiom_graph = graph
        records = []
        usage = {}
        for iri in sorted(names):
            for predicate in (RDFS.subClassOf, OWL.equivalentClass):
                for value in sorted(
                    axiom_graph.objects(URIRef(iri), predicate), key=str
                ):
                    if not isinstance(value, BNode):
                        continue
                    pending = [value]
                    visited = set()
                    triples = []
                    while pending:
                        node = pending.pop()
                        if node in visited:
                            continue
                        visited.add(node)
                        for pred, obj in axiom_graph.predicate_objects(node):
                            triples.append((node.n3(), pred.n3(), obj.n3()))
                            if isinstance(obj, BNode):
                                pending.append(obj)
                            if pred == OWL.onProperty:
                                if (
                                    not isinstance(obj, URIRef)
                                    or str(obj) not in declared
                                ):
                                    self._fail(
                                        "restriction_property",
                                        "Restrictions require declared named properties",
                                        str(obj),
                                    )
                                usage.setdefault(str(obj), set()).add(iri)
                    records.append(
                        {
                            "class_iri": iri,
                            "predicate_iri": str(predicate),
                            "root": value.n3(),
                            "triples": tuple(sorted(triples)),
                        }
                    )
        for category in (OWL.AllDifferent, OWL.AllDisjointClasses):
            for root in sorted(graph.subjects(RDF.type, category), key=str):
                pending = [root]
                seen = set()
                triples = []
                while pending:
                    node = pending.pop()
                    if node in seen:
                        continue
                    seen.add(node)
                    for predicate, value in graph.predicate_objects(node):
                        triples.append((node.n3(), predicate.n3(), value.n3()))
                        if isinstance(value, BNode):
                            pending.append(value)
                records.append(
                    {
                        "class_iri": None,
                        "predicate_iri": str(RDF.type),
                        "root": root.n3(),
                        "triples": tuple(sorted(triples)),
                    }
                )
        properties = []
        candidates = {}
        field_names = {}
        for kind, iris in (
            (PropertyKind.OBJECT, report.object_properties),
            (PropertyKind.DATATYPE, report.datatype_properties),
        ):
            for iri in iris:
                term = URIRef(iri)
                domains = tuple(graph.objects(term, RDFS.domain))
                ranges = tuple(graph.objects(term, RDFS.range))
                if (
                    len(domains) > 1
                    or len(ranges) > 1
                    or any(not isinstance(v, URIRef) for v in domains + ranges)
                ):
                    self._fail(
                        "property_signature",
                        "Multiple or anonymous global signatures are retained in source but require a future binding policy",
                        iri,
                    )
                domain = str(domains[0]) if domains and domains[0] != OWL.Thing else ""
                range_iri = str(ranges[0]) if ranges and ranges[0] != OWL.Thing else ""
                if domain and domain not in names:
                    self._fail(
                        "property_domain", "Domain must be a declared named class", iri
                    )
                if kind is PropertyKind.OBJECT:
                    if range_iri and range_iri not in names:
                        self._fail(
                            "property_range",
                            "Object range must be a declared named class",
                            iri,
                        )
                    range_type = names[range_iri] if range_iri else "_KrroodSymbol"
                elif range_iri:
                    range_type = self._datatype_type(URIRef(range_iri))
                else:
                    range_type = "_Any"
                name = self._name(iri, False)
                if name in field_names and field_names[name] != iri:
                    self._fail(
                        "field_collision",
                        f"Property name {name!r} also maps {field_names[name]}",
                        iri,
                    )
                field_names[name] = iri
                properties.append(
                    PropertyDefinition(
                        iri,
                        name,
                        domain,
                        names.get(domain, ""),
                        range_iri,
                        range_type,
                        kind,
                    )
                )
                candidates[iri] = {domain} if domain else usage.get(iri, set())
        classes = []
        bindings = []
        inherited = {}
        for iri in order:
            available = {}
            for parent in parents[iri]:
                available.update(inherited[parent])
            owned = []
            for prop in properties:
                if iri in candidates[prop.iri] and prop.iri not in available:
                    binding = replace(prop, domain_iri=iri, domain_name=names[iri])
                    owned.append(binding)
                    bindings.append(binding)
                    available[prop.iri] = binding
            inherited[iri] = available
            bases = tuple(names[v] for v in bases_by_iri[iri])
            classes.append(
                ClassDefinition(
                    iri,
                    names[iri],
                    bases[0] if len(bases) == 1 else None,
                    tuple(owned),
                    bases,
                    parents[iri],
                )
            )
        diagnostics = self._semantic_diagnostics(document, report, True)
        diagnostics += (
            CompilationDiagnostic(
                "local_usage_binding",
                "Field ownership follows declared domains or local axiom usage; it does not infer OWL signatures. Unbound properties remain registered without Python fields.",
            ),
        )
        return ModelDefinition(
            tuple(classes),
            tuple(properties),
            diagnostics,
            property_bindings=tuple(bindings),
            binding_policy="local-usage",
            schema_axioms=tuple(records),
        )
