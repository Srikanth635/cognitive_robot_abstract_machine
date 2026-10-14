"""Scoped helpers using native descriptor relations, without a second closure engine."""

from dataclasses import dataclass

from krrood.ontomatic.property_descriptor.property_descriptor_relation import (
    PropertyDescriptorRelation,
)
from krrood.symbol_graph.symbol_graph import SymbolGraph

from krrood_ontology.queries import scoped_registry
from krrood_ontology.runtime import GeneratedModel


@dataclass(frozen=True)
class RelationFact:
    subject_iri: str
    property_iri: str
    object_iri: str
    inferred: bool


@dataclass(frozen=True)
class ReflexivityResult:
    added: int
    skipped: tuple[tuple[str, str], ...] = ()


@dataclass
class ModelRelations:
    model: GeneratedModel
    descriptors: dict
    reflexive_properties: tuple[str, ...] = ()

    def _objects(self, scope):
        objects = scoped_registry(scope).by_iri
        for obj in objects.values():
            iri = getattr(type(obj), "__ontology_iri__", None)
            if (
                iri not in self.model.classes
                or type(obj) is not self.model.classes[iri]
            ):
                raise ValueError(
                    "Scope objects must have registered concrete generated types"
                )
        return objects

    def initialize_reflexivity(self, scope):
        """Seed eligible self-edges explicitly; native KRROOD applies implications."""
        objects = self._objects(scope)
        added = 0
        skipped = []
        for iri, obj in sorted(objects.items()):
            for prop_iri in self.reflexive_properties:
                prop = self.model.properties[prop_iri]
                eligible = all(
                    not constraint or isinstance(obj, self.model.classes[constraint])
                    for constraint in (prop.domain_iri, prop.range_iri)
                )
                mapping = self.model.mapping_for(prop_iri, type(obj).__ontology_iri__)
                if not eligible or mapping is None:
                    skipped.append((iri, prop_iri))
                    continue
                if any(v is obj for v in getattr(obj, mapping.name)):
                    continue
                descriptor = self.descriptors[prop_iri]
                PropertyDescriptorRelation(
                    obj, obj, descriptor.wrapped_field, inferred=True
                ).update_source_and_add_to_graph_and_apply_implications()
                added += 1
        return ReflexivityResult(added, tuple(skipped))

    def snapshot(self, scope):
        """Read scoped native edge flags; these describe first insertion, not proof history."""
        objects = self._objects(scope)
        identities = {id(v): k for k, v in objects.items()}
        facts = []
        for iri, obj in sorted(objects.items()):
            for edge in SymbolGraph().get_outgoing_relations(obj):
                if not isinstance(edge, PropertyDescriptorRelation):
                    continue
                prop_iri = getattr(
                    edge.property_descriptor_cls, "__ontology_iri__", None
                )
                target = identities.get(id(edge.target.instance))
                if target is not None and prop_iri in self.model.properties:
                    facts.append(RelationFact(iri, prop_iri, target, edge.inferred))
        return tuple(
            sorted(facts, key=lambda f: (f.subject_iri, f.property_iri, f.object_iri))
        )
