"""Build scoped positive class-expression queries with native KRROOD EQL."""

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.factories import (
    and_,
    contains,
    entity,
    exists,
    flat_variable,
    in_,
    issubclass_,
    or_,
    type_,
    variable,
)
from krrood.entity_query_language.predicate import Symbol, symbolic_function
from krrood.entity_query_language.query.query import Query
from rdflib import OWL, Graph, Literal, URIRef

from krrood_ontology.model import ClassExpression, ExpressionKind
from krrood_ontology.runtime import GeneratedModel, LoadedIndividuals


@symbolic_function
def field_values(owner: Any, domain: type[Symbol], name: str) -> tuple:
    """Expose a mapped field safely across a heterogeneous query scope."""
    return tuple(getattr(owner, name)) if isinstance(owner, domain) else ()


@symbolic_function
def mapped_values(owner: Any, model: GeneratedModel, iri: str) -> tuple:
    """Resolve a projected field through the generated mapping registry."""
    class_iri = getattr(type(owner), "__ontology_iri__", None)
    if class_iri not in model.classes or type(owner) is not model.classes[class_iri]:
        return ()
    mapping = model.mapping_for(iri, class_iri)
    return tuple(getattr(owner, mapping.name)) if mapping else ()


@symbolic_function
def literal_present(values: tuple, lexical: str, datatype: str) -> bool:
    """Use RDFLib value equality rather than Python's boolean-number equality."""
    expected = Literal(lexical, datatype=URIRef(datatype))
    return any(Literal(value).eq(expected) is True for value in values)


def scoped_registry(
    scope: LoadedIndividuals | Mapping[str, Symbol],
) -> LoadedIndividuals:
    """Adapt an explicit registry without merging or creating identities."""
    if isinstance(scope, LoadedIndividuals):
        return scope
    objects = dict(scope)
    if any(
        not isinstance(k, str) or not isinstance(v, Symbol) for k, v in objects.items()
    ):
        raise ValueError("Scope must map string IRIs to KRROOD Symbols")
    if len({id(v) for v in objects.values()}) != len(objects):
        raise ValueError("Scope requires one IRI per Python object")
    return LoadedIndividuals(objects, {}, {}, "", Graph())


@dataclass
class ModelQueries:
    """Match explicit types and known positive expression witnesses in a registry."""

    model: GeneratedModel
    definitions: dict[str, ClassExpression]
    class_aliases: dict[str, tuple[str, ...]] = field(default_factory=dict)
    partial_individuals: bool = False

    def query(
        self,
        class_iri: str,
        loaded: LoadedIndividuals,
        candidates: Iterable[Symbol] | None = None,
    ) -> Query:
        """Return an EQL query over the caller's explicit candidate domain."""
        loaded = scoped_registry(loaded)
        scope = tuple(loaded.by_iri.values())
        selected = scope if candidates is None else tuple(candidates)
        if any(not any(item is v for v in scope) for item in selected):
            raise ValueError(
                "Candidates must belong to the supplied individual registry"
            )
        if class_iri not in self.model.classes:
            raise KeyError(class_iri)
        subject = variable(Symbol, domain=selected)
        condition = self._named(class_iri, subject, loaded, scope)
        return entity(subject).where(condition)

    def match(
        self,
        class_iri: str,
        loaded: LoadedIndividuals,
        candidates: Iterable[Symbol] | None = None,
    ) -> list[Symbol]:
        """Evaluate a scoped query without altering types, objects or graph facts."""
        return list(self.query(class_iri, loaded, candidates).evaluate())

    def _named(
        self,
        iri: str,
        subject: Any,
        loaded: LoadedIndividuals,
        scope: tuple[Symbol, ...],
    ) -> SymbolicExpression:
        if iri == str(OWL.Thing):
            return in_(subject, scope)
        if iri == str(OWL.Nothing):
            return in_(subject, ())
        if not self.partial_individuals:
            explicit = issubclass_(type_(subject), self.model.classes[iri])
            return (
                or_(
                    explicit,
                    self._condition(self.definitions[iri], subject, loaded, scope),
                )
                if iri in self.definitions
                else explicit
            )
        evidence = [issubclass_(type_(subject), self.model.classes[iri])]
        for candidate_iri, concrete in self.model.classes.items():
            if not issubclass(concrete, self.model.classes[iri]):
                continue
            evidence.extend(
                issubclass_(type_(subject), self.model.classes[v])
                for v in self.class_aliases.get(candidate_iri, ())
            )
            if candidate_iri in self.definitions:
                evidence.append(
                    self._condition(
                        self.definitions[candidate_iri], subject, loaded, scope
                    )
                )
        return or_(*evidence)

    def _condition(
        self,
        expression: ClassExpression,
        subject: Any,
        loaded: LoadedIndividuals,
        scope: tuple[Symbol, ...],
    ) -> SymbolicExpression:
        kind = expression.kind
        if kind is ExpressionKind.CLASS:
            return self._named(expression.iri, subject, loaded, scope)
        if kind in (ExpressionKind.INTERSECTION, ExpressionKind.UNION):
            operator = and_ if kind is ExpressionKind.INTERSECTION else or_
            return operator(
                *(
                    self._condition(child, subject, loaded, scope)
                    for child in expression.operands
                )
            )
        if kind is ExpressionKind.ENUMERATION:
            return in_(
                subject,
                tuple(
                    self._individual(iri, loaded)
                    for iri in expression.individuals
                    if not self.partial_individuals or iri in loaded.by_iri
                ),
            )
        mapping = self.model.properties[expression.property_iri]
        values = (
            mapped_values(subject, self.model, mapping.iri)
            if self.model.binding_policy == "local-usage"
            else field_values(
                subject, self.model.classes[mapping.domain_iri], mapping.name
            )
        )
        if kind is ExpressionKind.HAS_VALUE:
            if self.partial_individuals and expression.literal:
                return literal_present(
                    values, expression.literal.lexical, expression.literal.datatype
                )
            if (
                self.partial_individuals
                and expression.iri
                and expression.iri not in loaded.by_iri
            ):
                return in_(subject, ())
            value = (
                Literal(
                    expression.literal.lexical,
                    datatype=URIRef(expression.literal.datatype),
                ).toPython()
                if expression.literal
                else self._individual(expression.iri, loaded)
            )
            return contains(values, value)
        if kind is ExpressionKind.SOME:
            filler = flat_variable(values)
            return exists(
                filler,
                and_(
                    in_(filler, scope),
                    self._condition(expression.operands[0], filler, loaded, scope),
                ),
            )
        raise ValueError(f"Unsupported expression kind {kind}")

    @staticmethod
    def _individual(iri: str, loaded: LoadedIndividuals) -> Symbol:
        if iri not in loaded.by_iri:
            raise ValueError(
                f"Expression individual is absent from the supplied registry: {iri}"
            )
        return loaded[iri]
