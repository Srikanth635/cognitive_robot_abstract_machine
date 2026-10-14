"""Scoped OWL constraint checks using native EQL membership, quantifiers and counts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date as _date
from datetime import datetime as _datetime
from datetime import time as _time
from decimal import Decimal as _Decimal
from itertools import combinations
from pathlib import Path
from typing import Mapping

from krrood.entity_query_language.factories import (
    count,
    entity,
    exists,
    for_all,
    in_,
    variable,
)
from krrood.entity_query_language.predicate import Symbol
from rdflib import OWL, RDF, RDFS, BNode, Graph, Literal, URIRef

from krrood_ontology.constraints import ConstraintParser, ConstraintTranslationError
from krrood_ontology.expressions import expression_members
from krrood_ontology.identity import IdentityConstraints
from krrood_ontology.model import ClassExpression, ExpressionKind
from krrood_ontology.queries import ModelQueries
from krrood_ontology.runtime import GeneratedModel, LoadedIndividuals
from krrood_ontology.translation import OntologyTranslator
from krrood_ontology.values import value_key as _value_key


@dataclass(frozen=True)
class ValidationCheck:
    subject_iri: str | None
    class_iri: str | None
    axiom_id: str
    kind: str
    status: str
    message: str
    property_iri: str | None = None
    observed_count: int | None = None
    lower_bound: int | None = None


@dataclass(frozen=True)
class ValidationReport:
    checks: tuple[ValidationCheck, ...]
    complete: bool
    identity_policy: str
    scope_size: int
    profile: str = "mapped_fields_explicit_structural_types"
    omissions: tuple[str, ...] = ()

    @property
    def violations(self):
        return tuple(c for c in self.checks if c.status == "violation")

    @property
    def unknown(self):
        return tuple(c for c in self.checks if c.status == "unknown")

    @property
    def unsupported(self):
        return tuple(c for c in self.checks if c.status == "unsupported")

    @property
    def valid(self):
        if self.violations:
            return False
        if self.unknown or self.unsupported:
            return None
        return True

    def to_dict(self):
        result = asdict(self)
        result["valid"] = self.valid
        result["summary"] = {
            s: sum(c.status == s for c in self.checks)
            for s in ("satisfied", "violation", "unknown", "unsupported")
        }
        return result


@dataclass(frozen=True)
class ConstraintDefinition:
    class_iri: str
    axiom_id: str
    expression: ClassExpression | None
    error: str | None = None
    relation: str = "subclass"


@dataclass
class ModelValidator:
    """Validate a declared object registry; never mutate objects or apply OWL inference."""

    model: GeneratedModel
    source: Path
    definition_queries: ModelQueries | None = None
    _graph: Graph | None = field(default=None, init=False, repr=False)
    _constraints: tuple[ConstraintDefinition, ...] = field(
        default=(), init=False, repr=False
    )
    _disjoint: tuple[tuple[str, str], ...] = field(default=(), init=False, repr=False)
    _functional: tuple[str, ...] = field(default=(), init=False, repr=False)
    _schema_errors: tuple[str, ...] = field(default=(), init=False, repr=False)

    def _prepare(self):
        if self._graph is not None:
            return
        graph = Graph().parse(Path(self.source), format="turtle")
        parser = ConstraintParser(graph, self.model)
        definitions = []
        errors = []
        pairs = set()
        for subject, term in sorted(
            graph.subject_objects(RDFS.subClassOf), key=lambda p: (str(p[0]), str(p[1]))
        ):
            if not isinstance(term, BNode):
                continue
            try:
                if str(subject) not in self.model.classes:
                    raise ConstraintTranslationError("Unregistered constraint owner")
                node = parser.parse(term)
                definitions.append(ConstraintDefinition(str(subject), term.n3(), node))
            except ConstraintTranslationError as error:
                definitions.append(
                    ConstraintDefinition(str(subject), term.n3(), None, str(error))
                )
        if self.definition_queries is not None:
            equivalents = set(graph.subject_objects(OWL.equivalentClass))
            equivalents.update((b, a) for a, b in tuple(equivalents))
            for subject, term in sorted(
                equivalents, key=lambda p: (str(p[0]), str(p[1]))
            ):
                if str(subject) not in self.model.classes:
                    continue
                try:
                    definitions.append(
                        ConstraintDefinition(
                            str(subject),
                            "equivalent:" + term.n3(),
                            parser.parse(term),
                            relation="equivalent",
                        )
                    )
                except ConstraintTranslationError as error:
                    definitions.append(
                        ConstraintDefinition(
                            str(subject),
                            "equivalent:" + term.n3(),
                            None,
                            str(error),
                            relation="equivalent",
                        )
                    )
        for left, right in graph.subject_objects(OWL.disjointWith):
            if (
                str(left) not in self.model.classes
                or str(right) not in self.model.classes
            ):
                errors.append(
                    "Anonymous or unregistered disjointWith operands are unsupported"
                )
            else:
                pairs.add(tuple(sorted((str(left), str(right)))))
        for group in graph.subjects(RDF.type, OWL.AllDisjointClasses):
            try:
                heads = tuple(graph.objects(group, OWL.members))
                if len(heads) != 1:
                    raise ConstraintTranslationError(
                        "AllDisjointClasses requires one members list"
                    )
                members = expression_members(graph, heads[0], parser.fail)
                if len(members) < 2 or any(
                    str(v) not in self.model.classes for v in members
                ):
                    raise ConstraintTranslationError(
                        "AllDisjointClasses requires at least two registered named classes"
                    )
                for a, b in combinations(members, 2):
                    pairs.add(tuple(sorted((str(a), str(b)))))
            except ConstraintTranslationError as error:
                errors.append(str(error))
        functional = tuple(
            sorted(str(v) for v in graph.subjects(RDF.type, OWL.FunctionalProperty))
        )
        for iri in functional:
            if iri not in self.model.properties:
                errors.append(f"Unregistered functional property: {iri}")
        self._graph = graph
        self._constraints = tuple(definitions)
        self._disjoint = tuple(sorted(pairs))
        self._functional = functional
        self._schema_errors = tuple(errors)

    def coverage(self):
        """Describe parsing coverage, separately from runtime or OWL conformance."""
        self._prepare()
        return {
            "subclass_constraints": sum(
                c.relation == "subclass" for c in self._constraints
            ),
            "equivalent_conditions": sum(
                c.relation == "equivalent" for c in self._constraints
            ),
            "asymmetric_properties": (
                len(set(self._graph.subjects(RDF.type, OWL.AsymmetricProperty)))
                if self.definition_queries
                else 0
            ),
            "parsed_constraints": sum(
                c.expression is not None for c in self._constraints
            ),
            "unsupported_constraints": [
                {"class_iri": c.class_iri, "axiom_id": c.axiom_id, "message": c.error}
                for c in self._constraints
                if c.error
            ],
            "functional_properties": len(self._functional),
            "disjoint_pairs": len(self._disjoint),
            "schema_errors": list(self._schema_errors),
            "unbound_functional_properties": (
                [
                    iri
                    for iri in self._functional
                    if not any(b.iri == iri for b in self.model.bindings)
                ]
                if self.model.binding_policy == "local-usage"
                else []
            ),
            "complete_owl_conformance": False,
        }

    def validate(
        self,
        scope: Mapping[str, Symbol] | LoadedIndividuals,
        *,
        complete: bool = False,
        identity_policy: str = "owl",
        identity_graph: Graph | None = None,
    ) -> ValidationReport:
        """complete closes mapped values and structural types within this registry only."""
        if type(complete) is not bool:
            raise ValueError("complete must be a boolean")
        if identity_policy not in {"owl", "python"}:
            raise ValueError("identity_policy must be owl or python")
        self._prepare()
        if isinstance(scope, LoadedIndividuals):
            identities = (
                scope.asserted_graph if identity_graph is None else identity_graph
            )
            objects = dict(scope.by_iri)
        else:
            objects = dict(scope)
            identities = identity_graph
        if any(
            not isinstance(k, str) or not isinstance(v, Symbol)
            for k, v in objects.items()
        ):
            raise ValueError("Scope must map string IRIs to KRROOD Symbols")
        if len({id(v) for v in objects.values()}) != len(objects):
            raise ValueError(
                "Scope requires one IRI per Python object; aliases need a separate identity policy"
            )
        session = _ValidationSession(
            self.model,
            objects,
            complete,
            identity_policy,
            self._disjoint,
            definition_queries=self.definition_queries,
        )
        checks = [
            ValidationCheck(None, None, "schema", "schema", "unsupported", e)
            for e in self._schema_errors
        ]
        parser = ConstraintParser(self._graph, self.model)
        for graph in (self._graph, identities):
            if graph is None:
                continue
            if any(graph.triples((None, OWL.sameAs, None))):
                checks.append(
                    ValidationCheck(
                        None,
                        None,
                        "sameAs",
                        "identity",
                        "unsupported",
                        "sameAs normalization is outside this validation profile",
                    )
                )
            try:
                session.distinct_groups.extend(
                    IdentityConstraints.from_graph(graph, parser.fail).groups
                )
            except ConstraintTranslationError as error:
                checks.append(
                    ValidationCheck(
                        None, None, "identity", "identity", "unsupported", str(error)
                    )
                )
        for definition in self._constraints:
            for iri in sorted(session.members(definition.class_iri)):
                if definition.expression is None:
                    checks.append(
                        ValidationCheck(
                            iri,
                            definition.class_iri,
                            definition.axiom_id,
                            "constraint",
                            "unsupported",
                            definition.error,
                        )
                    )
                    continue
                try:
                    status, message, n, lower = session.check(
                        definition.expression, objects[iri]
                    )
                    checks.append(
                        ValidationCheck(
                            iri,
                            definition.class_iri,
                            definition.axiom_id,
                            definition.expression.kind.value,
                            status,
                            message,
                            definition.expression.property_iri,
                            n,
                            lower,
                        )
                    )
                except ConstraintTranslationError as error:
                    checks.append(
                        ValidationCheck(
                            iri,
                            definition.class_iri,
                            definition.axiom_id,
                            "constraint",
                            "unsupported",
                            str(error),
                        )
                    )
        # Functional axioms are checked on actual projected field associations only.
        for prop in self._functional:
            if prop not in self.model.properties:
                continue
            for iri, obj in sorted(objects.items()):
                mapping = self.model.mapping_for(prop, _class_iri(self.model, obj))
                if mapping is None:
                    continue
                node = ClassExpression(
                    ExpressionKind.MAX, property_iri=prop, cardinality=1
                )
                try:
                    status, message, n, lower = session.check(node, obj)
                    checks.append(
                        ValidationCheck(
                            iri,
                            None,
                            prop,
                            "functional",
                            status,
                            message,
                            prop,
                            n,
                            lower,
                        )
                    )
                except ConstraintTranslationError as error:
                    checks.append(
                        ValidationCheck(
                            iri,
                            None,
                            prop,
                            "functional",
                            "unsupported",
                            str(error),
                            prop,
                        )
                    )
        if self.definition_queries is not None:
            for term in sorted(
                self._graph.subjects(RDF.type, OWL.AsymmetricProperty), key=str
            ):
                prop_iri = str(term)
                for iri, obj in sorted(objects.items()):
                    mapping = self.model.mapping_for(
                        prop_iri, _class_iri(self.model, obj)
                    )
                    if mapping is None:
                        continue
                    values = tuple(getattr(obj, mapping.name))
                    conflict = False
                    outside = False
                    for target in values:
                        if target is obj:
                            conflict = True
                        elif id(target) not in session.by_identity:
                            outside = True
                        else:
                            reverse = self.model.mapping_for(
                                prop_iri, _class_iri(self.model, target)
                            )
                            if reverse and any(
                                v is obj for v in getattr(target, reverse.name)
                            ):
                                conflict = True
                    status = (
                        "violation"
                        if conflict
                        else ("unknown" if outside or not complete else "satisfied")
                    )
                    checks.append(
                        ValidationCheck(
                            iri,
                            None,
                            prop_iri,
                            "asymmetric",
                            status,
                            (
                                "Observed reverse edge or self-loop"
                                if conflict
                                else "No observed conflict within the declared scope"
                            ),
                            prop_iri,
                        )
                    )
            for prop in self.model.properties.values():
                if not prop.domain_iri:
                    continue
                for iri, obj in sorted(objects.items()):
                    mapping = self.model.mapping_for(
                        prop.iri, _class_iri(self.model, obj)
                    )
                    if mapping is None or not tuple(getattr(obj, mapping.name)):
                        continue
                    truth = session.truth(
                        ClassExpression(ExpressionKind.CLASS, iri=prop.domain_iri), obj
                    )
                    status = (
                        "satisfied"
                        if truth is True
                        else ("violation" if truth is False else "unknown")
                    )
                    checks.append(
                        ValidationCheck(
                            iri,
                            None,
                            prop.iri,
                            "domain",
                            status,
                            "Declared domain membership in the scoped profile",
                            prop.iri,
                        )
                    )
        for prop in self.model.properties.values():
            if not prop.range_iri:
                continue
            for iri, obj in sorted(objects.items()):
                if (
                    self.model.mapping_for(prop.iri, _class_iri(self.model, obj))
                    is None
                ):
                    continue
                try:
                    node = ClassExpression(
                        ExpressionKind.ALL,
                        property_iri=prop.iri,
                        operands=(parser.parse(URIRef(prop.range_iri)),),
                    )
                    status, message, n, lower = session.check(node, obj)
                    checks.append(
                        ValidationCheck(
                            iri,
                            None,
                            prop.iri,
                            "range",
                            status,
                            message,
                            prop.iri,
                            n,
                            lower,
                        )
                    )
                except ConstraintTranslationError as error:
                    checks.append(
                        ValidationCheck(
                            iri,
                            None,
                            prop.iri,
                            "range",
                            "unsupported",
                            str(error),
                            prop.iri,
                        )
                    )
        for a, b in self._disjoint:
            left = session.members(a)
            right = session.members(b)
            for iri in sorted(left | right):
                status = (
                    "violation"
                    if iri in left & right
                    else ("satisfied" if complete else "unknown")
                )
                checks.append(
                    ValidationCheck(
                        iri,
                        a,
                        f"{a} disjointWith {b}",
                        "disjoint",
                        status,
                        (
                            "Conflicting memberships in the declared validation profile"
                            if status == "violation"
                            else (
                                "No observed conflict; additional membership remains open"
                                if status == "unknown"
                                else "No conflict in the complete structural type scope"
                            )
                        ),
                    )
                )
        omissions = tuple(
            f"Functional property has no projected field: {iri}"
            for iri in self.coverage()["unbound_functional_properties"]
        )
        if self.definition_queries is None and any(
            self._graph.triples((None, OWL.equivalentClass, None))
        ):
            omissions += (
                "Equivalent-class definitions are not used for classification or necessary-condition checking in this profile",
            )
        return ValidationReport(
            tuple(checks),
            complete,
            identity_policy,
            len(objects),
            profile=(
                "mapped_fields_positive_definitions"
                if self.definition_queries
                else "mapped_fields_explicit_structural_types"
            ),
            omissions=omissions,
        )


def _class_iri(model, obj):
    iri = getattr(type(obj), "__ontology_iri__", None)
    if iri not in model.classes or type(obj) is not model.classes[iri]:
        raise ValueError("Scope objects must have registered concrete generated types")
    return iri


def _native_count(values):
    return entity(
        count(variable(object, domain=tuple(values)), distinct=True)
    ).tolist()[0]


@dataclass
class _ValidationSession:
    model: GeneratedModel
    objects: dict[str, Symbol]
    complete: bool
    identity_policy: str
    disjoint: tuple[tuple[str, str], ...]
    definition_queries: ModelQueries | None = None
    distinct_groups: list[frozenset[str]] = field(default_factory=list)
    _members: dict[str, frozenset[str]] = field(default_factory=dict)

    def __post_init__(self):
        self.by_identity = {id(v): k for k, v in self.objects.items()}
        for obj in self.objects.values():
            _class_iri(self.model, obj)
        self.loaded = LoadedIndividuals(self.objects, {}, {}, "", Graph())
        self.queries = self.definition_queries or ModelQueries(self.model, {})

    def members(self, iri):
        if iri == str(OWL.Thing):
            return frozenset(self.objects)
        if iri == str(OWL.Nothing):
            return frozenset()
        if iri not in self.model.classes:
            raise ConstraintTranslationError(f"Unknown structural class: {iri}")
        if iri not in self._members:
            self._members[iri] = frozenset(
                self.by_identity[id(v)] for v in self.queries.match(iri, self.loaded)
            )
        return self._members[iri]

    def values(self, node, obj):
        mapping = self.model.mapping_for(node.property_iri, _class_iri(self.model, obj))
        if mapping is None:
            raise ConstraintTranslationError(
                "Constraint has no applicable projected field binding"
            )
        values = tuple(getattr(obj, mapping.name))
        if mapping.kind == "datatype" and any(
            type(v) not in {str, bool, int, float, _Decimal, _date, _datetime, _time}
            for v in values
        ):
            raise ConstraintTranslationError(
                "Field contains values outside the supported converted RDF literal types"
            )
        return values, mapping.kind

    def truth(self, node, obj):
        if node.kind == ExpressionKind.CLASS:
            if not isinstance(obj, Symbol):
                return False
            iri = self.by_identity.get(id(obj))
            if iri is None:
                return None
            if iri in self.members(node.iri):
                return True
            for left, right in self.disjoint:
                for expected, other in ((left, right), (right, left)):
                    if (
                        node.iri in self.model.classes
                        and issubclass(
                            self.model.classes[node.iri], self.model.classes[expected]
                        )
                        and iri in self.members(other)
                    ):
                        return False
            return False if self.complete else None
        if node.kind == ExpressionKind.DATATYPE:
            python_type = OntologyTranslator()._datatype_type(URIRef(node.iri))
            types = {
                "str": str,
                "bool": bool,
                "int": int,
                "float": float,
                "_Decimal": _Decimal,
                "_date": _date,
                "_datetime": _datetime,
                "_time": _time,
            }
            return type(obj) is types[python_type]
        if node.kind == ExpressionKind.ENUMERATION:
            known = [self.objects[v] for v in node.individuals if v in self.objects]
            if any(obj is v for v in known):
                return True
            if len(known) != len(node.individuals):
                return None
            return False if self.complete and self.identity_policy == "python" else None
        if node.kind in (ExpressionKind.UNION, ExpressionKind.INTERSECTION):
            results = [self.truth(v, obj) for v in node.operands]
            if node.kind == ExpressionKind.UNION:
                return True if True in results else None if None in results else False
            return False if False in results else None if None in results else True
        if not isinstance(obj, Symbol) or id(obj) not in self.by_identity:
            return None
        status, _, _, _ = self.check(node, obj)
        return {"satisfied": True, "violation": False, "unknown": None}[status]

    def quantify(self, owner, values, indices, universal=False):
        # The pinned native ForAll raises on an empty domain; use the logical boundary value.
        if not values:
            return universal
        subject = variable(Symbol, domain=(owner,))
        # Positive ordinal domains avoid the native Exists treating numeric zero as a false binding.
        index = variable(int, domain=range(1, len(values) + 1))
        quantifier = for_all if universal else exists
        return bool(
            entity(subject)
            .where(quantifier(index, in_(index, tuple(i + 1 for i in indices))))
            .tolist()
        )

    def bounds(self, values, kind):
        if kind == "datatype":
            n = _native_count(_value_key(v) for v in values)
            return n, n
        n = _native_count(values)
        if self.identity_policy == "python":
            return n, n
        iris = {self.by_identity[id(v)] for v in values}
        lower = 1 if n else 0
        for group in self.distinct_groups:
            lower = max(lower, len(iris & group))
        return lower, n

    def check(self, node, obj):
        if node.kind in (
            ExpressionKind.CLASS,
            ExpressionKind.DATATYPE,
            ExpressionKind.UNION,
            ExpressionKind.INTERSECTION,
            ExpressionKind.ENUMERATION,
        ):
            truth = self.truth(node, obj)
            return (
                (
                    "satisfied"
                    if truth is True
                    else "violation" if truth is False else "unknown"
                ),
                "Superclass expression checked against scoped structural types",
                None,
                None,
            )
        values, kind = self.values(node, obj)
        outside = [
            v
            for v in values
            if kind == "object"
            and (not isinstance(v, Symbol) or id(v) not in self.by_identity)
        ]
        inside = tuple(v for v in values if not any(v is x for x in outside))
        if node.kind == ExpressionKind.HAS_VALUE:
            if node.literal:
                expected = Literal(
                    node.literal.lexical, datatype=URIRef(node.literal.datatype)
                ).toPython()
                hit = any(_value_key(v) == _value_key(expected) for v in inside)
            else:
                if node.iri not in self.objects:
                    return (
                        "unknown",
                        "Required named individual is absent from the supplied scope",
                        None,
                        None,
                    )
                hit = any(v is self.objects[node.iri] for v in inside)
            if hit:
                return "satisfied", "Required value is present", None, None
            # Under OWL identity, another reference may denote the required individual.
            ambiguous = (
                kind == "object" and self.identity_policy == "owl" and bool(inside)
            )
            return (
                (
                    "violation"
                    if self.complete and not outside and not ambiguous
                    else "unknown"
                ),
                (
                    "Required value is absent in the complete scope"
                    if self.complete and not outside and not ambiguous
                    else "Required value is not established; values, identity or scope remain open"
                ),
                None,
                None,
            )
        truth = [
            self.truth(node.operands[0], v) if node.operands else True for v in inside
        ]
        yes = [i for i, t in enumerate(truth) if t is True]
        unknown = any(t is None for t in truth) or bool(outside)
        if node.kind == ExpressionKind.SOME:
            hit = self.quantify(obj, inside, yes)
            return (
                (
                    "satisfied"
                    if hit
                    else "violation" if self.complete and not unknown else "unknown"
                ),
                (
                    "Known existential witness found"
                    if hit
                    else (
                        "No qualifying witness in the complete scope"
                        if self.complete and not unknown
                        else "No known witness; values, types or scope remain open"
                    )
                ),
                None,
                None,
            )
        if node.kind == ExpressionKind.ALL:
            if False in truth:
                return (
                    "violation",
                    "A known filler fails the required class/datatype condition",
                    None,
                    None,
                )
            all_known = self.quantify(obj, inside, yes, universal=True)
            return (
                (
                    "satisfied"
                    if self.complete and not unknown and all_known
                    else "unknown"
                ),
                (
                    "All fillers qualify in the complete scope"
                    if self.complete and not unknown and all_known
                    else "Additional fillers, types or scope remain open"
                ),
                None,
                None,
            )
        qualifying = tuple(inside[i] for i in yes)
        lower, upper = self.bounds(qualifying, kind)
        n = node.cardinality
        if node.kind == ExpressionKind.MIN:
            status = (
                "satisfied"
                if lower >= n
                else (
                    "violation"
                    if self.complete and not unknown and upper < n
                    else "unknown"
                )
            )
        elif node.kind == ExpressionKind.MAX:
            status = (
                "violation"
                if lower > n
                else (
                    "satisfied"
                    if self.complete and not unknown and upper <= n
                    else "unknown"
                )
            )
        elif node.kind == ExpressionKind.EXACT:
            status = (
                "violation"
                if lower > n or (self.complete and not unknown and upper < n)
                else (
                    "satisfied"
                    if self.complete and not unknown and lower == upper == n
                    else "unknown"
                )
            )
        else:
            raise ConstraintTranslationError(
                f"Unsupported constraint kind: {node.kind}"
            )
        message = f"Known qualifying fillers: {upper}; established distinct lower bound: {lower}; required {node.kind.value}: {n}"
        if outside:
            message += "; referenced objects lie outside the supplied scope"
        elif unknown:
            message += "; filler membership remains unknown"
        elif status == "unknown":
            message += "; values or identity remain open"
        return status, message, upper, lower
