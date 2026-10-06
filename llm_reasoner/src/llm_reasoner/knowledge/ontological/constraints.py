"""
A declarative constraint engine over the object model.

WHY DECLARATIONS AND NOT PREDICATES
-----------------------------------
The thirty acceptance checks that guard hypothesis generation could each be
written as a function. They would then be thirty places where the metadata and
the constraint can quietly disagree: ``min_count=1`` says a phase needs a
terminating event, and a hand-written check says the same thing again, and one
of them gets edited.

So the checks are DATA. A handful of constraint kinds are implemented here once,
generically, with no knowledge of any domain; a microworld declares which of
them apply to what. Required-content checks are not even declared -- they are
derived from ``min_count``, which is already stated 171 times in the model, so
that half of the suite cannot drift by construction.

CLOSED-WORLD SEMANTICS
----------------------
Every check reads absence as falsity. That is deliberate and it is the whole
reason this layer exists rather than an OWL reasoner: under the open-world
assumption a phase with no terminating event is not an error, merely
underspecified, which is exactly the failure mode of an under-generating model.

RELATION TO EQL
---------------
Each constraint kind corresponds to a query shape EQL can express, and the
engine walks the same object graph EQL would. Evaluation is done directly here
rather than through EQL for now, so that the suite has no query-planner in its
failure path while it is the thing establishing whether the model is sound.
Wiring the kinds to EQL queries later is mechanical, because the kinds were
chosen to be query shapes rather than arbitrary code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any, Callable, Iterable, Optional, Sequence

from llm_reasoner.exceptions import LLMReasonerError
from llm_reasoner.knowledge.ontological.paths import (
    FieldPathResolver,
    PathReference,
)

from .decorators import (
    REGISTRY,
    ConceptReference,
    OntologyRegistry,
    property_meta,
)

Severity = str  # "error" | "warning"


@dataclass
class ConstraintPathError(LLMReasonerError):
    """
    A constraint names a path that its subject type cannot follow.

    Reported when the suite is built, because a path that resolves to nothing reads as
    an absent value and therefore never reports a violation.
    """

    constraint_id: str
    """
    Identifier of the constraint that declares the path.
    """

    path: str
    """
    The declared path, in full.
    """

    unknown_step: str
    """
    The first step no candidate type declares as a field.
    """

    subject_type: type
    """
    The type the constraint applies to.
    """

    def __str__(self) -> str:
        return (
            f"constraint {self.constraint_id}: '{self.path}' has no field "
            f"'{self.unknown_step}' reachable from {self.subject_type.__name__}"
        )


@dataclass(frozen=True)
class ConstraintViolation:
    """
    One failed check, naming the fragment responsible.
    """

    check: str
    severity: Severity
    subject: str
    message: str

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"[{self.check}] {self.severity.upper()} {self.subject}: {self.message}"


# ---------------------------------------------------------------------------
# Path expressions
# ---------------------------------------------------------------------------


def resolve(obj: Any, path: str) -> list[Any]:
    """
    Follow a dotted path, flattening collections at every step.

    ``"from_phase.terminated_by"`` yields the terminating events of the source phase. A
    missing or empty link yields nothing rather than raising, because under closed-world
    reading an absent value is simply no value.
    """
    current = [obj]
    for step in path.split("."):
        nxt: list[Any] = []
        for item in current:
            if item is None:
                continue
            value = getattr(item, step, None)
            if value is None:
                continue
            if isinstance(value, (list, tuple, set, frozenset)):
                nxt.extend(v for v in value if v is not None)
            else:
                nxt.append(value)
        current = nxt
    return current


def name_of(obj: Any) -> str:
    """
    A readable handle for a fragment.
    """
    label = getattr(obj, "label", None)
    if label:
        return str(label)
    local = getattr(obj, "local_id", None)
    if local:
        return f"{type(obj).__name__}<{local}>"
    return type(obj).__name__


# ---------------------------------------------------------------------------
# Constraint kinds
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Constraint(ABC):
    """
    Base for every declarable check.
    """

    id: str
    applies_to: type
    rationale: str
    severity: Severity = "error"
    when: Optional[Callable[[Any], bool]] = None
    """
    Guard: the constraint is skipped for instances it returns False for.
    """

    def check(
        self, subject: Any, context: "EvaluationContext"
    ) -> Iterable[ConstraintViolation]:
        raise NotImplementedError

    @abstractmethod
    def referenced_paths(self) -> tuple[PathReference, ...]:
        """
        Return every field path this constraint follows.

        Declared so that the suite can check each path against the fields of the type the
        constraint applies to.
        """

    def _violation(self, obj: Any, message: str) -> ConstraintViolation:
        return ConstraintViolation(self.id, self.severity, name_of(obj), message)


@dataclass(frozen=True)
class Inclusion(Constraint):
    """
    Every value at ``source`` must also appear at ``target``.

    The workhorse for coherence between two ways of saying the same thing -- a
    transition's gate and the terminating events of the phase it leaves.
    """

    source: str = ""
    target: str = ""
    note: str = ""

    def referenced_paths(self) -> tuple[PathReference, ...]:
        """
        Return every field path this constraint follows.
        """
        return (PathReference(self.source), PathReference(self.target))

    def check(self, obj, ctx):
        allowed = resolve(obj, self.target)
        for value in resolve(obj, self.source):
            if not any(value is a for a in allowed):
                yield self._violation(
                    obj,
                    f"{name_of(value)} at '{self.source}' is not among "
                    f"'{self.target}'{(' -- ' + self.note) if self.note else ''}",
                )


@dataclass(frozen=True)
class Disjoint(Constraint):
    """
    Two paths must not share a value.
    """

    left: str = ""
    right: str = ""
    note: str = ""

    def referenced_paths(self) -> tuple[PathReference, ...]:
        """
        Return every field path this constraint follows.
        """
        return (PathReference(self.left), PathReference(self.right))

    def check(self, obj, ctx):
        rights = resolve(obj, self.right)
        for value in resolve(obj, self.left):
            if any(value is r for r in rights):
                yield self._violation(
                    obj,
                    f"{name_of(value)} appears at both '{self.left}' and "
                    f"'{self.right}'{(' -- ' + self.note) if self.note else ''}",
                )


@dataclass(frozen=True)
class Distinct(Constraint):
    """
    Two single-valued paths must not resolve to the same object.
    """

    left: str = ""
    right: str = ""
    note: str = ""

    def referenced_paths(self) -> tuple[PathReference, ...]:
        """
        Return every field path this constraint follows.
        """
        return (PathReference(self.left), PathReference(self.right))

    def check(self, obj, ctx):
        left, right = resolve(obj, self.left), resolve(obj, self.right)
        if left and right and left[0] is right[0]:
            yield self._violation(
                obj,
                f"'{self.left}' and '{self.right}' are the same "
                f"{name_of(left[0])}{(' -- ' + self.note) if self.note else ''}",
            )


@dataclass(frozen=True)
class FlagOnPath(Constraint):
    """
    Every value at ``path`` must carry ``flag`` equal to ``expected``.

    Separates events the controller waits for from events it guards against.
    """

    path: str = ""
    flag: str = ""
    expected: Any = True
    note: str = ""

    def referenced_paths(self) -> tuple[PathReference, ...]:
        """
        Return every field path this constraint follows.
        """
        return (
            PathReference(self.path),
            PathReference(self.flag, relative_to=self.path),
        )

    def check(self, obj, ctx):
        for value in resolve(obj, self.path):
            if getattr(value, self.flag, None) != self.expected:
                yield self._violation(
                    obj,
                    f"{name_of(value)} at '{self.path}' has "
                    f"{self.flag}={getattr(value, self.flag, None)!r}, expected "
                    f"{self.expected!r}{(' -- ' + self.note) if self.note else ''}",
                )


@dataclass(frozen=True)
class Ordered(Constraint):
    """
    A numeric lower bound must not exceed its upper bound.

    Not expressible in OWL at any expressivity, which is why comparisons of two data
    values were always going to live in this layer.
    """

    lower: str = ""
    upper: str = ""

    def referenced_paths(self) -> tuple[PathReference, ...]:
        """
        Return every field path this constraint follows.
        """
        return (PathReference(self.lower), PathReference(self.upper))

    def check(self, obj, ctx):
        lo, hi = getattr(obj, self.lower, None), getattr(obj, self.upper, None)
        if lo is not None and hi is not None and lo > hi:
            yield self._violation(obj, f"{self.lower}={lo} exceeds {self.upper}={hi}")


@dataclass(frozen=True)
class NonEmptyString(Constraint):
    """
    A string field must carry content.
    """

    field_name: str = ""
    note: str = ""

    def referenced_paths(self) -> tuple[PathReference, ...]:
        """
        Return every field path this constraint follows.
        """
        return (PathReference(self.field_name),)

    def check(self, obj, ctx):
        if not str(getattr(obj, self.field_name, "") or "").strip():
            yield self._violation(
                obj,
                f"'{self.field_name}' is empty"
                f"{(' -- ' + self.note) if self.note else ''}",
            )


@dataclass(frozen=True)
class ExactlyOne(Constraint):
    """
    Require exactly one populated path from a declared field set.
    """

    fields: tuple[str, ...] = ()
    """
    Paths whose populated values are mutually exclusive and collectively required.
    """

    def referenced_paths(self) -> tuple[PathReference, ...]:
        """
        Return every field path this constraint follows.
        """
        return tuple(PathReference(name) for name in self.fields)

    def check(
        self,
        obj: Any,
        context: EvaluationContext,
    ) -> Iterable[ConstraintViolation]:
        """
        Report when zero or multiple declared paths are populated.
        """
        populated = [
            field_name for field_name in self.fields if resolve(obj, field_name)
        ]
        if len(populated) == 1:
            return
        expected = ", ".join(self.fields)
        actual = ", ".join(populated) if populated else "none"
        yield self._violation(
            obj,
            f"exactly one of ({expected}) must be populated; found {actual}",
        )


@dataclass(frozen=True)
class Forbidden(Constraint):
    """
    Fires whenever its guard holds.

    The general shape for "this combination must not occur", where the fault is a
    relationship between fields rather than a bad value in one of them -- a continuously
    modulated phase with nothing monitored, an invariant with no violation event.
    Without it these have to be smuggled into a field check, which then silently passes
    whenever that unrelated field happens to be populated.
    """

    message: str = ""

    def referenced_paths(self) -> tuple[PathReference, ...]:
        """
        Return every field path this constraint follows.
        """
        return ()

    def check(self, obj, ctx):
        yield self._violation(obj, self.message)


@dataclass(frozen=True)
class Closure(Constraint):
    """
    Everything reachable from a container must be declared by it.

    ``container`` names the field holding the declared members; ``reached_via`` the
    paths along which references escape. A hypothesis that references something it does
    not declare is not self-contained, and so cannot be cached, versioned or shipped as
    a unit.
    """

    container: str = ""
    reached_via: tuple[str, ...] = ()
    note: str = ""

    def referenced_paths(self) -> tuple[PathReference, ...]:
        """
        Return every field path this constraint follows.
        """
        return (
            PathReference(self.container),
            *(PathReference(path) for path in self.reached_via),
        )

    def check(self, obj, ctx):
        declared = resolve(obj, self.container)
        for path in self.reached_via:
            for value in resolve(obj, path):
                if not any(value is d for d in declared):
                    yield self._violation(
                        obj,
                        f"{name_of(value)} reached via '{path}' is not "
                        f"declared in '{self.container}'"
                        f"{(' -- ' + self.note) if self.note else ''}",
                    )


@dataclass(frozen=True)
class NotAmong(Constraint):
    """
    A value must not appear in a collection on the same object.
    """

    value_path: str = ""
    collection: str = ""
    note: str = ""

    def referenced_paths(self) -> tuple[PathReference, ...]:
        """
        Return every field path this constraint follows.
        """
        return (PathReference(self.value_path), PathReference(self.collection))

    def check(self, obj, ctx):
        members = resolve(obj, self.collection)
        for value in resolve(obj, self.value_path):
            if any(value is m or value == m for m in members):
                yield self._violation(
                    obj,
                    f"{name_of(value)} at '{self.value_path}' also appears "
                    f"in '{self.collection}'"
                    f"{(' -- ' + self.note) if self.note else ''}",
                )


@dataclass(frozen=True)
class SharedValue(Constraint):
    """
    Two paths must have at least one value in common.

    A safety margin relating a setpoint and a threshold about different participants is
    not a margin about anything.
    """

    left: str = ""
    right: str = ""
    note: str = ""

    def referenced_paths(self) -> tuple[PathReference, ...]:
        """
        Return every field path this constraint follows.
        """
        return (PathReference(self.left), PathReference(self.right))

    def check(self, obj, ctx):
        lefts, rights = resolve(obj, self.left), resolve(obj, self.right)
        if (
            lefts
            and rights
            and not any(left is right for left in lefts for right in rights)
        ):
            yield self._violation(
                obj,
                f"'{self.left}' and '{self.right}' share no value"
                f"{(' -- ' + self.note) if self.note else ''}",
            )


@dataclass(frozen=True)
class DistinctAcross(Constraint):
    """
    Members of a collection must not share an identical value set.

    Two phases whose terminating events coincide cannot be told apart by the controller,
    which usually means they should be merged or one needs its own boundary.
    """

    collection: str = ""
    key_path: str = ""
    note: str = ""

    def referenced_paths(self) -> tuple[PathReference, ...]:
        """
        Return every field path this constraint follows.
        """
        return (
            PathReference(self.collection),
            PathReference(self.key_path, relative_to=self.collection),
        )

    def check(self, obj, ctx):
        members = resolve(obj, self.collection)
        keys = [
            (m, frozenset(id(v) for v in resolve(m, self.key_path))) for m in members
        ]
        for i, (a, ka) in enumerate(keys):
            if not ka:
                continue
            for b, kb in keys[i + 1 :]:
                if ka == kb:
                    yield self._violation(
                        a,
                        f"shares its entire '{self.key_path}' with "
                        f"{name_of(b)}{(' -- ' + self.note) if self.note else ''}",
                    )


@dataclass(frozen=True)
class GraphShape(Constraint):
    """
    A container's members must form a single connected, entered graph.

    ``edges`` names the field holding edges; ``tail`` and ``head`` the edge endpoints.
    Checks that exactly one member has no incoming edge, and that no member is isolated.
    """

    container: str = ""
    edges: str = ""
    tail: str = "from_phase"
    head: str = "to_phase"

    def referenced_paths(self) -> tuple[PathReference, ...]:
        """
        Return every field path this constraint follows.
        """
        return (
            PathReference(self.container),
            PathReference(self.edges),
            PathReference(self.tail, relative_to=self.edges),
            PathReference(self.head, relative_to=self.edges),
        )

    def check(self, obj, ctx):
        members = resolve(obj, self.container)
        edges = resolve(obj, self.edges)
        heads = {id(h) for e in edges for h in resolve(e, self.head)}
        tails = {id(t) for e in edges for t in resolve(e, self.tail)}
        entries = [m for m in members if id(m) not in heads]
        if len(entries) != 1:
            yield self._violation(
                obj,
                f"{len(entries)} entry members in '{self.container}', "
                "expected exactly 1",
            )
        for m in members:
            if id(m) not in heads and id(m) not in tails:
                yield ConstraintViolation(
                    self.id,
                    self.severity,
                    name_of(m),
                    f"participates in no '{self.edges}'",
                )


@dataclass(frozen=True)
class Grounded(Constraint):
    """
    Every demand must be met somewhere, or declared as an assumption.

    ``needs`` selects the demanding members, ``supplies`` those that satisfy them, and
    ``excused_when`` marks demands that are presupposed of the world rather than
    produced by the plan.
    """

    container: str = ""
    needs: str = ""
    supplies: tuple[str, ...] = ()
    """
    Several paths, because a demand can be met in more than one way: a condition may be
    a phase's declared subgoal, or the exit condition of a requirement on some earlier
    phase.
    """

    supplies_when: Optional[Callable[[Any], bool]] = None
    """
    Filter on the supplying side.

    Without it a demand can satisfy itself: the same requirement is reachable both as something needed and as something
    supplied, and the check silently passes.
    """

    excused_when: Optional[Callable[[Any], bool]] = None
    note: str = ""

    def referenced_paths(self) -> tuple[PathReference, ...]:
        """
        Return every field path this constraint follows.
        """
        return (
            PathReference(self.container),
            PathReference(self.needs, relative_to=self.container),
            *(PathReference(path) for path in self.supplies),
        )

    def check(self, obj, ctx):
        supplied: set[int] = set()
        for path in self.supplies:
            for value in resolve(obj, path):
                if self.supplies_when is not None and not self.supplies_when(value):
                    continue
                supplied.update(
                    id(t) for t in (resolve(value, "on_condition") or [value])
                )
        for member in resolve(obj, self.container):
            for demand in resolve(member, self.needs):
                if self.excused_when and self.excused_when(demand):
                    continue
                target = resolve(demand, "on_condition") or [demand]
                if not any(id(t) in supplied for t in target):
                    yield ConstraintViolation(
                        self.id,
                        self.severity,
                        name_of(member),
                        f"{name_of(target[0])} is required but nothing supplies "
                        f"it{(' -- ' + self.note) if self.note else ''}",
                    )


@dataclass(frozen=True)
class MostSpecificTypes(Constraint):
    """
    Require collection members to use ontology leaf classes.
    """

    collection: str = ""
    """
    Path to the members whose ontology classifications are checked.
    """

    def referenced_paths(self) -> tuple[PathReference, ...]:
        """
        Return every field path this constraint follows.
        """
        return (PathReference(self.collection),)

    def check(
        self, subject: Any, context: "EvaluationContext"
    ) -> Iterable[ConstraintViolation]:
        """
        Report members that have a more specific registered class.
        """
        for member in resolve(subject, self.collection):
            descendants = [
                candidate
                for candidate in context.registry.all()
                if candidate.cls is not type(member)
                and not candidate.abstract
                and issubclass(candidate.cls, type(member))
            ]
            if descendants:
                choices = ", ".join(
                    sorted(candidate.local_name for candidate in descendants)
                )
                yield self._violation(
                    member,
                    f"uses the general type {type(member).__name__}; "
                    f"select a more specific type such as {choices}",
                )


@dataclass(frozen=True)
class ConceptReferenceType(Constraint):
    """
    Require concept references to denote descendants of one ontology class.
    """

    path: str = ""
    """
    Path to the concept reference being classified.
    """

    concept_root: type[Any] = object
    """
    Ontology class whose concrete descendants are allowed.
    """

    def referenced_paths(self) -> tuple[PathReference, ...]:
        """
        Return every field path this constraint follows.
        """
        return (PathReference(self.path),)

    def check(
        self, subject: Any, context: "EvaluationContext"
    ) -> Iterable[ConstraintViolation]:
        """
        Report unresolved, unrelated, or overly general concepts.
        """
        for reference in resolve(subject, self.path):
            metadata = (
                context.registry.meta_for_iri(reference.iri)
                if isinstance(reference, ConceptReference)
                else None
            )
            if (
                metadata is None
                or metadata.abstract
                or metadata.cls is self.concept_root
                or not issubclass(metadata.cls, self.concept_root)
            ):
                yield self._violation(
                    subject,
                    f"'{self.path}' must reference a concrete "
                    f"{self.concept_root.__name__} concept",
                )


@dataclass(frozen=True)
class TypeCoverage(Constraint):
    """
    Report expected ontology types absent from a collection.
    """

    collection: str = ""
    """
    Path to the members providing type coverage.
    """

    required_types: tuple[type[Any], ...] = ()
    """
    Ontology types expected at least once in the collection.
    """

    def referenced_paths(self) -> tuple[PathReference, ...]:
        """
        Return every field path this constraint follows.
        """
        return (PathReference(self.collection),)

    def check(
        self, subject: Any, context: "EvaluationContext"
    ) -> Iterable[ConstraintViolation]:
        """
        Report each expected type not represented by a member.
        """
        members = resolve(subject, self.collection)
        for required_type in self.required_types:
            if not any(isinstance(member, required_type) for member in members):
                yield self._violation(
                    subject,
                    f"'{self.collection}' has no {required_type.__name__}",
                )


@dataclass(frozen=True)
class TypeOrder(Constraint):
    """
    Require typed graph edges to follow a declared conceptual order.
    """

    edges: str = ""
    """
    Path to the graph edges being checked.
    """

    tail: str = "from_phase"
    """
    Path from an edge to its source member.
    """

    head: str = "to_phase"
    """
    Path from an edge to its target member.
    """

    expected_types: tuple[type[Any], ...] = ()
    """
    Ontology types in their permitted forward order.
    """

    def referenced_paths(self) -> tuple[PathReference, ...]:
        """
        Return every field path this constraint follows.
        """
        return (
            PathReference(self.edges),
            PathReference(self.tail, relative_to=self.edges),
            PathReference(self.head, relative_to=self.edges),
        )

    def check(
        self, subject: Any, context: "EvaluationContext"
    ) -> Iterable[ConstraintViolation]:
        """
        Report edges whose typed endpoints run backward.
        """
        for edge in resolve(subject, self.edges):
            sources = resolve(edge, self.tail)
            targets = resolve(edge, self.head)
            if not sources or not targets:
                continue
            source_position = self._position(sources[0])
            target_position = self._position(targets[0])
            if (
                source_position is not None
                and target_position is not None
                and source_position >= target_position
            ):
                yield self._violation(
                    edge,
                    f"{type(sources[0]).__name__} must precede "
                    f"{type(targets[0]).__name__}",
                )

    def _position(self, member: Any) -> Optional[int]:
        """
        Return the member's position in the declared type order.
        """
        for position, expected_type in enumerate(self.expected_types):
            if isinstance(member, expected_type):
                return position
        return None


# ---------------------------------------------------------------------------
# Derived checks: required content, from metadata alone
# ---------------------------------------------------------------------------


@dataclass
class EvaluationContext:
    """
    Everything reachable in the hypothesis under evaluation.
    """

    instances: list[Any] = dc_field(default_factory=list)
    """
    Every object supplied to the current suite evaluation.
    """

    registry: OntologyRegistry = REGISTRY
    """
    Ontology registry defining classifications used by generic checks.
    """

    def of_type(self, cls: type) -> list[Any]:
        return [i for i in self.instances if isinstance(i, cls)]


def derived_required_checks(
    obj: Any, registry: OntologyRegistry = REGISTRY
) -> Iterable[ConstraintViolation]:
    """
    Check every ``min_count`` declared on the object's class.

    Generated, never written. The model states ``min_count`` once, on the field, and
    this reads it back -- so a required-content check cannot fall out of step with the
    declaration it enforces.
    """
    for fname, pmeta in property_meta(type(obj)).items():
        if pmeta.min_count <= 0:
            continue
        value = getattr(obj, fname, None)
        count = (
            0
            if value is None
            else len(value) if isinstance(value, (list, tuple, set, frozenset)) else 1
        )
        if count < pmeta.min_count:
            yield ConstraintViolation(
                "REQ",
                "error",
                name_of(obj),
                f"'{fname}' has {count} value(s), needs at least "
                f"{pmeta.min_count}"
                + (f" -- {pmeta.comment.splitlines()[0]}" if pmeta.comment else ""),
            )
        if pmeta.max_count is not None and count > pmeta.max_count:
            yield ConstraintViolation(
                "CARD",
                "error",
                name_of(obj),
                f"'{fname}' has {count} value(s), at most {pmeta.max_count} allowed",
            )


def derived_range_checks(
    obj: Any, registry: OntologyRegistry = REGISTRY
) -> Iterable[ConstraintViolation]:
    """
    Check that field values are of their declared type.

    Python enforces annotations nowhere at runtime, so a generated hypothesis can put
    anything in any field. This is the cheapest guard against a structurally plausible
    but type-confused hypothesis.
    """
    from .validation import _range_types, _resolved_hints

    hints = _resolved_hints(type(obj))
    for fname in property_meta(type(obj)):
        targets = [
            t
            for t in _range_types(hints.get(fname))
            if isinstance(t, type) and t not in (str, int, float, bool, object)
        ]
        if not targets:
            continue
        value = getattr(obj, fname, None)
        values = value if isinstance(value, (list, tuple, set, frozenset)) else [value]
        for v in values:
            if v is None:
                continue
            if not any(isinstance(v, t) for t in targets):
                yield ConstraintViolation(
                    "RANGE",
                    "error",
                    name_of(obj),
                    f"'{fname}' holds {type(v).__name__}, expected one of "
                    + ", ".join(t.__name__ for t in targets),
                )


# ---------------------------------------------------------------------------
# Suite
# ---------------------------------------------------------------------------


@dataclass
class ConstraintSuite:
    """
    A microworld's declared constraints, plus the derived ones.
    """

    name: str
    constraints: Sequence[Constraint]
    registry: OntologyRegistry = REGISTRY

    def __post_init__(self) -> None:
        """
        Reject any constraint whose paths its subject type cannot follow.

        :raises ConstraintPathError: for the first path with a step no field declares.
        """
        resolver = FieldPathResolver(self.registry)
        for constraint in self.constraints:
            self._check_paths(constraint, resolver)

    @staticmethod
    def _check_paths(constraint: Constraint, resolver: FieldPathResolver) -> None:
        """
        Check every path one constraint follows against the fields of its subject type.
        """
        for reference in constraint.referenced_paths():
            origin = (constraint.applies_to,)
            if reference.relative_to:
                origin = resolver.resolve(origin, reference.relative_to).types
            if not origin:
                continue
            resolution = resolver.resolve(origin, reference.path)
            if resolution.unknown_step is not None:
                raise ConstraintPathError(
                    constraint_id=constraint.id,
                    path=reference.path,
                    unknown_step=resolution.unknown_step,
                    subject_type=constraint.applies_to,
                )

    def evaluate(self, instances: Iterable[Any]) -> list[ConstraintViolation]:
        pool = list(instances)
        ctx = EvaluationContext(instances=pool, registry=self.registry)
        out: list[ConstraintViolation] = []
        for obj in pool:
            if self.registry.meta_for(type(obj)) is None:
                continue
            out.extend(derived_required_checks(obj, self.registry))
            out.extend(derived_range_checks(obj, self.registry))
        for constraint in self.constraints:
            for obj in pool:
                if not isinstance(obj, constraint.applies_to):
                    continue
                if constraint.when is not None and not constraint.when(obj):
                    continue
                out.extend(constraint.check(obj, ctx))
        return out

    def report(self, instances: Iterable[Any]) -> str:
        violations = self.evaluate(instances)
        errors = [v for v in violations if v.severity == "error"]
        warnings = [v for v in violations if v.severity == "warning"]
        ids = sorted({v.check for v in violations})
        lines = [
            (
                f"{self.name}: {len(errors)} errors, {len(warnings)} warnings "
                f"| checks fired: {', '.join(ids) if ids else 'none'}"
            )
        ]
        lines += [f"  {v}" for v in errors + warnings]
        return "\n".join(lines)


__all__ = [
    "Closure",
    "ConceptReferenceType",
    "Constraint",
    "ConstraintSuite",
    "ConstraintViolation",
    "Disjoint",
    "Distinct",
    "DistinctAcross",
    "EvaluationContext",
    "ExactlyOne",
    "FlagOnPath",
    "Forbidden",
    "GraphShape",
    "Grounded",
    "Inclusion",
    "MostSpecificTypes",
    "NonEmptyString",
    "NotAmong",
    "Ordered",
    "SharedValue",
    "TypeCoverage",
    "TypeOrder",
    "derived_range_checks",
    "derived_required_checks",
    "name_of",
    "resolve",
]
