"""
Acceptance rules: what makes a hypothesis admissible.

Where the declarations already say what must hold, such as required counts, alternatives,
fixed values or the ownership of fragments, the checks are derived from them and never
written. The rules here state the relationships between several objects that a declaration
cannot.

Every check reads absence as falsity. A phase with no terminating event is an error and
not merely underspecified, which is what an under-generating model needs to be told. A
rule identifies itself by a member of an enumeration, and the docstring of that member is
its rationale, so the explanation is written once beside the name.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from operator import attrgetter
from typing import Any, Callable, Iterable, Optional, Sequence

from llm_reasoner.exceptions import LLMReasonerError
from llm_reasoner.knowledge_updated.ontological.entities import KnowledgeEntity
from llm_reasoner.knowledge_updated.ontological.ontology import Ontology
from llm_reasoner.knowledge_updated.ontological.paths import (
    FieldPathResolver,
    PathReference,
)
from llm_reasoner.knowledge_updated.ontological.vocabulary import Severity

# %% findings


@dataclass(frozen=True)
class AcceptanceViolation:
    """
    One way a hypothesis is inadmissible, naming the fragment responsible.
    """

    rule: StrEnum
    """
    The rule that was broken.
    """

    severity: Severity
    """
    How seriously the break counts.
    """

    subject: str
    """
    The fragment that breaks the rule.
    """

    message: str
    """
    What is wrong, in terms of the fragment.
    """


@dataclass(frozen=True)
class EvaluationContext:
    """
    Everything a check may read while it runs.
    """

    instances: tuple[Any, ...]
    """
    Every fragment of the hypothesis under evaluation.
    """

    ontology: Ontology
    """
    The declarations the fragments are instances of.
    """


# %% reading a hypothesis


def follow(subject: Any, path: str) -> list[Any]:
    """
    Return the values reached by following a dotted path, flattening collections.

    A step that holds nothing yields nothing. Paths are checked against the declared
    fields when a suite is built, so a step that names no field cannot occur here.
    """
    current = [subject]
    for step in path.split("."):
        reached: list[Any] = []
        for item in current:
            value = attrgetter(step)(item)
            if value is None:
                continue
            if isinstance(value, (list, tuple, set, frozenset)):
                reached.extend(member for member in value if member is not None)
            else:
                reached.append(value)
        current = reached
    return current


def identities(items: Iterable[Any]) -> set[int]:
    """
    Return the identities of some objects, for asking whether one of them is among them.
    """
    return {id(item) for item in items}


def name_of(fragment: Any) -> str:
    """
    Return how a violation refers to a fragment: its label, or its type and identifier.
    """
    if not isinstance(fragment, KnowledgeEntity):
        return str(fragment)
    return fragment.label or f"{type(fragment).__name__}<{fragment.local_id}>"


# %% rules


@dataclass(frozen=True, kw_only=True)
class AcceptanceRule(ABC):
    """
    A declared check over one kind of fragment.
    """

    id: StrEnum
    """
    Names the rule. The documentation of this member states why the rule exists.
    """

    applies_to: type
    """
    The kind of fragment the rule is evaluated for, subtypes included.
    """

    severity: Severity = Severity.ERROR
    """
    How seriously a break counts.
    """

    only_if: Optional[Callable[[Any], bool]] = None
    """
    Precondition: the rule is skipped for fragments it returns false for.
    """

    @abstractmethod
    def check(
        self, subject: Any, context: EvaluationContext
    ) -> Iterable[AcceptanceViolation]:
        """
        Return the violations of this rule by one fragment.
        """

    @abstractmethod
    def referenced_paths(self) -> tuple[PathReference, ...]:
        """
        Return every field path this rule follows, so a suite can check each one.
        """

    def _violation(self, subject: Any, message: str) -> AcceptanceViolation:
        """
        Report a break by a fragment.
        """
        return AcceptanceViolation(self.id, self.severity, name_of(subject), message)


@dataclass(frozen=True, kw_only=True)
class Inclusion(AcceptanceRule):
    """
    Every value at ``source`` must also appear at ``target``.
    """

    source: str
    """
    Path to the values that must be included.
    """

    target: str
    """
    Path to the values that must include them.
    """

    def check(self, subject, context):
        allowed = follow(subject, self.target)
        for value in follow(subject, self.source):
            if not any(value is candidate for candidate in allowed):
                yield self._violation(
                    subject,
                    f"{name_of(value)} at '{self.source}' is not among '{self.target}'",
                )

    def referenced_paths(self):
        return (PathReference(self.source), PathReference(self.target))


@dataclass(frozen=True, kw_only=True)
class Disjoint(AcceptanceRule):
    """
    Two paths must not share a value.
    """

    left: str
    """
    Path to the first set of values.
    """

    right: str
    """
    Path to the second set of values.
    """

    def check(self, subject, context):
        rights = follow(subject, self.right)
        for value in follow(subject, self.left):
            if any(value is candidate for candidate in rights):
                yield self._violation(
                    subject,
                    f"{name_of(value)} appears at both '{self.left}' and '{self.right}'",
                )

    def referenced_paths(self):
        return (PathReference(self.left), PathReference(self.right))


@dataclass(frozen=True, kw_only=True)
class Distinct(AcceptanceRule):
    """
    Two single-valued paths must not reach the same object.
    """

    left: str
    """
    Path to the first object.
    """

    right: str
    """
    Path to the second object.
    """

    def check(self, subject, context):
        left, right = follow(subject, self.left), follow(subject, self.right)
        if left and right and left[0] is right[0]:
            yield self._violation(
                subject,
                f"'{self.left}' and '{self.right}' are the same {name_of(left[0])}",
            )

    def referenced_paths(self):
        return (PathReference(self.left), PathReference(self.right))


@dataclass(frozen=True, kw_only=True)
class FlagOnPath(AcceptanceRule):
    """
    Every value at ``path`` must carry ``flag`` equal to ``expected``.
    """

    path: str
    """
    Path to the values that are checked.
    """

    flag: str
    """
    The field of each value that is read.
    """

    expected: Any = True
    """
    The value the flag must have.
    """

    def check(self, subject, context):
        for value in follow(subject, self.path):
            actual = attrgetter(self.flag)(value)
            if actual != self.expected:
                yield self._violation(
                    subject,
                    f"{name_of(value)} at '{self.path}' has {self.flag}={actual!r}, "
                    f"expected {self.expected!r}",
                )

    def referenced_paths(self):
        return (
            PathReference(self.path),
            PathReference(self.flag, relative_to=self.path),
        )


@dataclass(frozen=True, kw_only=True)
class Ordered(AcceptanceRule):
    """
    A numeric lower bound must not exceed its upper bound.
    """

    lower: str
    """
    The field holding the lower bound.
    """

    upper: str
    """
    The field holding the upper bound.
    """

    def check(self, subject, context):
        lower, upper = attrgetter(self.lower)(subject), attrgetter(self.upper)(subject)
        if lower is not None and upper is not None and lower > upper:
            yield self._violation(
                subject, f"{self.lower}={lower} exceeds {self.upper}={upper}"
            )

    def referenced_paths(self):
        return (PathReference(self.lower), PathReference(self.upper))


@dataclass(frozen=True, kw_only=True)
class SharedValue(AcceptanceRule):
    """
    Two paths must have at least one value in common, when both hold any.
    """

    left: str
    """
    Path to the first set of values.
    """

    right: str
    """
    Path to the second set of values.
    """

    def check(self, subject, context):
        lefts, rights = follow(subject, self.left), follow(subject, self.right)
        if (
            lefts
            and rights
            and not any(left is right for left in lefts for right in rights)
        ):
            yield self._violation(
                subject, f"'{self.left}' and '{self.right}' share no value"
            )

    def referenced_paths(self):
        return (PathReference(self.left), PathReference(self.right))


@dataclass(frozen=True, kw_only=True)
class DistinctAcross(AcceptanceRule):
    """
    Members of a collection must not share an identical, non-empty set of values.
    """

    collection: str
    """
    Path to the members.
    """

    key_path: str
    """
    Path from each member to the values compared.
    """

    def check(self, subject, context):
        keyed = [
            (member, frozenset(id(v) for v in follow(member, self.key_path)))
            for member in follow(subject, self.collection)
        ]
        for position, (member, key) in enumerate(keyed):
            if not key:
                continue
            for other, other_key in keyed[position + 1 :]:
                if key == other_key:
                    yield self._violation(
                        member,
                        f"shares its entire '{self.key_path}' with {name_of(other)}",
                    )

    def referenced_paths(self):
        return (
            PathReference(self.collection),
            PathReference(self.key_path, relative_to=self.collection),
        )


@dataclass(frozen=True, kw_only=True)
class GraphShape(AcceptanceRule):
    """
    A container's members must form a single connected graph with exactly one entry.
    """

    container: str
    """
    Path to the members.
    """

    edges: str
    """
    Path to the edges between the members.
    """

    tail: str
    """
    The field of an edge holding the member it leaves.
    """

    head: str
    """
    The field of an edge holding the member it enters.
    """

    def check(self, subject, context):
        members = follow(subject, self.container)
        edges = follow(subject, self.edges)
        heads = {id(h) for edge in edges for h in follow(edge, self.head)}
        tails = {id(t) for edge in edges for t in follow(edge, self.tail)}
        entries = [member for member in members if id(member) not in heads]
        if len(entries) != 1:
            yield self._violation(
                subject,
                f"{len(entries)} entry members in '{self.container}', expected exactly 1",
            )
        for member in members:
            if id(member) not in heads and id(member) not in tails:
                yield AcceptanceViolation(
                    self.id,
                    self.severity,
                    name_of(member),
                    f"participates in no '{self.edges}'",
                )

    def referenced_paths(self):
        return (
            PathReference(self.container),
            PathReference(self.edges),
            PathReference(self.tail, relative_to=self.edges),
            PathReference(self.head, relative_to=self.edges),
        )


@dataclass(frozen=True, kw_only=True)
class Demand:
    """
    What the members of a container ask for.
    """

    path: str
    """
    Path from a member to the demands it makes.
    """

    target: str = ""
    """
    Path from a demand to the thing that must be supplied. Empty when the demand itself
    is what must be supplied.
    """


@dataclass(frozen=True, kw_only=True)
class Supply:
    """
    Something that can meet a demand.
    """

    path: str
    """
    Path from the subject to the suppliers.
    """

    target: str = ""
    """
    Path from a supplier to the thing it supplies. Empty when the supplier itself is what
    it supplies.
    """

    supplied_when: Optional[Callable[[Any], bool]] = None
    """
    Filter on suppliers. Without it a demand could supply itself.
    """


@dataclass(frozen=True, kw_only=True)
class Grounded(AcceptanceRule):
    """
    Every demand must be met somewhere, or be excused as an assumption.
    """

    container: str
    """
    Path to the members that make demands.
    """

    demand: Demand
    """
    What the members demand.
    """

    supplies: tuple[Supply, ...]
    """
    Everything that can meet a demand, since a demand can be met in more than one way.
    """

    excused_when: Optional[Callable[[Any], bool]] = None
    """
    Demands that are presupposed of the world and need not be met.
    """

    def check(self, subject, context):
        supplied = self._supplied(subject)
        for member in follow(subject, self.container):
            for demand in follow(member, self.demand.path):
                if self.excused_when is not None and self.excused_when(demand):
                    continue
                wanted = (
                    follow(demand, self.demand.target)
                    if self.demand.target
                    else [demand]
                )
                if not any(id(item) in supplied for item in wanted):
                    yield AcceptanceViolation(
                        self.id,
                        self.severity,
                        name_of(member),
                        f"{name_of(wanted[0] if wanted else demand)} is required but "
                        "nothing supplies it",
                    )

    def _supplied(self, subject: Any) -> set[int]:
        """
        Return the identities of everything that meets a demand.
        """
        supplied: set[int] = set()
        for supply in self.supplies:
            for supplier in follow(subject, supply.path):
                if supply.supplied_when is not None and not supply.supplied_when(
                    supplier
                ):
                    continue
                provided = (
                    follow(supplier, supply.target) if supply.target else [supplier]
                )
                supplied.update(id(item) for item in provided)
        return supplied

    def referenced_paths(self):
        paths = [
            PathReference(self.container),
            PathReference(self.demand.path, relative_to=self.container),
        ]
        if self.demand.target:
            paths.append(
                PathReference(
                    self.demand.target,
                    relative_to=f"{self.container}.{self.demand.path}",
                )
            )
        for supply in self.supplies:
            paths.append(PathReference(supply.path))
            if supply.target:
                paths.append(PathReference(supply.target, relative_to=supply.path))
        return tuple(paths)


@dataclass(frozen=True, kw_only=True)
class RequiresKind(AcceptanceRule):
    """
    Some value at ``path`` must be of ``kind``.
    """

    path: str
    """
    Path to the values that are searched.
    """

    kind: type
    """
    The type at least one of the values must have.
    """

    def check(self, subject, context):
        if not any(
            isinstance(value, self.kind) for value in follow(subject, self.path)
        ):
            yield self._violation(
                subject, f"nothing at '{self.path}' is a {self.kind.__name__}"
            )

    def referenced_paths(self):
        return (PathReference(self.path),)


@dataclass(frozen=True, kw_only=True)
class MostSpecificTypes(AcceptanceRule):
    """
    Every member of ``collection`` must be of a kind that nothing concrete specialises.
    """

    collection: str
    """
    Path to the members whose kinds are checked.
    """

    def check(self, subject, context):
        concepts = context.ontology.concepts()
        for member in follow(subject, self.collection):
            more_specific = sorted(
                concept.local_name
                for concept in concepts
                if concept.cls is not type(member)
                and not concept.abstract
                and issubclass(concept.cls, type(member))
            )
            if more_specific:
                yield self._violation(
                    member,
                    f"is a general {type(member).__name__}; use a more specific kind, "
                    f"such as {', '.join(more_specific)}",
                )

    def referenced_paths(self):
        return (PathReference(self.collection),)


@dataclass(frozen=True, kw_only=True)
class TypeOrder(AcceptanceRule):
    """
    Every edge must lead from an earlier kind in ``expected_types`` to a later one.
    """

    edges: str
    """
    Path to the edges that are checked.
    """

    tail: str
    """
    Path from an edge to the member it leaves.
    """

    head: str
    """
    Path from an edge to the member it enters.
    """

    expected_types: tuple[type, ...]
    """
    The kinds in the order edges may lead through them. Members of none of them are not
    ordered.
    """

    def check(self, subject, context):
        for edge in follow(subject, self.edges):
            leaving = follow(edge, self.tail)
            entering = follow(edge, self.head)
            if not leaving or not entering:
                continue
            before = self._position(leaving[0])
            after = self._position(entering[0])
            if before is not None and after is not None and before >= after:
                yield self._violation(
                    edge,
                    f"{type(leaving[0]).__name__} must precede "
                    f"{type(entering[0]).__name__}",
                )

    def _position(self, member: Any) -> Optional[int]:
        """
        Return where the kind of a member comes in the order, or ``None`` if it is unlisted.
        """
        return next(
            (
                position
                for position, kind in enumerate(self.expected_types)
                if isinstance(member, kind)
            ),
            None,
        )

    def referenced_paths(self):
        return (
            PathReference(self.edges),
            PathReference(self.tail, relative_to=self.edges),
            PathReference(self.head, relative_to=self.edges),
        )


@dataclass(frozen=True, kw_only=True)
class Forbidden(AcceptanceRule):
    """
    Fires whenever its predicate holds.

    The general shape for a relationship between fields that no other kind expresses.
    """

    violated_when: Callable[[Any], bool]
    """
    Predicate that is true of a fragment that breaks the rule.
    """

    message: str
    """
    What is wrong when the predicate holds.
    """

    def check(self, subject, context):
        if self.violated_when(subject):
            yield self._violation(subject, self.message)

    def referenced_paths(self):
        return ()


# %% checks derived from declarations


class InstanceCheck(ABC):
    """
    A check over a whole hypothesis that reads the declarations and never a written rule.
    """

    @abstractmethod
    def check(self, context: EvaluationContext) -> Iterable[AcceptanceViolation]:
        """
        Return the violations found among the fragments of the context.
        """


# %% the suite


@dataclass
class RulePathError(LLMReasonerError):
    """
    A rule follows a path that its subject type cannot.
    """

    rule: StrEnum
    """
    The rule that declares the path.
    """

    path: str
    """
    The declared path.
    """

    unknown_step: str
    """
    The first step that no declared field matches.
    """

    subject_type: type
    """
    The type the rule applies to.
    """

    def __str__(self) -> str:
        return (
            f"rule {self.rule}: '{self.path}' has no field '{self.unknown_step}' "
            f"reachable from {self.subject_type.__name__}"
        )


@dataclass
class AcceptanceSuite:
    """
    The checks that decide whether a hypothesis is admissible.
    """

    name: str
    """
    Names the suite.
    """

    ontology: Ontology
    """
    The declarations the rules are written against.
    """

    rules: Sequence[AcceptanceRule]
    """
    The declared rules.
    """

    checks: Sequence[InstanceCheck] = ()
    """
    The checks derived from the declarations, run before the rules.
    """

    def __post_init__(self) -> None:
        """
        Refuse any rule that follows a field its subject cannot reach.

        :raises RulePathError: for the first such path.
        """
        resolver = FieldPathResolver(self.ontology)
        for rule in self.rules:
            self._check_paths(rule, resolver)

    @staticmethod
    def _check_paths(rule: AcceptanceRule, resolver: FieldPathResolver) -> None:
        """
        Check every path one rule follows against the fields of its subject type.
        """
        for reference in rule.referenced_paths():
            origin = (rule.applies_to,)
            if reference.relative_to:
                origin = resolver.resolve(origin, reference.relative_to).types
            if not origin:
                continue
            resolution = resolver.resolve(origin, reference.path)
            if resolution.unknown_step is not None:
                raise RulePathError(
                    rule=rule.id,
                    path=reference.path,
                    unknown_step=resolution.unknown_step,
                    subject_type=rule.applies_to,
                )

    def evaluate(self, instances: Iterable[Any]) -> tuple[AcceptanceViolation, ...]:
        """
        Run every derived check and every rule over the fragments of a hypothesis.
        """
        context = EvaluationContext(instances=tuple(instances), ontology=self.ontology)
        violations: list[AcceptanceViolation] = []
        for check in self.checks:
            violations.extend(check.check(context))
        for rule in self.rules:
            for subject in context.instances:
                if not isinstance(subject, rule.applies_to):
                    continue
                if rule.only_if is not None and not rule.only_if(subject):
                    continue
                violations.extend(rule.check(subject, context))
        return tuple(violations)
