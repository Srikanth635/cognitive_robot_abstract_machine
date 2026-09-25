"""The ``@ontological`` decorator and the field-level metadata that goes with it.

DESIGN NOTE -- WHY DECORATORS AND NOT METACLASSES
-------------------------------------------------
A metaclass would fight everything this model is built on at once: dataclasses,
``Role[T]``'s generics, ORMatic's introspection, and static typing. krrood's own
house pattern is descriptors and decorators, not metaclass magic, and staying
with it means generated classes behave like ordinary dataclasses everywhere else
in the stack.

The decorator does not modify the class. It records metadata in a registry that
validators, the schema builder and the OWL exporter read. A class is a normal
dataclass that happens to be described.
"""

from __future__ import annotations

from dataclasses import dataclass, field, field as dc_field
from typing import Any, Callable, Optional, Sequence, Type, TypeVar
from uuid import uuid4

from krrood.symbol_graph.symbol_graph import Symbol

from .rigidity import ConceptLevel, Rigidity

T = TypeVar("T")

#: Key under which field metadata is stored in ``dataclasses.field(metadata=...)``.
METADATA_KEY = "ontological"


@dataclass(eq=False)
class KnowledgeEntity(Symbol):
    """Base for every description-level thing a hypothesis is made of.

    Exists for two reasons that are cheap now and expensive to retrofit.

    STABLE IDENTITY. Every fragment needs an identifier that survives
    serialisation: the cache is keyed on hypothesis identity, and the resolver
    query surface must return fragments carrying the record they came from so
    that a resolution can cite which stereotypical assumptions it leaned on.
    Python object identity does not survive a round trip; ``local_id`` does.

    UNIFORM LABEL. Without a shared base, labels were declared ad hoc and five
    classes ended up with none, which makes generated content ragged and some
    fragments unexplainable.
    """

    local_id: str = field(default_factory=lambda: uuid4().hex[:12])
    """Stable within one hypothesis. Not globally unique: hypotheses are the
    unit of identity, and a fragment is identified by (hypothesis, local_id)."""

    label: str = ""
    """Human-readable name. The handle an explanation refers to."""


@dataclass(frozen=True)
class ConceptReference:
    """A reference to a term in an ontology this package does not compile.

    Fields such as ``ActionPhaseStructure.decomposes_task`` point at SOMA terms
    like ``soma:PouringInto``. Those are imported vocabulary, not compiled
    vocabulary, so there is no Python class to hold -- and holding a ``Type`` in
    a field would need a custom ORM serialiser before anything could be
    persisted, while every other field holds instances.

    So a reference is an ordinary value object carrying the IRI plus the
    materialised ancestor IRIs. Subsumption -- which the capability registry
    depends on to let an unseen action inherit the right modules -- becomes a set
    membership test, with no reasoner at runtime.

    Reversible by design: if SOMA is ever compiled too, this grows a
    ``resolved_type`` field and becomes a proxy. Unwinding ``Type[...]`` fields
    after ORM mappings were built on them would not be.
    """

    iri: str
    label: Optional[str] = None
    ancestors: frozenset[str] = frozenset()
    """Materialised ancestor IRIs, transitively closed and including ``iri``."""

    def is_subsumed_by(self, other: "ConceptReference | str") -> bool:
        """Whether this term is ``other`` or a specialisation of it."""
        target = other.iri if isinstance(other, ConceptReference) else other
        return target == self.iri or target in self.ancestors

    def __str__(self) -> str:  # pragma: no cover - display only
        return self.label or self.iri.rsplit("#", 1)[-1]


@dataclass(frozen=True)
class PropertyMeta:
    """Ontological metadata for one field.

    Attached through ``dataclasses.field(metadata=...)`` rather than a custom
    descriptor, so the field stays an ordinary dataclass field and nothing
    downstream needs to know it is described.
    """

    iri: Optional[str] = None
    aligns_to: Optional[str] = None
    """IRI of the foreign property this specialises, e.g. ``dul:definesTask``.
    Alignment is annotation only: it is exported faithfully but no foreign
    axioms are imported or reasoned with."""

    concept_root: Optional[Type[Any]] = None
    """Local ontology class whose descendants form a concept-reference range.

    This applies only to :class:`ConceptReference` fields. It closes a field
    over a selected vocabulary without turning referenced occurrence-level
    concepts into description-level hypothesis fragments.
    """

    functional: bool = False
    """At most one value. Realised as an ``Optional`` field rather than a list."""

    inverse_of: Optional[str] = None
    """Name of the field on the range class that is this field's inverse. Only
    one side is stored; the other is derived, so the two cannot disagree."""

    min_count: int = 0
    """Minimum values required for a well-formed instance. Not enforced at
    construction -- a hypothesis is assembled incrementally and is allowed to be
    incomplete until it is validated."""

    max_count: Optional[int] = None
    transitive: bool = False

    marks_assumption: bool = False
    """This boolean field flags a fragment as something the hypothesis
    presupposes rather than establishes.

    Declared as metadata rather than recognised by field name so that assumption
    extraction stays microworld-agnostic: the next microworld will call its
    equivalent something else, and a name-matching extractor would silently
    return nothing for it."""

    comment: Optional[str] = None


def prop(
    *,
    iri: Optional[str] = None,
    aligns_to: Optional[str] = None,
    concept_root: Optional[Type[Any]] = None,
    functional: bool = False,
    inverse_of: Optional[str] = None,
    min_count: int = 0,
    max_count: Optional[int] = None,
    transitive: bool = False,
    marks_assumption: bool = False,
    comment: Optional[str] = None,
    default: Any = None,
    default_factory: Optional[Callable[[], Any]] = None,
) -> Any:
    """Declare a dataclass field carrying ontological metadata.

    Thin wrapper over ``dataclasses.field``; the result is an ordinary field.
    """
    meta = PropertyMeta(
        iri=iri,
        aligns_to=aligns_to,
        concept_root=concept_root,
        functional=functional,
        inverse_of=inverse_of,
        min_count=min_count,
        max_count=max_count,
        transitive=transitive,
        marks_assumption=marks_assumption,
        comment=comment,
    )
    kwargs: dict[str, Any] = {"metadata": {METADATA_KEY: meta}}
    if default_factory is not None:
        kwargs["default_factory"] = default_factory
    else:
        kwargs["default"] = default
    return dc_field(**kwargs)


@dataclass(frozen=True)
class ClassMeta:
    """Ontological metadata for one class."""

    cls: Type[Any]
    iri: str
    rigidity: Rigidity = Rigidity.RIGID
    level: ConceptLevel = ConceptLevel.DESCRIPTION
    aligns_to: Optional[str] = None

    display_name: Optional[str] = None
    """Concise concept name used by ontology-oriented presentations."""

    disjoint_with: tuple[str, ...] = ()
    """Names of sibling classes this one is disjoint from. Symmetric: declaring
    it on one side is enough, and the validator closes the relation."""

    disjoint_siblings: bool = False
    """All DIRECT subclasses of this class are pairwise disjoint.

    The OWL ``AllDisjointClasses`` idiom, and the reason it exists: declaring
    disjointness pairwise across eight sibling phases means twenty-eight
    declarations that must be kept in sync by hand. One flag on the parent says
    the same thing and cannot drift."""

    role_taker: Optional[str] = None
    """For anti-rigid classes, the name of the class whose instances take the
    role. Required when rigidity is ANTI_RIGID -- an anti-rigid class with no
    taker has nothing to attach to."""

    abstract: bool = False
    comment: Optional[str] = None

    @property
    def local_name(self) -> str:
        return self.cls.__name__


class OntologyRegistry:
    """Everything declared by ``@ontological``, per module.

    Keyed by module so that a microworld can be introspected, validated and
    exported on its own -- which is what makes selective loading meaningful.
    """

    def __init__(self) -> None:
        self._by_class: dict[Type[Any], ClassMeta] = {}
        self._by_module: dict[str, list[ClassMeta]] = {}
        self._by_iri: dict[str, ClassMeta] = {}

    def register(self, meta: ClassMeta) -> None:
        if meta.iri in self._by_iri and self._by_iri[meta.iri].cls is not meta.cls:
            raise ValueError(
                f"IRI {meta.iri} already declared by "
                f"{self._by_iri[meta.iri].cls.__name__}"
            )
        self._by_class[meta.cls] = meta
        self._by_iri[meta.iri] = meta
        self._by_module.setdefault(meta.cls.__module__, []).append(meta)

    def meta_for(self, cls: Type[Any]) -> Optional[ClassMeta]:
        return self._by_class.get(cls)

    def meta_for_iri(self, iri: str) -> Optional[ClassMeta]:
        return self._by_iri.get(iri)

    def by_name(self, name: str, near: Optional[str] = None) -> Optional[ClassMeta]:
        """Look a class up by its Python name.

        ``disjoint_with`` names classes as strings because a class may be
        declared disjoint from one defined later in the same file. Resolving
        those names globally would silently mis-resolve as soon as two
        microworlds reused a name, so resolution prefers the declaring module,
        then its package, then falls back to a global match.

        :param near: module of the class doing the referring.
        """
        candidates = [m for m in self._by_class.values() if m.local_name == name]
        if not candidates:
            return None
        if len(candidates) == 1 or near is None:
            return candidates[0]
        for m in candidates:  # same module
            if m.cls.__module__ == near:
                return m
        package = near.rsplit(".", 1)[0]
        for m in candidates:  # same package
            if m.cls.__module__.startswith(package):
                return m
        return candidates[0]

    def names_are_ambiguous(self, name: str) -> bool:
        """Whether more than one declared class carries this Python name."""
        return sum(1 for m in self._by_class.values() if m.local_name == name) > 1

    def module(self, module_name: str) -> list[ClassMeta]:
        """All classes declared in a module, including submodules."""
        out: list[ClassMeta] = []
        for mod, metas in self._by_module.items():
            if mod == module_name or mod.startswith(module_name + "."):
                out.extend(metas)
        return out

    def modules(self) -> list[str]:
        return sorted(self._by_module)

    def all(self) -> list[ClassMeta]:
        return list(self._by_class.values())

    def clear(self) -> None:
        """Reset. For tests only."""
        self._by_class.clear()
        self._by_module.clear()
        self._by_iri.clear()


REGISTRY = OntologyRegistry()


def ontological(
    *,
    iri: str,
    rigidity: Rigidity = Rigidity.RIGID,
    level: ConceptLevel = ConceptLevel.DESCRIPTION,
    aligns_to: Optional[str] = None,
    display_name: Optional[str] = None,
    disjoint_with: Sequence[str] = (),
    disjoint_siblings: bool = False,
    role_taker: Optional[str] = None,
    abstract: bool = False,
    comment: Optional[str] = None,
    registry: Optional[OntologyRegistry] = None,
) -> Callable[[Type[T]], Type[T]]:
    """Describe a class ontologically without modifying it.

    :param iri: this term's own identifier.
    :param rigidity: RIGID for a subclass, ANTI_RIGID for a ``Role[T]``.
    :param level: DESCRIPTION for type-level content, META for statements about
        the knowledge itself.
    :param aligns_to: IRI of the foreign class this specialises. Annotation
        only -- no foreign axioms are imported.
    :param display_name: Concise concept name for ontology-oriented displays.
    :param disjoint_with: names of sibling classes this is disjoint from.
    :param disjoint_siblings: all direct subclasses of this class are pairwise
        disjoint, so they need not each declare it.
    :param role_taker: for anti-rigid classes, the name of the taker class.
    :param abstract: not directly instantiable in a hypothesis. Advisory: it
        excludes the class from the generation schema rather than blocking
        construction, since a decorator that rewrote ``__init__`` would stop the
        class behaving like an ordinary dataclass everywhere else.
    :param comment: falls back to the class docstring when omitted.
    :param registry: where to record. Defaults to the module-level REGISTRY;
        pass an explicit one to keep tests from polluting it.
    """

    def decorate(cls: Type[T]) -> Type[T]:
        (registry or REGISTRY).register(
            ClassMeta(
                cls=cls,
                iri=iri,
                rigidity=rigidity,
                level=level,
                aligns_to=aligns_to,
                display_name=display_name,
                disjoint_with=tuple(disjoint_with),
                disjoint_siblings=disjoint_siblings,
                role_taker=role_taker,
                abstract=abstract,
                comment=comment or (cls.__doc__ or "").strip() or None,
            )
        )
        return cls

    return decorate


def property_meta(cls: Type[Any]) -> dict[str, PropertyMeta]:
    """Ontological metadata for every described field of a dataclass."""
    from dataclasses import fields as dc_fields

    out: dict[str, PropertyMeta] = {}
    try:
        all_fields = dc_fields(cls)
    except TypeError:  # not a dataclass
        return out
    for f in all_fields:
        meta = f.metadata.get(METADATA_KEY)
        if isinstance(meta, PropertyMeta):
            out[f.name] = meta
    return out


def value_partition(
    iri: str, *, comment: Optional[str] = None
) -> Callable[[Type[T]], Type[T]]:
    """Describe an ``Enum`` standing for a closed set of alternatives.

    Value partitions -- coupling regime, condition mode -- are enums rather than
    class hierarchies because they are orthogonal dimensions. Modelling them as
    sibling classes is what forced the original ontology into the error of
    making a continuously modulated phase a sibling of a coupled one, when a
    pouring tilt phase is both.
    """
    return ontological(iri=iri, abstract=False, comment=comment)


__all__ = [
    "ClassMeta",
    "KnowledgeEntity",
    "ConceptReference",
    "OntologyRegistry",
    "PropertyMeta",
    "REGISTRY",
    "METADATA_KEY",
    "ontological",
    "prop",
    "property_meta",
    "value_partition",
]
