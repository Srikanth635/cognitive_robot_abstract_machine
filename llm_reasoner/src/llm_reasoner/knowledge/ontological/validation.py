"""Checks derivable from ontological metadata alone, with no domain knowledge.

WHY THIS IS LOAD-BEARING
------------------------
While the model lived in OWL, a handful of modelling errors were caught by the
formalism: an anti-rigid class could not quietly become a subclass, and a
disjointness violation was a logical contradiction a reasoner would report.
Python catches none of that. Ordinary inheritance is happy to let a role be
declared as a subclass, and nothing complains.

These checks are the replacement. For the Flanagan microworld there is also an
OWL export to diff against hand-authored TTL, which is a stronger test -- but a
second microworld will have no independent TTL, because its export is generated
from the same metadata it would be checked against. From that point on this
module is the only automated guard on the modelling discipline, which is why it
is worth more than its size suggests.

The checks here are about STRUCTURE, not content: whether the declarations are
coherent, not whether a hypothesis is well formed. Domain constraints live in
the microworld that owns them.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Optional, Type, get_args, get_origin

from .decorators import REGISTRY, ClassMeta, ConceptReference, OntologyRegistry, property_meta
from .rigidity import ConceptLevel, Rigidity


@dataclass(frozen=True)
class Violation:
    """One failed check."""

    check: str
    severity: str  # "error" | "warning"
    subject: str
    message: str

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"[{self.check}] {self.severity.upper()} {self.subject}: {self.message}"


def _is_role_class(cls: Type[Any]) -> bool:
    """Whether a class is realised through krrood's Role pattern."""
    try:
        from krrood.patterns.role import Role
    except ImportError:  # pragma: no cover - krrood always present in practice
        return False
    return any(base is Role or getattr(base, "__origin__", None) is Role
               for base in inspect.getmro(cls)) or any(
        getattr(b, "__origin__", None) is Role for b in getattr(cls, "__orig_bases__", ())
    )


def _declared_classes(registry: OntologyRegistry, module: Optional[str]) -> list[ClassMeta]:
    return registry.module(module) if module else registry.all()


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_rigidity_realisation(metas: list[ClassMeta]) -> list[Violation]:
    """V01 -- anti-rigid classes must be roles, rigid ones must not.

    The single most consequential check in the module. An anti-rigid class
    realised as a subclass is exactly the mistake OWL used to make visible: it
    asserts that filling a role is essential to the thing filling it, so a body
    that stops being the initial support would have to stop being that body.
    """
    out = []
    for m in metas:
        is_role = _is_role_class(m.cls)
        if m.rigidity is Rigidity.ANTI_RIGID and not is_role:
            out.append(Violation(
                "V01", "error", m.local_name,
                "declared anti-rigid but is not a Role[T]; filling a role must "
                "not be essential to the thing that fills it. If this is a "
                "hypothesis-side slot rather than a bound role, it is "
                "description-level and should not be declared anti-rigid at "
                "all -- anti-rigidity is a claim about instantiation."))
        elif m.rigidity is Rigidity.RIGID and is_role:
            out.append(Violation(
                "V01", "error", m.local_name,
                "declared rigid but realised as a Role[T]; a rigid class should "
                "be an ordinary subclass"))
    return out


def check_role_taker_declared(metas: list[ClassMeta]) -> list[Violation]:
    """V02 -- an anti-rigid class must name what takes the role."""
    return [
        Violation("V02", "error", m.local_name,
                  "anti-rigid class declares no role_taker; a role with nothing "
                  "to attach to is unbindable")
        for m in metas
        if m.rigidity is Rigidity.ANTI_RIGID and not m.role_taker
    ]


def check_disjointness_consistency(
    metas: list[ClassMeta], registry: OntologyRegistry
) -> list[Violation]:
    """V03 -- disjointness must name real classes and must not contradict
    subsumption.

    Declaring A disjoint from B while A inherits from B is a contradiction the
    reasoner used to catch. Here it is silent, and produces a class whose
    instances are required to be instances of something they are excluded from.
    """
    out = []
    names = {m.local_name for m in metas}
    for m in metas:
        for other_name in m.disjoint_with:
            if registry.names_are_ambiguous(other_name):
                out.append(Violation(
                    "V03", "warning", m.local_name,
                    f"disjoint_with names '{other_name}', which is declared by "
                    "more than one class; resolution falls back to the nearest "
                    "module and may not be what was meant"))
            if other_name not in names:
                out.append(Violation(
                    "V03", "error", m.local_name,
                    f"disjoint_with names an undeclared class: {other_name}"))
                continue
            other = registry.by_name(other_name, near=m.cls.__module__)
            if other is None:
                continue
            if issubclass(m.cls, other.cls) or issubclass(other.cls, m.cls):
                out.append(Violation(
                    "V03", "error", m.local_name,
                    f"declared disjoint from {other_name} but the two are "
                    "related by inheritance"))
            if m.local_name == other_name:
                out.append(Violation(
                    "V03", "error", m.local_name, "declared disjoint from itself"))
    return out


def check_sibling_overlap(metas: list[ClassMeta], registry: OntologyRegistry) -> list[Violation]:
    """V04 -- rigid siblings that are neither disjoint nor deliberately
    overlapping.

    A warning, not an error. Non-disjoint rigid siblings are legitimate, but
    they are also the shape that should usually have been a role, so silence
    here is worth breaking. Suppress by declaring the disjointness explicitly.
    """
    out = []
    by_parent: dict[Type[Any], list[ClassMeta]] = {}
    declared = {m.cls for m in metas}
    for m in metas:
        for base in m.cls.__bases__:
            if base in declared:
                by_parent.setdefault(base, []).append(m)
    for parent, children in by_parent.items():
        if len(children) < 2:
            continue
        for i, a in enumerate(children):
            for b in children[i + 1:]:
                if a.rigidity is not Rigidity.RIGID or b.rigidity is not Rigidity.RIGID:
                    continue
                if b.local_name in a.disjoint_with or a.local_name in b.disjoint_with:
                    continue
                parent_meta = registry.meta_for(parent)
                if parent_meta is not None and parent_meta.disjoint_siblings:
                    continue
                out.append(Violation(
                    "V04", "warning", a.local_name,
                    f"rigid sibling of {b.local_name} with no disjointness "
                    "declared either way; if the overlap is intended, say so "
                    "explicitly, and if it is not, one of them is probably a role"))
    return out


def check_field_ranges(metas: list[ClassMeta], registry: OntologyRegistry) -> list[Violation]:
    """V05 -- every described field must range over something known.

    A field pointing at an undeclared class is a hole in the model: nothing can
    validate it, generate it, or export it.
    """
    out = []
    declared = {m.cls for m in metas}
    for m in metas:
        hints = _resolved_hints(m.cls)
        for fname in property_meta(m.cls):
            hint = hints.get(fname)
            if hint is None:
                out.append(Violation(
                    "V05", "warning", f"{m.local_name}.{fname}",
                    "field type could not be resolved; range is unknown"))
                continue
            for target in _range_types(hint):
                if target in (str, int, float, bool, type(None), Any):
                    continue
                if target in (list, set, frozenset, tuple, dict):
                    # a bare collection annotation: V06 reports the missing
                    # element type, so V05 stays quiet rather than duplicating
                    continue
                if target is ConceptReference or isinstance(target, str):
                    continue
                if isinstance(target, type) and issubclass(target, Enum):
                    continue
                if target not in declared and registry.meta_for(target) is None:
                    out.append(Violation(
                        "V05", "error", f"{m.local_name}.{fname}",
                        f"ranges over undeclared type {getattr(target, '__name__', target)}"))
    return out


def _is_collection_hint(hint: Any) -> bool:
    """Whether a type hint denotes a collection.

    Handles both parameterised hints (``list[X]``) and bare builtins (``list``),
    since a bare ``list`` has no ``get_origin`` and would otherwise read as a
    single-valued field.
    """
    if hint in (list, set, frozenset, tuple):
        return True
    return get_origin(hint) in (list, set, frozenset, tuple)


def check_functional_cardinality(metas: list[ClassMeta]) -> list[Violation]:
    """V06 -- functional and collection declarations must match the field type.

    A functional field typed as a list, or a non-functional one typed as a
    single value, means the metadata and the code disagree -- and the metadata
    is what the exporter and the schema builder believe.
    """
    out = []
    for m in metas:
        hints = _resolved_hints(m.cls)
        for fname, pmeta in property_meta(m.cls).items():
            hint = hints.get(fname)
            if hint is None:
                continue
            is_collection = _is_collection_hint(hint)
            if pmeta.functional and is_collection:
                out.append(Violation(
                    "V06", "error", f"{m.local_name}.{fname}",
                    "declared functional but typed as a collection"))
            if not pmeta.functional and not is_collection and pmeta.max_count != 1:
                out.append(Violation(
                    "V06", "warning", f"{m.local_name}.{fname}",
                    "not declared functional but typed as a single value; "
                    "either mark it functional or make it a collection"))
            if pmeta.max_count is not None and pmeta.min_count > pmeta.max_count:
                out.append(Violation(
                    "V06", "error", f"{m.local_name}.{fname}",
                    f"min_count {pmeta.min_count} exceeds max_count {pmeta.max_count}"))
    return out


def check_inverse_pairs(metas: list[ClassMeta], registry: OntologyRegistry) -> list[Violation]:
    """V07 -- an inverse must name a field that exists on the range class."""
    out = []
    for m in metas:
        hints = _resolved_hints(m.cls)
        for fname, pmeta in property_meta(m.cls).items():
            if not pmeta.inverse_of:
                continue
            targets = [t for t in _range_types(hints.get(fname))
                       if isinstance(t, type) and registry.meta_for(t)]
            if not targets:
                out.append(Violation(
                    "V07", "warning", f"{m.local_name}.{fname}",
                    f"declares inverse_of={pmeta.inverse_of} but its range is "
                    "not a declared class, so the inverse cannot be checked"))
                continue
            for target in targets:
                if pmeta.inverse_of not in property_meta(target):
                    out.append(Violation(
                        "V07", "error", f"{m.local_name}.{fname}",
                        f"inverse_of names '{pmeta.inverse_of}', which is not a "
                        f"described field of {target.__name__}"))
    return out


def check_symbol_base(metas: list[ClassMeta]) -> list[Violation]:
    """V08 -- described classes must be Symbol subclasses.

    Symbol subclassing is what places instances in the symbol graph, which is
    what makes them EQL-queryable. A described class that is not a Symbol is
    invisible to every query written against the model.
    """
    try:
        from krrood.symbol_graph.symbol_graph import Symbol
    except ImportError:  # pragma: no cover
        return []
    out = []
    for m in metas:
        if isinstance(m.cls, type) and issubclass(m.cls, Enum):
            continue
        if not issubclass(m.cls, Symbol):
            out.append(Violation(
                "V08", "error", m.local_name,
                "is not a Symbol subclass, so its instances never enter the "
                "symbol graph and cannot be queried"))
    return out


def check_level_consistency(metas: list[ClassMeta], registry: OntologyRegistry) -> list[Violation]:
    """V09 -- a description-level class must not range over occurrence-level
    content.

    The stereotype/episode boundary, checked mechanically. A hypothesis holding
    a reference to a world occurrence is the contamination that Phase 5's scope
    separation exists to prevent, and it is silent under EQL's closed-world
    quantification rather than being an error.
    """
    out = []
    for m in metas:
        if m.level is not ConceptLevel.DESCRIPTION:
            continue
        hints = _resolved_hints(m.cls)
        for fname in property_meta(m.cls):
            for target in _range_types(hints.get(fname)):
                tm = registry.meta_for(target) if isinstance(target, type) else None
                if tm is not None and tm.level is ConceptLevel.OCCURRENCE:
                    out.append(Violation(
                        "V09", "error", f"{m.local_name}.{fname}",
                        f"description-level class ranges over occurrence-level "
                        f"{tm.local_name}; stereotypical knowledge must not "
                        "reference episodes"))
    return out


def check_iri_hygiene(metas: list[ClassMeta]) -> list[Violation]:
    """V10 -- IRIs must be well formed and locally unique."""
    out = []
    seen: dict[str, str] = {}
    for m in metas:
        if "#" not in m.iri and "/" not in m.iri:
            out.append(Violation("V10", "error", m.local_name,
                                 f"IRI is not a valid identifier: {m.iri}"))
        if m.iri in seen and seen[m.iri] != m.local_name:
            out.append(Violation("V10", "error", m.local_name,
                                 f"IRI collides with {seen[m.iri]}"))
        seen[m.iri] = m.local_name
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolved_hints(cls: Type[Any]) -> dict[str, Any]:
    """Type hints resolved field by field, in the namespace of the class that
    declared each one.

    Deliberately not ``typing.get_type_hints``, which is all-or-nothing across
    the MRO: a single unresolvable annotation anywhere in a base class -- and
    krrood's ``Symbol`` has one -- would make every field of every subclass look
    unresolvable, turning a validator into a source of noise. Here a field that
    cannot be resolved is skipped and the rest still get checked.
    """
    import sys
    import typing

    out: dict[str, Any] = {}
    for klass in reversed(inspect.getmro(cls)):
        annotations = klass.__dict__.get("__annotations__", {})
        if not annotations:
            continue
        module = sys.modules.get(klass.__module__)
        globalns = dict(vars(module)) if module else {}
        globalns.setdefault("Optional", typing.Optional)
        globalns.setdefault("List", typing.List)
        for name, ann in annotations.items():
            if not isinstance(ann, str):
                out[name] = ann
                continue
            try:
                resolved = eval(ann, globalns, {cls.__name__: cls})
            except Exception:
                continue
            out[name] = _resolve_forward_refs(resolved, globalns, cls)
    return out


def _resolve_forward_refs(hint: Any, globalns: dict[str, Any], cls: Type[Any]) -> Any:
    """Replace quoted names left inside a generic with the classes they name.

    ``list["ActionPhase"]`` evaluates to a generic whose argument is the STRING
    ``"ActionPhase"``, not the class. Anything reading the type -- the validator,
    the schema builder, the exporter -- then sees no declared class and silently
    treats the field as having no range, which drops it from the generated schema
    without any error being raised.

    Resolving the inner names here means one place handles it rather than every
    consumer independently, and a field whose forward reference cannot be
    resolved is left as it was rather than being lost.
    """
    args = get_args(hint)
    if not args:
        return hint
    replaced = []
    changed = False
    for arg in args:
        if isinstance(arg, str):
            candidate = globalns.get(arg) or ({cls.__name__: cls}).get(arg)
            if isinstance(candidate, type):
                replaced.append(candidate)
                changed = True
                continue
        replaced.append(arg)
    if not changed:
        return hint
    origin = get_origin(hint)
    try:
        return origin[tuple(replaced)] if len(replaced) > 1 else origin[replaced[0]]
    except Exception:
        return hint


def _range_types(hint: Any) -> list[Any]:
    """Flatten a type hint into the concrete types it can hold."""
    if hint is None:
        return []
    origin = get_origin(hint)
    if origin is None:
        return [hint]
    args = get_args(hint)
    out: list[Any] = []
    for a in args:
        if a is type(None):
            continue
        out.extend(_range_types(a))
    return out or [origin]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def check_abstract_has_subclasses(metas: list[ClassMeta]) -> list[Violation]:
    """V11 -- an abstract class with no concrete subclass is dead vocabulary.

    ``abstract`` is advisory: it excludes a class from the generation schema
    rather than blocking construction. That makes it silent when wrong, so the
    one thing worth checking is that something actually specialises it.
    """
    declared = {m.cls for m in metas}
    out = []
    for m in metas:
        if not m.abstract:
            continue
        if not any(c is not m.cls and issubclass(c, m.cls) for c in declared):
            out.append(Violation(
                "V11", "warning", m.local_name,
                "declared abstract but nothing specialises it, so it can never "
                "appear in a hypothesis"))
    return out


def check_identity_base(metas: list[ClassMeta]) -> list[Violation]:
    """V12 -- description-level classes must carry a stable identity.

    Without ``local_id`` a fragment cannot be cached, cited in an explanation,
    or round-tripped: Python object identity does not survive serialisation.
    Retrofitting identity after ORM mappings exist is painful, so it is checked
    from the start.
    """
    from .decorators import KnowledgeEntity

    out = []
    for m in metas:
        if m.level is not ConceptLevel.DESCRIPTION or m.abstract:
            continue
        if isinstance(m.cls, type) and issubclass(m.cls, Enum):
            continue
        if _is_role_class(m.cls):
            continue                       # bound roles are token-level
        if not issubclass(m.cls, KnowledgeEntity):
            out.append(Violation(
                "V12", "error", m.local_name,
                "description-level class does not inherit KnowledgeEntity, so "
                "its instances have no stable identity or label"))
    return out


ALL_CHECKS = (
    "V01 rigidity realisation",
    "V02 role taker declared",
    "V03 disjointness consistency",
    "V04 sibling overlap",
    "V05 field ranges",
    "V06 functional cardinality",
    "V07 inverse pairs",
    "V08 symbol base",
    "V09 level consistency",
    "V10 IRI hygiene",
    "V11 abstract has subclasses",
    "V12 identity base",
)


def validate(
    module: Optional[str] = None, registry: OntologyRegistry = REGISTRY
) -> list[Violation]:
    """Run every structural check over the declared model.

    :param module: restrict to one module and its submodules; ``None`` for all.
    :returns: violations, errors and warnings together.
    """
    metas = _declared_classes(registry, module)
    out: list[Violation] = []
    out += check_rigidity_realisation(metas)
    out += check_role_taker_declared(metas)
    out += check_disjointness_consistency(metas, registry)
    out += check_sibling_overlap(metas, registry)
    out += check_field_ranges(metas, registry)
    out += check_functional_cardinality(metas)
    out += check_inverse_pairs(metas, registry)
    out += check_symbol_base(metas)
    out += check_level_consistency(metas, registry)
    out += check_iri_hygiene(metas)
    out += check_abstract_has_subclasses(metas)
    out += check_identity_base(metas)
    return out


def report(module: Optional[str] = None, registry: OntologyRegistry = REGISTRY) -> str:
    """Human-readable validation summary."""
    violations = validate(module, registry)
    errors = [v for v in violations if v.severity == "error"]
    warnings = [v for v in violations if v.severity == "warning"]
    n = len(_declared_classes(registry, module))
    lines = [f"{n} declared classes | {len(errors)} errors, {len(warnings)} warnings"]
    for v in errors + warnings:
        lines.append(f"  {v}")
    return "\n".join(lines)


__all__ = ["Violation", "validate", "report", "ALL_CHECKS"]