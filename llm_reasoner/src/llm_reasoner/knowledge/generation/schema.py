"""The output shape a model must produce, derived from the compiled classes.

DERIVED, NEVER HAND-WRITTEN
---------------------------
A hand-written schema drifts from the model it claims to describe, and the drift
is silent: the model emits something the schema permits and the classes reject.
Everything here is read back from the same ``@ontological`` metadata that the
constraint suite and the OWL exporter read, so a field renamed in a microworld
changes the schema without a second edit.

WHY A FLAT GRAPH AND NOT A NESTED OBJECT
----------------------------------------
Resolution asks for one object's slots and gets a flat array of leaf values back.
A hypothesis is not one object: it is a graph of a hundred or more fragments --
a structure referencing twelve phases, each referencing subgoals, condition
requirements, boundary events and role slots, cross-linked in both directions.

Nesting that as a single JSON literal fails on three counts. It cannot express a
cycle, and phases genuinely do refer back to conditions that other phases
establish. It duplicates any fragment referenced twice, so a shared condition
becomes two conditions and the identity that made sharing meaningful is lost.
And a single malformed bracket invalidates everything, leaving the model no way
to know which fragment was at fault.

So the shape is a flat list of fragments, each with an ``id`` and a ``type``, and
every cross-reference is an id. That is uniform, cycle-tolerant, preserves
sharing, and lets a repair name one fragment. Construction is the same two-pass
approach the Turtle codec already uses: instantiate everything by type, then
wire the references.
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass, field
from dataclasses import fields as dataclass_fields
from enum import Enum, StrEnum
from typing import Any, Iterable, Optional, Sequence, Type, get_origin

from llm_reasoner.knowledge.ontological.decorators import (
    REGISTRY,
    ClassMeta,
    ConceptReference,
    OntologyRegistry,
    property_meta,
)
from llm_reasoner.knowledge.ontological.rigidity import ConceptLevel
from llm_reasoner.knowledge.ontological.validation import _range_types, _resolved_hints

PRIMITIVES = (str, bool, int, float)


class FieldKind(StrEnum):
    """What kind of answer a field wants.

    Deliberately not the resolver's ``SlotKind``. Two of its four kinds have no
    counterpart here: REFERENT selects an individual from a world, and
    generation has no world; BOUND marks what a caller already supplied, and a
    hypothesis starts entirely unfilled. What generation adds is CONCEPT: a
    reference to an ontology term rather than a hypothesis fragment. Such a
    field may be open, as with an external SOMA task, or closed over locally
    declared reference concepts, as with manipulation roles.
    """

    VALUE = "value"
    """A primitive or an enum member. The model invents it, or picks from a
    closed set of names."""

    FRAGMENT = "fragment"
    """A reference to another fragment in the same hypothesis, by id. The model
    must have emitted that fragment, which is why generation is bottom-up."""

    CONCEPT = "concept"
    """An ontology term referenced by IRI or canonical local name.

    Open fields accept external terms. Closed fields expose leaf concepts from
    a declared local ontology root without instantiating those occurrence-level
    concepts inside the hypothesis.
    """


@dataclass(frozen=True)
class FieldSchema:
    """One field a model may fill on a fragment."""

    name: str
    kind: FieldKind
    type_name: str
    collection: bool
    required: int
    """Minimum values for a well-formed fragment. Zero means optional."""

    options: tuple[str, ...] = ()
    """Closed values for a VALUE or CONCEPT field, or the concrete classes
    acceptable for a FRAGMENT field. Empty when the answer is open."""

    wire_value_aliases: tuple[str, ...] = ()
    """Accepted legacy spellings that are not advertised as canonical options."""
    docstring: Optional[str] = None
    """The field's own comment. Carries the reasoning behind the field, which is
    what tells a model how to choose rather than merely what shape to emit."""

    default: Any = None
    has_default: bool = False


@dataclass(frozen=True)
class ClassSchema:
    """One kind of fragment a model may emit."""

    name: str
    docstring: Optional[str]
    fields: tuple[FieldSchema, ...]
    parent: Optional[str] = None

    @property
    def required_fields(self) -> tuple[FieldSchema, ...]:
        return tuple(f for f in self.fields if f.required)


@dataclass
class GenerationSchema:
    """Everything a model needs to know to emit a hypothesis."""

    root: str
    """Class of the single top-level fragment. Everything else hangs off it, so
    the record can be discarded by discarding this node."""

    classes: tuple[ClassSchema, ...]
    modules: tuple[str, ...] = ()

    unreachable_modules: tuple[str, ...] = ()
    """
    The requested modules none of whose classes can appear under the root.
    """

    _by_name: dict[str, ClassSchema] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._by_name = {c.name: c for c in self.classes}

    def get(self, name: str) -> Optional[ClassSchema]:
        return self._by_name.get(name)

    @property
    def class_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.classes)

    def subtypes_of(self, name: str) -> tuple[str, ...]:
        """:return: Emittable classes acceptable where ``name`` is expected.

        Walks the emitted hierarchy. For a field whose declared range is
        abstract, use the field's own ``options`` instead: an abstract base
        never appears here, so the walk would start nowhere.
        """
        schema = self.get(name)
        if schema is None:
            return ()
        out = [name]
        frontier = [name]
        while frontier:
            current = frontier.pop()
            for candidate in self.classes:
                if candidate.parent == current and candidate.name not in out:
                    out.append(candidate.name)
                    frontier.append(candidate.name)
        return tuple(out)


# ---------------------------------------------------------------------------
# Derivation
# ---------------------------------------------------------------------------


def _concrete(meta: ClassMeta) -> bool:
    """Whether a class may appear in a hypothesis at all.

    Three exclusions, each for a different reason. Abstract classes exist to
    declare shared fields and are not instantiable. Occurrence-level classes are
    the resolver's business -- a bound role comes into existence at binding time,
    and offering one here would put an episodic entity inside a stereotype.
    Enums are values held by fragments, not fragments themselves.
    """
    if meta.abstract or meta.level is not ConceptLevel.DESCRIPTION:
        return False
    return not (isinstance(meta.cls, type) and issubclass(meta.cls, Enum))


def concrete_descendants(
    cls: Type[Any], declared: Iterable[ClassMeta]
) -> tuple[str, ...]:
    """Return the most specific emittable classes acceptable for ``cls``.

    Traversal passes THROUGH abstract and occurrence-level classes rather than
    stopping at them. A field ranging over an abstract base -- ``defines_parameter``
    over ``Parameter``, say -- would otherwise offer nothing at all: the base is
    excluded from the vocabulary because it cannot be instantiated, and its
    concrete subclasses are never reached, so the model is told to reference a
    type it was never shown and every threshold and setpoint becomes
    unreachable.
    """
    candidates = [
        meta
        for meta in declared
        if _concrete(meta) and isinstance(meta.cls, type) and issubclass(meta.cls, cls)
    ]
    leaves = [
        candidate
        for candidate in candidates
        if not any(
            other.cls is not candidate.cls and issubclass(other.cls, candidate.cls)
            for other in candidates
        )
    ]
    return tuple(sorted(candidate.local_name for candidate in leaves))


def concept_descendants(
    concept_root: Type[Any],
    declared: Iterable[ClassMeta],
) -> tuple[str, ...]:
    """Return leaf ontology concepts allowed in a concept-reference field.

    Occurrence-level concepts are intentionally included here. They are names
    referenced by a description-level slot, not fragments instantiated inside
    the generated hypothesis.
    """
    candidates = [
        candidate
        for candidate in declared
        if not candidate.abstract
        and isinstance(candidate.cls, type)
        and issubclass(candidate.cls, concept_root)
        and not issubclass(candidate.cls, Enum)
    ]
    leaves = [
        candidate
        for candidate in candidates
        if not any(
            other.cls is not candidate.cls and issubclass(other.cls, candidate.cls)
            for other in candidates
        )
    ]
    return tuple(sorted(candidate.local_name for candidate in leaves))


def _field_schema(
    name: str,
    pmeta: Any,
    hints: dict[str, Any],
    registry: OntologyRegistry,
    declared_metas: Sequence[ClassMeta] = (),
    default: Any = None,
    has_default: bool = False,
) -> Optional[FieldSchema]:
    hint = hints.get(name)
    targets = _range_types(hint)
    collection = hint in (list, set, tuple) or get_origin(hint) in (
        list,
        set,
        frozenset,
        tuple,
    )

    enums = [t for t in targets if isinstance(t, type) and issubclass(t, Enum)]
    fragments = [
        t
        for t in targets
        if isinstance(t, type)
        and registry.meta_for(t) is not None
        and not issubclass(t, Enum)
    ]
    primitives = [t for t in targets if t in PRIMITIVES]
    concepts = [t for t in targets if t is ConceptReference]

    wire_value_aliases: tuple[str, ...] = ()
    if enums:
        kind, type_name = FieldKind.VALUE, enums[0].__name__
        options = tuple(m.name for m in enums[0])
        if issubclass(enums[0], StrEnum):
            wire_value_aliases = tuple(
                member.value for member in enums[0] if member.value != member.name
            )
    elif concepts:
        kind, type_name = FieldKind.CONCEPT, "ConceptReference"
        options = (
            concept_descendants(
                pmeta.concept_root,
                declared_metas,
            )
            if pmeta.concept_root is not None
            else ()
        )
    elif fragments:
        kind = FieldKind.FRAGMENT
        type_name = fragments[0].__name__
        options = concrete_descendants(fragments[0], declared_metas)
    elif primitives:
        kind = FieldKind.VALUE
        type_name = primitives[0].__name__
        options = ()
    else:
        return None

    return FieldSchema(
        name=name,
        kind=kind,
        type_name=type_name,
        collection=collection,
        required=pmeta.min_count,
        options=options,
        wire_value_aliases=wire_value_aliases,
        docstring=pmeta.comment,
        default=default,
        has_default=has_default,
    )


def derive_schema(
    modules: Sequence[str],
    root: str,
    registry: OntologyRegistry = REGISTRY,
) -> GenerationSchema:
    """Build the output shape from the classes the selected modules declare.

    :param modules: Dotted module paths, as chosen by the capability registry.
    :param root: Name of the top-level class the hypothesis is rooted at.
    """
    metas: list[ClassMeta] = []
    seen: set[str] = set()
    for module in modules:
        for meta in registry.module(module):
            if meta.iri in seen:
                continue
            seen.add(meta.iri)
            metas.append(meta)

    declared = {m.cls for m in metas}
    classes: list[ClassSchema] = []
    for meta in sorted(metas, key=lambda m: m.local_name):
        if not _concrete(meta):
            continue
        hints = _resolved_hints(meta.cls)
        raw_fields_by_name = {f.name: f for f in dataclass_fields(meta.cls)}
        fields: list[FieldSchema] = []
        for name, pmeta in property_meta(meta.cls).items():
            # PropertyMeta carries ontological metadata only; the field's actual
            # default lives on the plain dataclasses.Field property_meta() built
            # it from, so it is read back here rather than from pmeta.
            raw_field = raw_fields_by_name.get(name)
            has_default = raw_field is not None and raw_field.default is not MISSING
            schema = _field_schema(
                name,
                pmeta,
                hints,
                registry,
                declared_metas=metas,
                default=raw_field.default if has_default else None,
                has_default=has_default,
            )
            if schema is not None:
                fields.append(schema)
        parent_metadata = [
            registry.meta_for(base) for base in meta.cls.__bases__ if base in declared
        ]
        parent = next(
            (
                parent_class_metadata.local_name
                for parent_class_metadata in parent_metadata
                if parent_class_metadata is not None
            ),
            None,
        )
        classes.append(
            ClassSchema(
                name=meta.local_name,
                docstring=(meta.comment or "").split("\n\n")[0] or None,
                fields=tuple(fields),
                parent=parent,
            )
        )
    reachable = _reachable_from(root, classes)
    reachable_names = {class_schema.name for class_schema in reachable}
    return GenerationSchema(
        root=root,
        classes=reachable,
        modules=tuple(modules),
        unreachable_modules=tuple(
            module
            for module in modules
            if not any(
                meta.local_name in reachable_names for meta in registry.module(module)
            )
        ),
    )


def _reachable_from(
    root: str, classes: Sequence[ClassSchema]
) -> tuple[ClassSchema, ...]:
    """Keep only what can actually be used, starting from the root.

    A class the root cannot reach through any chain of fields can never appear
    in a valid hypothesis, so listing it costs prompt budget and invites the
    model to emit something that will be rejected. With the pouring root this
    removes the generic structure class, which is the root only in the degraded
    case; with a degraded root it removes the pouring-specific vocabulary.

    Ordering is preserved so the vocabulary stays stable between runs.
    """
    by_name = {c.name: c for c in classes}
    if root not in by_name:
        return tuple(classes)
    reached = {root}
    frontier = [root]
    while frontier:
        current = by_name.get(frontier.pop())
        if current is None:
            continue
        for field_schema in current.fields:
            candidates = [field_schema.type_name, *field_schema.options]
            for name in candidates:
                if name in by_name and name not in reached:
                    reached.add(name)
                    frontier.append(name)
        # a subclass of something reachable is itself reachable, since it may
        # stand wherever its base may
        for candidate in classes:
            if candidate.parent in reached and candidate.name not in reached:
                reached.add(candidate.name)
                frontier.append(candidate.name)
    return tuple(c for c in classes if c.name in reached)


def to_json_schema(schema: GenerationSchema) -> dict[str, Any]:
    """Render the derived shape as JSON Schema, for constrained decoding.

    Emitted as one ``anyOf`` over per-type fragment shapes rather than as a
    nested document, mirroring the flat graph the loop expects: a fragment is an
    object with ``id``, ``type`` and that type's fields, and every reference is
    an id string.

    What it CAN enforce is worth having: the set of legal type names, the set of
    legal field names per type, enum membership, and that arrays are arrays. That
    removes the whole class of failures where a model invents a type or misspells
    a field, which are otherwise only caught after the response is parsed.

    What it cannot enforce is everything that makes a hypothesis coherent --
    that a reference resolves, that a gate ends the phase it leaves, that a
    subgoal is signalled by its own terminating event. Those are the acceptance
    constraints, and they remain the reason validation exists downstream. A
    schema-valid hypothesis can still be nonsense.
    """
    fragment_shapes: list[dict[str, Any]] = []
    for class_schema in schema.classes:
        properties: dict[str, Any] = {
            "id": {"type": "string"},
            "type": {"const": class_schema.name},
            "label": {"type": "string"},
        }
        required = ["id", "type"]
        for field_schema in class_schema.fields:
            properties[field_schema.name] = _json_field(field_schema)
            if field_schema.required:
                required.append(field_schema.name)
        fragment_shapes.append(
            {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            }
        )
    return {
        "type": "object",
        "properties": {
            "fragments": {
                "type": "array",
                "minItems": 1,
                "items": {"anyOf": fragment_shapes},
            }
        },
        "required": ["fragments"],
        "additionalProperties": False,
    }


def _json_field(field_schema: FieldSchema) -> dict[str, Any]:
    if field_schema.options and field_schema.kind in {
        FieldKind.VALUE,
        FieldKind.CONCEPT,
    }:
        leaf: dict[str, Any] = {"enum": list(field_schema.options)}
    elif field_schema.kind in (FieldKind.FRAGMENT, FieldKind.CONCEPT):
        leaf = {"type": "string"}
    elif field_schema.type_name == "bool":
        leaf = {"type": "boolean"}
    elif field_schema.type_name == "int":
        leaf = {"type": "integer"}
    elif field_schema.type_name == "float":
        leaf = {"type": "number"}
    else:
        leaf = {"type": "string"}
    if field_schema.collection:
        return {"type": "array", "items": leaf}
    return leaf


__all__ = [
    "ClassSchema",
    "FieldKind",
    "FieldSchema",
    "GenerationSchema",
    "concept_descendants",
    "concrete_descendants",
    "derive_schema",
    "to_json_schema",
]
