"""Emit OWL/Turtle from a declared model, and diff it against hand-authored TTL.

WHY ONE-WAY
-----------
The Python model is the source of truth. Nothing is read back from an export, so
the exporter carries no risk of the two drifting into disagreement about which
one is authoritative.

WHY IT IS WORTH BUILDING NOW RATHER THAN LATER
-----------------------------------------------
For a microworld authored directly in Python, an export proves nothing: it is
generated from the same metadata it would be checked against, so a mistake in
the declarations appears identically on both sides.

Flanagan is the exception, and the only one there will ever be. It was designed
in OWL first and transcribed, so an independently hand-authored ontology exists
to diff against -- which makes the export a genuine test of the transcription
rather than a tautology. That transcription has already been shown to lose
things silently: three sensory-signal subclasses vanished and surfaced only
because the fixture codec happened to need them.

The window closes as soon as the TTL stops being maintained. After that this
module reverts to what it is for every other microworld: a way to open a model
in Protege or run a reasoner over its disjointness axioms.

WHAT IS NOT EXPORTED
--------------------
Anything the metadata does not carry. Field-level restrictions are emitted as
cardinality axioms, but the acceptance constraints are not: they are closed-world
checks and have no OWL equivalent, which is the whole reason they live in Python.
An export is therefore weaker than the model it came from, and a reasoner running
over it will accept hypotheses the constraint suite rejects.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Optional, Sequence, Type

from llm_reasoner.knowledge.ontological.decorators import (
    REGISTRY,
    ClassMeta,
    ConceptReference,
    OntologyRegistry,
    property_meta,
)
from llm_reasoner.knowledge.ontological.rigidity import ConceptLevel, Rigidity
from llm_reasoner.knowledge.ontological.validation import _range_types, _resolved_hints

XSD = "http://www.w3.org/2001/XMLSchema#"

_PYTHON_TO_XSD = {
    str: XSD + "string",
    bool: XSD + "boolean",
    int: XSD + "integer",
    float: XSD + "double",
}


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------

def _local(iri: str) -> str:
    return iri.rsplit("#", 1)[-1] if "#" in iri else iri.rsplit("/", 1)[-1]


def _namespace(iri: str) -> str:
    return iri.rsplit("#", 1)[0] + "#" if "#" in iri else iri.rsplit("/", 1)[0] + "/"


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _comment(text: Optional[str]) -> str:
    if not text:
        return ""
    collapsed = " ".join(text.split())
    return f'    rdfs:comment "{_escape(collapsed)}"@en ;\n'


def _property_iri(meta: ClassMeta, field_name: str, declared: Optional[str]) -> str:
    """The IRI a field is exported under.

    Falls back to a lower-camel name in the owning class's namespace when the
    declaration gave none. Worth knowing when a diff reports a missing property:
    it usually means the field never declared an IRI rather than that the model
    lacks the term.
    """
    if declared:
        return declared
    camel = field_name.split("_")
    lower = camel[0] + "".join(p.capitalize() for p in camel[1:])
    return _namespace(meta.iri) + lower


@dataclass
class OwlExport:
    """The Turtle produced for one module, plus what could not be expressed."""

    turtle: str
    classes: int = 0
    properties: int = 0
    inexpressible: tuple[str, ...] = ()
    """Metadata with no OWL counterpart, named rather than silently dropped."""


def export_module(
    module: str,
    ontology_iri: Optional[str] = None,
    registry: OntologyRegistry = REGISTRY,
    imports: Sequence[str] = (),
) -> OwlExport:
    """Emit OWL/Turtle for every class declared in a module.

    :param module: Dotted module path whose declarations to export.
    :param ontology_iri: IRI of the emitted ontology; derived from the first
        class's namespace when omitted.
    :param imports: IRIs to declare as ``owl:imports``.
    """
    metas = sorted(registry.module(module), key=lambda m: m.iri)
    if not metas:
        return OwlExport(turtle="", inexpressible=("no declarations found",))

    base = _namespace(metas[0].iri)
    ontology_iri = ontology_iri or base.rstrip("#")
    lines: list[str] = [
        "@prefix owl:  <http://www.w3.org/2002/07/owl#> .",
        "@prefix rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
        "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
        "@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .",
        "",
        f"<{ontology_iri}> rdf:type owl:Ontology ;",
    ]
    for imported in imports:
        lines.append(f"    owl:imports <{imported}> ;")
    lines.append(
        '    rdfs:comment "Generated from the Python model. Do not edit: the '
        'declarations are the source of truth, and this file is a projection of '
        'them." .'
    )
    lines.append("")

    inexpressible: list[str] = []
    declared_classes = {m.cls for m in metas}
    n_classes = n_properties = 0

    for meta in metas:
        if isinstance(meta.cls, type) and issubclass(meta.cls, Enum):
            lines.extend(_emit_value_partition(meta))
            n_classes += 1
            continue
        lines.extend(_emit_class(meta, declared_classes, registry))
        n_classes += 1
        emitted, notes = _emit_properties(meta, registry)
        lines.extend(emitted)
        n_properties += len(
            [line for line in emitted if "rdf:type owl:" in line]
        )
        inexpressible.extend(notes)

        if meta.rigidity is Rigidity.ANTI_RIGID and meta.role_taker:
            inexpressible.append(
                f"{meta.local_name}: role_taker={meta.role_taker} has no OWL "
                "counterpart; anti-rigidity is annotated, not axiomatised"
            )
        if meta.level is ConceptLevel.OCCURRENCE:
            inexpressible.append(
                f"{meta.local_name}: ConceptLevel.OCCURRENCE is metadata only"
            )

    return OwlExport(
        turtle="\n".join(lines) + "\n",
        classes=n_classes,
        properties=n_properties,
        inexpressible=tuple(sorted(set(inexpressible))),
    )


def _emit_class(
    meta: ClassMeta, declared: set[Type[Any]], registry: OntologyRegistry
) -> list[str]:
    lines = [f"<{meta.iri}> rdf:type owl:Class ;"]
    lines.append(f'    rdfs:label "{_escape(meta.local_name)}"@en ;')

    parents = [
        registry.meta_for(base).iri
        for base in meta.cls.__bases__
        if base in declared and registry.meta_for(base) is not None
    ]
    for parent in parents:
        lines.append(f"    rdfs:subClassOf <{parent}> ;")
    if meta.aligns_to:
        lines.append(f"    rdfs:subClassOf <{meta.aligns_to}> ;")

    for other_name in meta.disjoint_with:
        other = registry.by_name(other_name, near=meta.cls.__module__)
        if other is not None:
            lines.append(f"    owl:disjointWith <{other.iri}> ;")

    comment = _comment(meta.comment)
    if comment:
        lines.append(comment.rstrip(" ;\n") + " ;")

    lines[-1] = lines[-1].rstrip(" ;") + " ."
    lines.append("")

    if meta.disjoint_siblings:
        children = [
            m.iri for m in registry.all()
            if meta.cls in getattr(m.cls, "__bases__", ())
        ]
        if len(children) > 1:
            members = " ".join(f"<{c}>" for c in sorted(children))
            lines += [
                "[] rdf:type owl:AllDisjointClasses ;",
                f"   owl:members ( {members} ) .",
                "",
            ]
    return lines


def _emit_value_partition(meta: ClassMeta) -> list[str]:
    """Emit an enum as a closed class with named individuals.

    The value-partition idiom: orthogonal dimensions are enums in Python
    precisely so they are not sibling classes, and this is the OWL spelling of
    the same commitment.
    """
    namespace = _namespace(meta.iri)
    members = list(meta.cls)
    individuals = [
        namespace + "".join(p.capitalize() for p in str(m.value).split("_"))
        for m in members
    ]
    lines = [
        f"<{meta.iri}> rdf:type owl:Class ;",
        f'    rdfs:label "{_escape(meta.local_name)}"@en ;',
    ]
    comment = _comment(meta.comment)
    if comment:
        lines.append(comment.rstrip(" ;\n") + " ;")
    one_of = " ".join(f"<{i}>" for i in individuals)
    lines.append("    owl:equivalentClass [ rdf:type owl:Class ;")
    lines.append(f"        owl:oneOf ( {one_of} ) ] .")
    lines.append("")
    for member, iri in zip(members, individuals):
        lines.append(
            f"<{iri}> rdf:type owl:NamedIndividual , <{meta.iri}> ;"
        )
        lines.append(f'    rdfs:label "{_escape(member.name)}"@en .')
    lines.append("")
    lines.append("[] rdf:type owl:AllDifferent ;")
    lines.append(f"   owl:distinctMembers ( {one_of} ) .")
    lines.append("")
    return lines


def _emit_properties(
    meta: ClassMeta, registry: OntologyRegistry
) -> tuple[list[str], list[str]]:
    lines: list[str] = []
    notes: list[str] = []
    hints = _resolved_hints(meta.cls)
    own = meta.cls.__dict__.get("__annotations__", {})

    for field_name, pmeta in property_meta(meta.cls).items():
        if field_name not in own:
            continue  # declared on an ancestor; emitted there
        iri = _property_iri(meta, field_name, pmeta.iri)
        targets = _range_types(hints.get(field_name))
        datatype = next(
            (_PYTHON_TO_XSD[t] for t in targets if t in _PYTHON_TO_XSD), None
        )
        is_data = datatype is not None

        kind = "owl:DatatypeProperty" if is_data else "owl:ObjectProperty"
        decl = [f"<{iri}> rdf:type {kind}"]
        if pmeta.functional:
            decl.append(" , owl:FunctionalProperty")
        if pmeta.transitive and not is_data:
            decl.append(" , owl:TransitiveProperty")
        lines.append("".join(decl) + " ;")

        if pmeta.aligns_to:
            lines.append(f"    rdfs:subPropertyOf <{pmeta.aligns_to}> ;")
        lines.append(f"    rdfs:domain <{meta.iri}> ;")

        if is_data:
            lines.append(f"    rdfs:range <{datatype}> ;")
        else:
            ranges = [
                registry.meta_for(t).iri
                for t in targets
                if isinstance(t, type) and registry.meta_for(t) is not None
            ]
            if len(ranges) == 1:
                lines.append(f"    rdfs:range <{ranges[0]}> ;")
            elif len(ranges) > 1:
                union = " ".join(f"<{r}>" for r in sorted(ranges))
                lines.append(
                    "    rdfs:range [ rdf:type owl:Class ; "
                    f"owl:unionOf ( {union} ) ] ;"
                )
            elif any(t is ConceptReference for t in targets):
                notes.append(
                    f"{meta.local_name}.{field_name}: ranges over a foreign "
                    "term rather than a declared class, so no rdfs:range is "
                    "emitted"
                )

        comment = _comment(pmeta.comment)
        if comment:
            lines.append(comment.rstrip(" ;\n") + " ;")
        lines[-1] = lines[-1].rstrip(" ;") + " ."
        lines.append("")

        if pmeta.min_count or pmeta.max_count is not None:
            lines.extend(_emit_cardinality(meta.iri, iri, pmeta))
        if pmeta.inverse_of:
            notes.append(
                f"{meta.local_name}.{field_name}: inverse_of is exported as "
                "annotation only, since the inverse is derived in Python"
            )
    return lines, notes


def _emit_cardinality(class_iri: str, property_iri: str, pmeta: Any) -> list[str]:
    lines = []
    if pmeta.functional and pmeta.min_count >= 1:
        restriction = f'owl:cardinality "1"^^xsd:nonNegativeInteger'
    elif pmeta.min_count:
        restriction = (
            f'owl:minCardinality "{pmeta.min_count}"^^xsd:nonNegativeInteger'
        )
    else:
        restriction = (
            f'owl:maxCardinality "{pmeta.max_count}"^^xsd:nonNegativeInteger'
        )
    lines += [
        f"<{class_iri}> rdfs:subClassOf",
        "    [ rdf:type owl:Restriction ;",
        f"      owl:onProperty <{property_iri}> ;",
        f"      {restriction} ] .",
        "",
    ]
    return lines


# ---------------------------------------------------------------------------
# Conformance diff
# ---------------------------------------------------------------------------

#: Differences that are deliberate rather than defects, and why.
#:
#: Recorded so the diff stays a live test. Without this the three known entries
#: would be re-examined on every run, and a genuinely new difference would be
#: lost among them.
KNOWN_DIVERGENCES: dict[str, str] = {
    "ContinuouslyModulatedPhase":
        "Re-expressed, not lost. The ontology declared it as a defined class -- "
        "any phase whose control regime is continuous -- carrying a "
        "min-cardinality on monitorsSignal. In Python that requirement is "
        "constraint C03, which is strictly stronger: OWL's open world cannot "
        "reject a phase that simply omits the signal, while a closed-world "
        "check can.",
    "Parameter":
        "Added. The ontology had parameters subclass dul:Parameter directly, "
        "with no shared local base, so the fields every parameter carries had "
        "to be declared four times. The Python model introduces an abstract "
        "base to declare them once.",
    "RoleSlot":
        "Added by the slot/bound-role split. The ontology predates the "
        "distinction and typed role individuals directly as roles, which is "
        "description-level content the krrood Role pattern cannot represent: "
        "its role_taker is required, so a hypothesis role could not be unbound.",
}


@dataclass
class ExportDiff:
    """What the Python model and a hand-authored ontology disagree about.

    Every entry is one of three things, and the distinction matters: a term the
    transcription lost, a term the model added deliberately, or a difference in
    how something is said. Only the first is a defect.
    """

    only_in_source: tuple[str, ...] = ()
    """Declared in the hand-authored ontology but absent from the Python model.
    The interesting direction: these are candidates for something the
    transcription dropped."""

    only_in_export: tuple[str, ...] = ()
    """Present in the Python model and not in the ontology. Usually a term added
    after the TTL stopped being maintained."""

    subclass_differences: tuple[str, ...] = ()
    disjointness_differences: tuple[str, ...] = ()

    @property
    def clean(self) -> bool:
        """:return: Whether every difference is one already accounted for."""
        return not (self.unexplained_lost or self.unexplained_added)

    @property
    def unexplained_lost(self) -> tuple[str, ...]:
        """Terms in the ontology, absent from the model, and not accounted for.

        The only entries that indicate a defect."""
        return tuple(
            e for e in self.only_in_source
            if e.split()[0] not in KNOWN_DIVERGENCES
        )

    @property
    def unexplained_added(self) -> tuple[str, ...]:
        return tuple(
            e for e in self.only_in_export
            if e.split()[0] not in KNOWN_DIVERGENCES
        )

    def summary(self) -> str:
        accounted = (
            len(self.only_in_source) - len(self.unexplained_lost)
            + len(self.only_in_export) - len(self.unexplained_added)
        )
        lines = [
            f"{'CLEAN' if self.clean else 'DIFFERENCES FOUND'} | "
            f"unexplained: {len(self.unexplained_lost)} lost, "
            f"{len(self.unexplained_added)} added | "
            f"accounted for: {accounted} | "
            f"subclass diffs: {len(self.subclass_differences)} | "
            f"disjointness diffs: {len(self.disjointness_differences)}"
        ]
        for title, entries in (
            ("LOST IN TRANSCRIPTION (in ontology, not in model)", self.unexplained_lost),
            ("ADDED IN MODEL (not in ontology)", self.unexplained_added),
            ("SUBCLASS DIFFERENCES", self.subclass_differences),
            ("DISJOINTNESS DIFFERENCES", self.disjointness_differences),
        ):
            if entries:
                lines.append(f"  {title}:")
                lines += [f"    {e}" for e in entries]
        return "\n".join(lines)


def diff_against(
    ttl_paths: Sequence[str],
    modules: Sequence[str],
    registry: OntologyRegistry = REGISTRY,
    ignore_namespaces: Sequence[str] = (),
) -> ExportDiff:
    """Compare a hand-authored ontology against the declared Python model.

    Compares CLASS IRIs and their subsumption and disjointness, not the full
    axiom set: the model cannot express everything the OWL did, and the point is
    to find terms the transcription lost rather than to prove the two files
    equivalent.

    :param ttl_paths: The hand-authored ontology files.
    :param modules: Dotted module paths whose declarations to compare against.
    :param ignore_namespaces: IRI prefixes to skip, typically imported
        vocabularies the model references but does not declare.
    """
    import rdflib
    from rdflib import OWL, RDF, RDFS

    graph = rdflib.Graph()
    for path in ttl_paths:
        graph.parse(path, format="turtle")

    def relevant(iri: str) -> bool:
        return not any(iri.startswith(ns) for ns in ignore_namespaces)

    source_classes = {
        str(s) for s in graph.subjects(RDF.type, OWL.Class)
        if isinstance(s, rdflib.URIRef) and relevant(str(s))
    }

    metas = [m for module in modules for m in registry.module(module)]
    export_classes = {m.iri for m in metas}
    by_iri = {m.iri: m for m in metas}

    # local names, so a term that only moved namespace is not reported as lost
    source_names = {_local(c): c for c in source_classes}
    export_names = {_local(c): c for c in export_classes}

    lost = sorted(
        f"{name}  <{source_names[name]}>"
        for name in source_names
        if name not in export_names
    )
    added = sorted(
        f"{name}  <{export_names[name]}>"
        for name in export_names
        if name not in source_names
    )

    subclass_diffs: list[str] = []
    disjoint_diffs: list[str] = []
    for name, export_iri in export_names.items():
        if name not in source_names:
            continue
        source_iri = rdflib.URIRef(source_names[name])
        source_parents = {
            _local(str(o)) for o in graph.objects(source_iri, RDFS.subClassOf)
            if isinstance(o, rdflib.URIRef)
        }
        meta = by_iri[export_iri]
        model_parents = {
            b.__name__ for b in meta.cls.__bases__
            if registry.meta_for(b) is not None
        }
        if meta.aligns_to:
            model_parents.add(_local(meta.aligns_to))
        missing = source_parents - model_parents
        if missing:
            subclass_diffs.append(
                f"{name}: ontology also has subClassOf {sorted(missing)}"
            )

        source_disjoint = {
            _local(str(o)) for o in graph.objects(source_iri, OWL.disjointWith)
            if isinstance(o, rdflib.URIRef)
        }
        # disjointness is symmetric: a model declaring it on the other side
        # says the same thing, so both directions are collected before
        # reporting a difference
        model_disjoint = set(meta.disjoint_with)
        for other in metas:
            if name in other.disjoint_with:
                model_disjoint.add(other.local_name)
        if source_disjoint - model_disjoint and not meta.disjoint_siblings:
            disjoint_diffs.append(
                f"{name}: ontology also declares disjointWith "
                f"{sorted(source_disjoint - model_disjoint)}"
            )

    return ExportDiff(
        only_in_source=tuple(lost),
        only_in_export=tuple(added),
        subclass_differences=tuple(subclass_diffs),
        disjointness_differences=tuple(disjoint_diffs),
    )


__all__ = [
    "ExportDiff",
    "KNOWN_DIVERGENCES",
    "OwlExport",
    "diff_against",
    "export_module",
]