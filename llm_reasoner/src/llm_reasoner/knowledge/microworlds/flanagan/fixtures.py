"""Load hypothesis Turtle into instances of the Python model.

WHY THIS EXISTS AND WHAT IT IS NOT
----------------------------------
The Python model is now the source of truth; the Turtle files are a design
document. This codec is therefore not an interchange layer -- it is a bridge for
two specific purposes:

* the two Phase-2 fixtures, one sound and one deliberately broken, are the only
  independent test data that exists, and the constraint suite needs something
  to run against;
* loading them cross-checks the transcription. Anything the TTL says that the
  Python model cannot receive is a hole in the transcription, and the codec
  reports it rather than silently dropping it.

Once generation runs, fixtures come from the LLM and this becomes throwaway.

TRANSLATION THAT IS NOT ONE-TO-ONE
----------------------------------
The Turtle predates the slot/bound-role split. It has role INDIVIDUALS typed
directly as roles -- ``hyp:r_Source a pour:SourceContainer`` -- because at the
time roles were description-level. In the Python model those are unbound slots
naming the role class they bind as, so the codec rewrites them. This is the one
place the two representations genuinely disagree, and the disagreement is a
correction rather than a loss.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field as dc_field, fields as dc_fields
from enum import Enum
from typing import Any, Optional, Type, get_args, get_origin

import rdflib
from rdflib import RDF, RDFS, BNode, Literal, URIRef

from llm_reasoner.knowledge.ontological.decorators import (
    REGISTRY,
    ClassMeta,
    ConceptReference,
    KnowledgeEntity,
    OntologyRegistry,
    property_meta,
)
from llm_reasoner.knowledge.ontological.rigidity import ConceptLevel
from llm_reasoner.knowledge.ontological.validation import _resolved_hints, _range_types


@dataclass
class LoadReport:
    """What the codec could and could not translate.

    Unmapped types and predicates are the interesting output: each is either a
    term the Python model is missing, or a term the Turtle carries that the
    model deliberately dropped. Both are worth seeing explicitly.
    """

    instances: dict[str, Any] = dc_field(default_factory=dict)
    unmapped_types: set[str] = dc_field(default_factory=set)
    unmapped_predicates: set[str] = dc_field(default_factory=set)
    rewritten_roles: int = 0
    notes: list[str] = dc_field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.unmapped_types and not self.unmapped_predicates

    def summary(self) -> str:
        lines = [
            f"{len(self.instances)} instances | "
            f"{self.rewritten_roles} role individuals rewritten as slots"
        ]
        if self.unmapped_types:
            lines.append(f"  unmapped types ({len(self.unmapped_types)}):")
            lines += [f"    {t}" for t in sorted(self.unmapped_types)]
        if self.unmapped_predicates:
            lines.append(f"  unmapped predicates ({len(self.unmapped_predicates)}):")
            lines += [f"    {p}" for p in sorted(self.unmapped_predicates)]
        lines += [f"  {n}" for n in self.notes]
        return "\n".join(lines)


#: RDF/OWL vocabulary that appears in a T-Box and is not domain content.
_SCHEMA_NAMESPACES = (
    "http://www.w3.org/2002/07/owl#",
    "http://www.w3.org/2000/01/rdf-schema#",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "http://www.w3.org/2001/XMLSchema#",
)

#: Predicates the Turtle carries under an IRI the Python model no longer uses.
#:
#: These are not bugs in either representation -- they are the residue of
#: decisions taken after the Turtle was written, and recording them explicitly
#: is better than a silent fallback that would hide a genuine mismatch:
#:
#: * ``adaptsTo`` moved from the pouring module into the core when in-flight
#:   adaptation turned out not to be specific to pouring;
#: * ``predictedTrajectory`` ran phase -> prediction in the Turtle and runs
#:   prediction -> phase in Python, because the prediction is the thing that
#:   owns its own identity and can be cited.
_LEGACY_PREDICATES: dict[str, tuple[str, bool]] = {
    "http://example.org/ontology/flanagan-pouring#adaptsTo": ("adapts_to", False),
    "http://example.org/ontology/flanagan-pouring#predictedTrajectory":
        ("predicted_for_phase", True),
}


def _camel(text: str) -> str:
    return "".join(part.capitalize() for part in re.split(r"[_\s]+", text) if part)


def _lower_camel(text: str) -> str:
    out = _camel(text)
    return out[:1].lower() + out[1:]


class _Index:
    """IRI lookup tables derived from the registry.

    Built from metadata rather than hand-maintained, so a class or field renamed
    in the model is picked up here without a second edit.
    """

    def __init__(self, registry: OntologyRegistry = REGISTRY, scope: Optional[str] = None):
        metas = registry.module(scope) if scope else registry.all()
        self.by_iri: dict[str, ClassMeta] = {m.iri: m for m in metas}
        self.registry = registry

        # enum member lookup: the TTL names value-partition members as
        # individuals, e.g. flan:ContinuouslyModulated
        self.enum_members: dict[str, Any] = {}
        for m in metas:
            if isinstance(m.cls, type) and issubclass(m.cls, Enum):
                namespace = m.iri.rsplit("#", 1)[0] + "#"
                for member in m.cls:
                    self.enum_members[namespace + _camel(str(member.value))] = member

        # (class, predicate IRI) -> field name
        self.fields: dict[tuple[Type[Any], str], str] = {}
        for m in metas:
            namespace = m.iri.rsplit("#", 1)[0] + "#"
            for fname, pmeta in property_meta(m.cls).items():
                for iri in self._candidate_iris(pmeta.iri, fname, namespace):
                    self.fields.setdefault((m.cls, iri), fname)

    @staticmethod
    def _candidate_iris(declared: Optional[str], fname: str, namespace: str) -> list[str]:
        """IRIs a predicate might legitimately carry for this field.

        Explicit metadata wins. The fallbacks cover fields whose IRI was never
        declared, which is itself worth knowing about -- see the note the loader
        emits for them.
        """
        if declared:
            return [declared]
        return [
            namespace + _lower_camel(fname),
            namespace + "has" + _camel(fname),
        ]

    def field_for(self, cls: Type[Any], predicate: str) -> Optional[str]:
        for klass in cls.__mro__:
            hit = self.fields.get((klass, predicate))
            if hit:
                return hit
        return None


def load_hypothesis(
    path: str,
    *,
    ontology_paths: tuple[str, ...] = (),
    scope: str = "llm_reasoner.knowledge.microworlds.flanagan",
    registry: OntologyRegistry = REGISTRY,
) -> LoadReport:
    """Load one hypothesis Turtle file into Python instances.

    :param path: the hypothesis (A-Box) file.
    :param ontology_paths: T-Box files, parsed only so that ``rdfs:subClassOf``
        is available for resolving a typed individual to the nearest class the
        Python model actually declares.
    :param scope: module whose declarations to translate against.
    """
    index = _Index(registry, scope)
    report = LoadReport()

    graph = rdflib.Graph()
    for extra in ontology_paths:
        graph.parse(extra, format="turtle")
    tbox_size = len(graph)
    graph.parse(path, format="turtle")

    # --- resolve each individual to the most specific declared class --------
    def declared_class_for(subject) -> Optional[ClassMeta]:
        """Walk up rdfs:subClassOf until a class the model declares is found.

        The Turtle sometimes types an individual with a class the Python model
        folded away; taking the nearest declared ancestor keeps such an
        individual usable rather than dropping it.
        """
        best: Optional[ClassMeta] = None
        for type_iri in graph.objects(subject, RDF.type):
            if not isinstance(type_iri, URIRef):
                continue
            seen, frontier = set(), [type_iri]
            depth = 0
            while frontier:
                current = frontier.pop()
                meta = index.by_iri.get(str(current))
                if meta is not None:
                    if best is None or depth == 0:
                        best = meta
                    break
                seen.add(current)
                frontier += [p for p in graph.objects(current, RDFS.subClassOf)
                             if isinstance(p, URIRef) and p not in seen]
                depth += 1
            else:
                if not str(type_iri).startswith(_SCHEMA_NAMESPACES):
                    report.unmapped_types.add(str(type_iri))
        return best

    # --- domain inference the Turtle relied on and Python does not ---------
    # In OWL, asserting a property whose domain is Subgoal silently retypes the
    # subject as a Subgoal. Python has no such inference, so an individual typed
    # only as a condition would lose its attainment signal. This reproduces that
    # one entailment explicitly, which is the honest way to consume a file
    # written against open-world semantics.
    from llm_reasoner.knowledge.microworlds.flanagan.core import RoleSlot, Subgoal

    attainment = URIRef("http://example.org/ontology/flanagan-core#attainmentSignalledBy")
    inferred_subgoals = {s for s, _, _ in graph.triples((None, attainment, None))}
    if inferred_subgoals:
        report.notes.append(
            f"{len(inferred_subgoals)} conditions retyped as Subgoal by domain "
            "inference on attainmentSignalledBy")

    # sorted, because set iteration order varies between runs and a codec whose
    # output depends on PYTHONHASHSEED makes every downstream test flaky
    subjects = sorted(
        (s for s in set(graph.subjects()) if isinstance(s, (URIRef, BNode))),
        key=str,
    )
    for subject in subjects:
        if not list(graph.objects(subject, RDF.type)):
            continue
        meta = declared_class_for(subject)
        if meta is None:
            continue
        key = str(subject)

        if meta.level is ConceptLevel.OCCURRENCE:
            # a role individual in the old description-level sense: rewrite it
            # as an unbound slot naming the role class it will bind as
            label = str(graph.value(subject, RDFS.label) or "") or key.rsplit("#", 1)[-1]
            report.instances[key] = RoleSlot(
                label=label, binds_as=ConceptReference(iri=meta.iri, label=meta.local_name)
            )
            report.rewritten_roles += 1
            continue

        if subject in inferred_subgoals and not issubclass(meta.cls, Subgoal):
            meta = index.by_iri["http://example.org/ontology/flanagan-core#Subgoal"]
        if isinstance(meta.cls, type) and issubclass(meta.cls, Enum):
            continue
        if meta.abstract:
            report.notes.append(f"individual typed as abstract class {meta.local_name}: {key}")
            continue
        try:
            instance = meta.cls()
        except Exception as exc:  # pragma: no cover - construction is defaulted
            report.notes.append(f"could not construct {meta.local_name}: {exc}")
            continue
        if isinstance(instance, KnowledgeEntity):
            instance.label = str(graph.value(subject, RDFS.label) or "") or key.rsplit("#", 1)[-1]
        report.instances[key] = instance

    # --- pass 2: wire properties -------------------------------------------
    # sorted for the same reason pass 1 is: functional properties take the last
    # value assigned, so an unordered walk makes the result depend on hashing
    for subject, predicate, obj in sorted(graph, key=lambda t: (str(t[0]), str(t[1]), str(t[2]))):
        key = str(subject)
        instance = report.instances.get(key)
        if instance is None:
            continue
        pred = str(predicate)
        if pred in (str(RDF.type), str(RDFS.label), str(RDFS.comment)):
            continue
        legacy = _LEGACY_PREDICATES.get(pred)
        if legacy is not None:
            fname, reverse = legacy
            target = report.instances.get(str(obj))
            if reverse and target is not None:
                # the Turtle states it the other way round: assign on the object
                if index.field_for(type(target), pred) or hasattr(target, fname):
                    _assign(target, fname, instance)
                    continue
            if hasattr(instance, fname):
                value = _coerce(obj, instance, fname, index, report)
                if value is not None:
                    _assign(instance, fname, value)
                continue

        fname = index.field_for(type(instance), pred)
        if fname is None:
            if pred.startswith("http://example.org/ontology/"):
                report.unmapped_predicates.add(pred)
            continue
        value = _coerce(obj, instance, fname, index, report)
        if value is None:
            continue
        _assign(instance, fname, value)

    return report


def _coerce(obj, instance, fname: str, index: _Index, report: LoadReport):
    """Turn one RDF object into the Python value the field expects."""
    if isinstance(obj, Literal):
        return obj.toPython()
    key = str(obj)
    if key in index.enum_members:
        return index.enum_members[key]
    if key in report.instances:
        return report.instances[key]

    hints = _resolved_hints(type(instance))
    targets = _range_types(hints.get(fname))

    # A class used where an individual belongs. The Turtle does this for
    # vocabularies with a fixed, closed membership -- signal types, for
    # instance -- where naming the class was the natural way to say "this
    # kind". OWL permits the punning; Python does not, so one canonical
    # instance is minted per class and shared for the rest of the load.
    declared = index.by_iri.get(key)
    if declared is not None and any(
        isinstance(t, type) and isinstance(declared.cls, type)
        and issubclass(declared.cls, t) for t in targets
    ):
        canonical = report.instances.get("canonical:" + key)
        if canonical is None:
            canonical = declared.cls()
            if isinstance(canonical, KnowledgeEntity):
                canonical.label = declared.local_name
            report.instances["canonical:" + key] = canonical
            report.notes.append(
                f"minted a canonical instance for class-as-individual: "
                f"{declared.local_name}")
        return canonical

    # not something we built: a reference into a foreign ontology
    if any(t is ConceptReference for t in targets):
        return ConceptReference(iri=key, label=key.rsplit("#", 1)[-1])
    return None


def _assign(instance, fname: str, value) -> None:
    current = getattr(instance, fname, None)
    if isinstance(current, list):
        if value not in current:
            current.append(value)
    else:
        setattr(instance, fname, value)


def structures_in(report: LoadReport) -> list[Any]:
    """The phase structures found, richest first.

    A file may hold more than one -- the negative fixture carries a second,
    near-empty structure to isolate a check that the main one would otherwise
    satisfy. Ordering by content means ``structures_in(...)[0]`` is the
    hypothesis under test rather than whichever happened to be built first.
    """
    from llm_reasoner.knowledge.microworlds.flanagan.core import ActionPhaseStructure

    found = [v for v in report.instances.values() if isinstance(v, ActionPhaseStructure)]
    return sorted(found, key=lambda s: len(s.defines_phase), reverse=True)


__all__ = ["LoadReport", "load_hypothesis", "structures_in"]