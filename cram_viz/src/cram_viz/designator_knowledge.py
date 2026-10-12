"""
A real coraplex action designator's own fields, reflected into a static slot table --
the Designator tab.

Unlike the Hypothesis Graph tab, nothing here is generated and nothing is resolved:
``coraplex.robot_plans.actions.composite.tool_based.PouringAction``'s directly declared
dataclass fields are read straight off the class (no live
:class:`~semantic_digital_twin.world_description.world.World`, no
:class:`~llm_reasoner.resolver.backend.LanguageModelBackend` call) and classified value
versus referent exactly as :mod:`llm_reasoner.resolver.reflection` classifies an open
slot on a krrood ``Match`` -- but without constructing one, since every field here is
fully open and none is bound, sortal or a live extension.

A referent slot (:attr:`~llm_reasoner.resolver.reflection.SlotKind.REFERENT`) has no
live World to draw its extension from, so it is offered illustrative stand-ins instead:
the scene's own :class:`~cram_viz.kb.BenchObject`\\ s, which are graspable things but not
real ``Body``/``Tool`` individuals.

A referent slot that fills a role of the pouring ontology is tagged with the role's
class, and resolved illustratively to the scene object whose label reads closest to the
active hypothesis's filler of that role -- still not a live resolver run.

Once the resolver has resolved one of the viewer's predefined designators
(:mod:`cram_viz.resolution`), each slot it resolved holds the value it chose instead, and
no illustrative candidates are offered for it.
"""

from __future__ import annotations

import dataclasses
import inspect
from dataclasses import dataclass, field
from typing_extensions import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

from krrood.entity_query_language.verbalization.rendering.source_documentation import (
    _attribute_docstrings,
)
from llm_reasoner.resolver.reflection import (
    SlotKind,
    is_generatable,
    module_namespace_of,
    raw_annotation_of,
    type_display_name,
    union_alternatives,
)

from coraplex.robot_plans.actions.composite.tool_based import PouringAction
from llm_reasoner.knowledge.access import (
    core,
    HypothesisView,
    pouring,
)

if TYPE_CHECKING:
    from cram_viz.kb import BenchObject
    from cram_viz.resolution.resolver import ResolvedDesignator


# %% slot table


@dataclass
class IllustrativeCandidate:
    """One scene object offered as a stand-in for a referent slot's real domain."""

    identifier: str
    """The scene object's identifier, used as the graph node id."""

    label: str
    """Human-readable label shown on the node."""

    resolved: bool = False
    """
    Whether this is the candidate the active hypothesis names for the slot's role.
    Still not a live resolver run -- just which of the illustrative stand-ins reads
    like the role's filler.
    """


@dataclass
class DesignatorSlot:
    """One field the designator declares, classified and described for display."""

    name: str
    """The field name on the designator."""

    type_name: str
    """Display name of the declared type."""

    kind: SlotKind
    """Whether the field is filled by generation or by selecting a real individual."""

    docstring: Optional[str]
    """The field's own docstring, written directly below it in the designator."""

    default: Any
    """The field's declared default value, when it has one."""

    has_default: bool
    """Whether the field declares a default at all."""

    candidates: List[IllustrativeCandidate] = field(default_factory=list)
    """Illustrative stand-in candidates for a referent slot; empty for a value slot."""

    role: Optional[type[core.ManipulationRole]] = None
    """
    The ontology role this slot fills, when it fills one.
    """

    resolved_value: Optional[str] = None
    """
    The value the resolver chose for this slot, written for reading, when a resolution
    chose one.
    """

    @property
    def resolved_candidate(self) -> Optional[IllustrativeCandidate]:
        """:return: This slot's candidate resolved for the active hypothesis, if any."""
        return next((c for c in self.candidates if c.resolved), None)

    @property
    def lisp_keyword(self) -> str:
        """:return: The field name as a Lisp keyword, kebab-cased."""
        return ":" + self.name.replace("_", "-")

    @property
    def lisp_value(self) -> str:
        """
        :return: The slot's value rendered as Lisp -- the resolved candidate's
            identifier if it has one, otherwise a logic variable for anything left
            open (a referent slot, or a value slot with no default), otherwise the
            conventional default.
        """
        if self.resolved_value is not None:
            return self.resolved_value
        resolved = self.resolved_candidate
        if resolved is not None:
            return resolved.identifier
        if self.kind is SlotKind.REFERENT or not self.has_default:
            return "?" + self.name.replace("_", "-")
        return _lisp_literal(self.default)


@dataclass
class DesignatorKnowledge:
    """A designator class reflected into its slot table."""

    designator_name: str
    """The reflected designator's class name."""

    docstring: Optional[str]
    """The designator class's own docstring."""

    slots: List[DesignatorSlot]
    """Every directly-declared field of the designator, in declaration order."""

    resolved_for_instruction: Optional[str] = None
    """
    The instruction of the hypothesis a slot's :attr:`~DesignatorSlot.resolved_candidate`
    was chosen for, or ``None`` when there was none.
    """

    resolution: Optional[ResolvedDesignator] = None
    """
    The resolver's resolution of the designator, when one was made.
    """


# %% Lisp rendering


def _lisp_literal(value: Any) -> str:
    """
    :param value: A slot's default value.
    :return: Its Lisp literal spelling.
    """
    if value is None:
        return "NIL"
    if isinstance(value, bool):
        return "T" if value else "NIL"
    if isinstance(value, str):
        return '"%s"' % value
    return str(value)


#: PouringAction's own field names that fill a role of the pouring ontology
_ROLE_BY_FIELD_NAME = {
    "source_container": pouring.SourceContainer,
    "target_container": pouring.TargetReceptacle,
}


# %% reflection


def _direct_field_names(designator_type: type) -> List[str]:
    """
    :param designator_type: The designator class to reflect.
    :return: Names of the fields the class declares itself, excluding whatever it
        inherits -- the base classes' fields are execution context (world, plan,
        robot), not parameters a caller fills in.
    """
    return list(designator_type.__dict__.get("__annotations__", {}).keys())


def _default_for(declared_field: dataclasses.Field) -> tuple:
    """
    :param declared_field: The dataclass field to read a default from.
    :return: The default value and whether one is declared.
    """
    if declared_field.default is not dataclasses.MISSING:
        return declared_field.default, True
    if declared_field.default_factory is not dataclasses.MISSING:
        return declared_field.default_factory(), True
    return None, False


def resolved_field_type(owner: type, attribute_name: str) -> Any:
    """
    Resolve one field's annotation to a real type, without evaluating any other
    field's annotation on the class or its bases.

    ``typing.get_type_hints`` resolves every annotation reachable through
    ``owner.__mro__``, including base-class fields this module never reflects on
    (``ActionDescription``/``Designator``'s ``world``, ``plan``, ...) -- some of
    those are only importable under ``TYPE_CHECKING`` and raise ``NameError`` the
    moment anything forces them to resolve. Evaluating just this one field's
    annotation against its own declaring module avoids that entirely.

    :param owner: The class declaring the attribute.
    :param attribute_name: The field name on the owner.
    :return: The resolved type.
    """
    raw_annotation = raw_annotation_of(owner, attribute_name)
    if not isinstance(raw_annotation, str):
        return raw_annotation
    return eval(raw_annotation, module_namespace_of(owner))  # noqa: S307


def _best_matching_candidate(
    concrete_name: str, candidates: Sequence[IllustrativeCandidate]
) -> Optional[IllustrativeCandidate]:
    """
    Pick the scene object whose label reads closest to a role filler's
    filler name (e.g. matching "Breakfast Cereal" to "Cereal Box" through the shared
    word "cereal"), rather than naming a specific object outright -- the same match
    would still make sense if the scene's objects changed.

    :param concrete_name: The label of the role's filler.
    :param candidates: The referent slot's candidates.
    :return: The best-scoring candidate, or ``None`` if none share a word with it.
    """
    if not candidates:
        return None
    target_words = set(concrete_name.lower().split())
    scored = [
        (len(target_words & set(candidate.label.lower().split())), candidate)
        for candidate in candidates
    ]
    best_score, best_candidate = max(scored, key=lambda scored_item: scored_item[0])
    return best_candidate if best_score > 0 else None


def _candidates_for_role(
    role: Optional[type[core.ManipulationRole]],
    view: Optional[HypothesisView],
    objects: Sequence[BenchObject],
) -> List[IllustrativeCandidate]:
    """
    Build one referent slot's candidates, marking resolved the one reading like the
    active hypothesis's filler of the slot's role.

    :param role: The slot's role, or ``None`` for a slot with none.
    :param view: The active hypothesis, or ``None`` when there is none.
    :param objects: The scene's bench objects.
    :return: The slot's candidates.
    """
    candidates = [
        IllustrativeCandidate(identifier=bench_object.name, label=bench_object.label)
        for bench_object in objects
        if bench_object.kind == "object"
    ]
    if role is None or view is None:
        return candidates
    fillers = [filler for filler in view.root.defines_role if type(filler) is role]
    if not fillers or not fillers[0].label:
        return candidates
    resolved = _best_matching_candidate(fillers[0].label, candidates)
    if resolved is not None:
        resolved.resolved = True
    return candidates


def _classify(declared_type: Any, alternatives: Optional[List[type]]) -> SlotKind:
    """
    :param declared_type: The field's annotated type.
    :param alternatives: The union branches, when the annotation is a union.
    :return: ``VALUE`` when a value of this type may be generated outright,
        ``REFERENT`` when it must be a real individual.
    """
    if alternatives is not None:
        if all(is_generatable(alternative) for alternative in alternatives):
            return SlotKind.VALUE
        return SlotKind.REFERENT
    if is_generatable(declared_type):
        return SlotKind.VALUE
    return SlotKind.REFERENT


def reflect_pouring_action(
    objects: Sequence[BenchObject],
    view: Optional[HypothesisView],
    resolution: Optional[ResolvedDesignator] = None,
) -> DesignatorKnowledge:
    """
    Reflect :class:`PouringAction` into a slot table.

    :param objects: The scene's bench objects, offered as illustrative candidates for
        referent slots the resolution does not resolve.
    :param view: The active hypothesis, whose role fillers resolve the referent
        slots illustratively, or ``None`` when there is none.
    :param resolution: The resolver's resolution, whose first action's values the
        slots it resolved hold, or ``None`` when there is none.
    :return: The reflected designator.
    """
    chosen: Dict[str, str] = (
        {}
        if resolution is None or not resolution.actions
        else {slot.name: slot.value for slot in resolution.actions[0].slots}
    )
    designator_type = PouringAction
    docstrings = inspect.getdoc(designator_type)
    fields_by_name = {f.name: f for f in dataclasses.fields(designator_type)}
    field_docstrings = _attribute_docstrings(designator_type)
    slots = []
    for name in _direct_field_names(designator_type):
        declared_type = resolved_field_type(designator_type, name)
        alternatives = union_alternatives(declared_type)
        kind = _classify(declared_type, alternatives)
        default, has_default = _default_for(fields_by_name[name])
        role = _ROLE_BY_FIELD_NAME.get(name)
        candidates = (
            _candidates_for_role(role, view, objects)
            if kind is SlotKind.REFERENT and name not in chosen
            else []
        )
        slots.append(
            DesignatorSlot(
                name=name,
                type_name=type_display_name(declared_type, alternatives),
                kind=kind,
                docstring=field_docstrings.get(name),
                default=default,
                has_default=has_default,
                candidates=candidates,
                role=role,
                resolved_value=chosen.get(name),
            )
        )
    return DesignatorKnowledge(
        designator_name=designator_type.__name__,
        docstring=docstrings,
        slots=slots,
        resolved_for_instruction=view.record.instruction if view is not None else None,
        resolution=resolution,
    )
