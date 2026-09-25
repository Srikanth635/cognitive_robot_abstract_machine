"""Which microworld modules an action needs.

WHY THIS IS KNOWLEDGE AND NOT CONFIGURATION
-------------------------------------------
A lookup table maps the actions someone already thought of and fails silently on
the first one they did not. What generalises is the REASON a module is needed:
pouring needs the prehensile vocabulary because pouring involves grasping an
object, and any other action involving a grasp needs it for the same reason.
Naming that reason -- a capability -- lets an unseen action inherit the right
modules from its placement in the task hierarchy rather than from an entry
someone remembered to add.

WHY SUBSUMPTION ALONE IS NOT ENOUGH
-----------------------------------
SOMA's task hierarchy is carved by outcome, not by sensorimotor requirement:
``soma:Pouring`` sits under ``soma:Actuating`` while ``soma:Grasping`` sits under
``soma:Manipulating``, so the two share no ancestor below ``soma:PhysicalTask``
even though pouring presupposes a grasp. Selection therefore cannot ride on task
subsumption alone.

SOMA is kept for task IDENTITY -- no parallel hierarchy is minted -- and capability
is added as the join term between what a task presupposes and what a module
provides. Subsumption still does real work: a profile stated once for
``soma:Pouring`` covers ``soma:PouringInto`` and any locally declared subclass
without a new entry.

WHY MODULES ARE NEVER SELECTED BY NAME
--------------------------------------
A module is selected because it provides a required capability. That indirection
is what allows the prehensile half of the pouring module to be extracted into its
own module later without touching a single profile: the module's entry loses one
capability and gains a dependency, and nothing else changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Iterable, Mapping, Optional, Sequence

from llm_reasoner.exceptions import LLMReasonerError
from llm_reasoner.knowledge.ontological.decorators import ConceptReference

SOMA = "http://www.ease-crc.org/ont/SOMA.owl#"


class Capability(StrEnum):
    """A kind of sensorimotor competence an action presupposes and a module
    supplies vocabulary for.

    Deliberately coarse. Capabilities are a handful of families, not one per
    action, because their purpose is to let unseen actions reuse existing
    modules -- a capability per action would just be the lookup table again.
    """

    PHASE_STRUCTURING = "phase_structuring"
    """The action-neutral apparatus: phases, subgoals, condition requirements,
    control points, internal models, the parameter trichotomy. Required by every
    action without exception."""

    OBJECT_DYNAMICS = "object_dynamics"
    """Object and limb dynamics, efference-copy prediction, coordinate frames,
    sensorimotor memories, and representation-purpose separation."""

    PREHENSILE_MANIPULATION = "prehensile_manipulation"
    """Grasping and holding an object against gravity: contact and
    force-dynamic events, grip-load coupling, slip and liftoff thresholds,
    grasp roles."""

    SUBSTANCE_TRANSFER = "substance_transfer"
    """Moving a substance between vessels or surfaces under continuous flow
    regulation: flow events, source and target roles, tilt and volume
    parameters, viscosity and fill-level estimates."""


@dataclass(frozen=True)
class MicroworldModule:
    """A loadable microworld, described by what it provides and depends on."""

    name: str
    """Import path of the package, which is what the loader resolves and what
    ``validate`` scopes by."""

    version: str
    """Feeds the hypothesis cache key: a bump on any selected module marks
    hypotheses generated under it stale."""

    provides: frozenset[Capability]
    depends_on: frozenset[str] = frozenset()
    mandatory: bool = False
    """Loaded regardless of the resolved profile. The fallback path depends on
    this: an action with no matching profile still gets the mandatory set."""

    note: str = ""


@dataclass(frozen=True)
class TaskProfile:
    """The claim that a kind of task presupposes a set of capabilities.

    A defeasible claim about what an action involves, not a definition -- which
    is why it is a separate object that can be revised, rather than an attribute
    of the task itself.
    """

    task: ConceptReference
    """Stated at the most general level that holds, so specialisations are
    covered by subsumption and need no entry of their own."""

    requires: frozenset[Capability]
    is_fallback: bool = False

    @property
    def specificity(self) -> int:
        """:return: How deep in the task hierarchy this profile sits.

        Used to pick the most specific matching profile. Derived from the
        materialised ancestors the reference already carries, so no reasoner
        runs at selection time.
        """
        return len(self.task.ancestors)


@dataclass(frozen=True)
class ModuleSelection:
    """The outcome of selecting modules for one action."""

    task: ConceptReference
    profile: TaskProfile
    modules: tuple[MicroworldModule, ...]

    @property
    def degraded(self) -> bool:
        """:return: Whether only the fallback profile matched.

        Worth propagating onto the hypothesis: one generated under a fallback
        profile used action-neutral vocabulary and free-text labels for phases
        with no named class, which is valid but weaker evidence than one
        generated against an authored profile.
        """
        return self.profile.is_fallback

    @property
    def module_names(self) -> tuple[str, ...]:
        return tuple(m.name for m in self.modules)

    @property
    def module_versions(self) -> Mapping[str, str]:
        """:return: The versions to record in a hypothesis's provenance."""
        return {m.name: m.version for m in self.modules}

    @property
    def capabilities(self) -> frozenset[Capability]:
        if not self.modules:
            return frozenset()
        return frozenset().union(*(m.provides for m in self.modules))


class AmbiguousProfile(LLMReasonerError):
    """Two profiles match a task equally specifically.

    An authoring error rather than something to break silently: whichever way a
    tie is resolved, one of the two profiles is wrong about the action.
    """


# %% catalogue

FLANAGAN_CORE = MicroworldModule(
    name="llm_reasoner.knowledge.microworlds.flanagan.core",
    version="0.3.0",
    provides=frozenset({Capability.PHASE_STRUCTURING}),
    mandatory=True,
    note="Action-neutral. Mandatory, so it is loaded even under the fallback "
    "profile, which is what lets an unknown action still generate.",
)

FLANAGAN_OBJECT_DYNAMICS = MicroworldModule(
    name="llm_reasoner.knowledge.microworlds.flanagan.object_dynamics",
    version="0.1.0",
    provides=frozenset({Capability.OBJECT_DYNAMICS}),
    depends_on=frozenset({FLANAGAN_CORE.name}),
    note="Action-neutral represented dynamics used by prehensile manipulation.",
)


FLANAGAN_PREHENSION = MicroworldModule(
    name="llm_reasoner.knowledge.microworlds.flanagan.prehension",
    version="0.1.0",
    provides=frozenset({Capability.PREHENSILE_MANIPULATION}),
    depends_on=frozenset({FLANAGAN_OBJECT_DYNAMICS.name}),
    note="Action-neutral contact, grasp, lifting, replacement, and release control.",
)


FLANAGAN_POURING = MicroworldModule(
    name="llm_reasoner.knowledge.microworlds.flanagan.pouring",
    version="0.4.0",
    provides=frozenset({Capability.SUBSTANCE_TRANSFER}),
    depends_on=frozenset({FLANAGAN_PREHENSION.name}),
    note="Substance-transfer extension composed with prehensile manipulation.",
)

MODULES: tuple[MicroworldModule, ...] = (
    FLANAGAN_CORE,
    FLANAGAN_OBJECT_DYNAMICS,
    FLANAGAN_PREHENSION,
    FLANAGAN_POURING,
)

POURING_PROFILE = TaskProfile(
    task=ConceptReference(
        iri=SOMA + "Pouring",
        label="Pouring",
        ancestors=frozenset({SOMA + "Actuating", SOMA + "PhysicalTask"}),
    ),
    requires=frozenset(
        {
            Capability.PHASE_STRUCTURING,
            Capability.PREHENSILE_MANIPULATION,
            Capability.OBJECT_DYNAMICS,
            Capability.SUBSTANCE_TRANSFER,
        }
    ),
)

GRASPING_PROFILE = TaskProfile(
    task=ConceptReference(
        iri=SOMA + "Grasping",
        label="Grasping",
        ancestors=frozenset({SOMA + "Manipulating", SOMA + "PhysicalTask"}),
    ),
    requires=frozenset(
        {
            Capability.PHASE_STRUCTURING,
            Capability.PREHENSILE_MANIPULATION,
            Capability.OBJECT_DYNAMICS,
        }
    ),
)

DEFAULT_PROFILE = TaskProfile(
    task=ConceptReference(iri=SOMA + "PhysicalTask", label="PhysicalTask"),
    requires=frozenset({Capability.PHASE_STRUCTURING}),
    is_fallback=True,
)

PROFILES: tuple[TaskProfile, ...] = (
    POURING_PROFILE,
    GRASPING_PROFILE,
    DEFAULT_PROFILE,
)


# %% selection


def matching_profiles(
    task: ConceptReference, profiles: Sequence[TaskProfile] = PROFILES
) -> list[TaskProfile]:
    """:return: Every profile whose task subsumes this one, most specific first."""
    matched = [p for p in profiles if task.is_subsumed_by(p.task)]
    return sorted(matched, key=lambda p: p.specificity, reverse=True)


def resolve_profile(
    task: ConceptReference, profiles: Sequence[TaskProfile] = PROFILES
) -> TaskProfile:
    """Choose the profile that speaks for a task.

    :raises AmbiguousProfile: when two equally specific non-fallback profiles
        match, which is an authoring error rather than something to resolve
        arbitrarily.
    """
    matched = matching_profiles(task, profiles)
    if not matched:
        return next((p for p in profiles if p.is_fallback), DEFAULT_PROFILE)
    best = matched[0]
    if best.is_fallback:
        return best
    tied = [
        p
        for p in matched[1:]
        if p.specificity == best.specificity and not p.is_fallback
    ]
    if tied:
        raise AmbiguousProfile(
            f"{task} matches {best.task} and "
            f"{', '.join(str(p.task) for p in tied)} equally specifically"
        )
    return best


def select_modules(
    task: ConceptReference,
    profiles: Sequence[TaskProfile] = PROFILES,
    modules: Sequence[MicroworldModule] = MODULES,
) -> ModuleSelection:
    """Work out which microworld modules an action needs.

    The procedure, stated so it is auditable rather than only executable:

    1. find every profile whose task subsumes this one;
    2. take the most specific, falling back when nothing else matches;
    3. select every mandatory module, plus every module providing a required
       capability;
    4. close under declared dependencies.

    Step 3 adds no module for a capability that was not required, but does not
    try to minimise across modules with overlapping capabilities -- with a
    catalogue this size that would be premature, and a genuine overlap is worth
    a human deciding rather than a loader breaking the tie.
    """
    profile = resolve_profile(task, profiles)
    by_name = {m.name: m for m in modules}

    selected: dict[str, MicroworldModule] = {
        m.name: m for m in modules if m.mandatory or (m.provides & profile.requires)
    }

    frontier = list(selected.values())
    while frontier:
        module = frontier.pop()
        for dependency in module.depends_on:
            if dependency in selected or dependency not in by_name:
                continue
            selected[dependency] = by_name[dependency]
            frontier.append(by_name[dependency])

    ordered = tuple(m for m in modules if m.name in selected)
    return ModuleSelection(task=task, profile=profile, modules=ordered)


def current_versions(
    modules: Sequence[MicroworldModule] = MODULES,
) -> dict[str, str]:
    """:return: Every module's version, for testing a record's staleness."""
    return {m.name: m.version for m in modules}


def task_reference(
    iri: str, ancestors: Iterable[str] = (), label: Optional[str] = None
) -> ConceptReference:
    """Build a task reference with its ancestors materialised.

    Convenience for declaring an action the catalogue has never seen: place it
    under an existing task and it inherits that task's profile without any new
    entry here.
    """
    return ConceptReference(
        iri=iri,
        label=label or iri.rsplit("#", 1)[-1],
        ancestors=frozenset(ancestors),
    )


__all__ = [
    "AmbiguousProfile",
    "Capability",
    "DEFAULT_PROFILE",
    "FLANAGAN_CORE",
    "FLANAGAN_POURING",
    "FLANAGAN_OBJECT_DYNAMICS",
    "FLANAGAN_PREHENSION",
    "GRASPING_PROFILE",
    "MODULES",
    "PROFILES",
    "POURING_PROFILE",
    "MicroworldModule",
    "ModuleSelection",
    "TaskProfile",
    "current_versions",
    "matching_profiles",
    "resolve_profile",
    "select_modules",
    "task_reference",
]
