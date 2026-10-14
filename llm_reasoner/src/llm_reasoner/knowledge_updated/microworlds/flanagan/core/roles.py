"""
The participants of a manipulation.

A hypothesis describes what kind of role will be borne, never who bears it. That is why
there are two things here: a :class:`RoleSlot`, an unbound placeholder that appears in
hypotheses, and the :class:`ManipulationRole` classes, real roles that only exist once a
resolver has bound a slot to an entity of the world.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from krrood.patterns.role import Role
from krrood.symbol_graph.symbol_graph import Symbol

from llm_reasoner.knowledge_updated.microworlds.flanagan.alignments import (
    DUL,
    SOMA,
    subclass_of,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.ontology import CORE
from llm_reasoner.knowledge_updated.ontological.entities import KnowledgeEntity
from llm_reasoner.knowledge_updated.ontological.properties import attribute, reference
from llm_reasoner.knowledge_updated.ontological.references import ConceptReference
from llm_reasoner.knowledge_updated.ontological.vocabulary import (
    EvidenceStatus,
    ConceptLevel,
    Rigidity,
)

# %% bound roles


@CORE.concept(
    rigidity=Rigidity.ANTI_RIGID,
    level=ConceptLevel.OCCURRENCE,
    role_taker=Symbol,
    alignments=(subclass_of(DUL, "Role"),),
    abstract=True,
    disjoint_siblings=True,
    status=EvidenceStatus.ENGINEERING_EXTENSION,
)
@dataclass(eq=False)
class ManipulationRole(Role[Symbol]):
    """
    A participant role borne by an actual entity of the world.

    Instances exist only after resolution has bound a slot to something. Holding the
    role is not essential to the entity, so the role is a separate object that composes
    the entity.
    """


@CORE.concept(
    rigidity=Rigidity.ANTI_RIGID,
    level=ConceptLevel.OCCURRENCE,
    role_taker=Symbol,
    alignments=(subclass_of(SOMA, "Patient"),),
    status=EvidenceStatus.SUPPORTED_INTERPRETATION,
)
@dataclass(eq=False)
class Manipulandum(ManipulationRole):
    """
    The object whose state the manipulation is about.
    """


@CORE.concept(
    rigidity=Rigidity.ANTI_RIGID,
    level=ConceptLevel.OCCURRENCE,
    role_taker=Symbol,
    alignments=(subclass_of(SOMA, "AgentRole"),),
    status=EvidenceStatus.EMPIRICAL_FINDING,
)
@dataclass(eq=False)
class Actor(ManipulationRole):
    """
    The agent performing the manipulation.

    An agent is never the object it manipulates, and it is distinct from the effector it
    acts through.
    """


# %% unbound slots


@CORE.concept(status=EvidenceStatus.ENGINEERING_EXTENSION)
@dataclass(eq=False)
class RoleSlot(KnowledgeEntity):
    """
    A participant slot of a phase structure that nothing fills yet.

    It says which kind of role will be borne, and by how many entities, without naming
    any. Resolution binds slots to world entities, which is where the corresponding role
    objects come into existence.
    """

    binds_as: Optional[ConceptReference] = reference(
        functional=True, minimum_count=1, concept_root=ManipulationRole
    )
    """
    The role concept an entity becomes once it fills the slot.

    A reference and not a class, so slots persist and serialise like every other
    fragment.
    """

    min_bindings: int = attribute(functional=True, default=1)
    """
    How many world entities must fill the slot: one for a manipulandum, more where a
    role is collective.
    """
