from __future__ import annotations

import pytest
from krrood.patterns.role import Role
from krrood.symbol_graph.symbol_graph import Symbol
from llm_reasoner.knowledge_updated.microworlds.flanagan.alignments import (
    DUL,
    SOMA,
    subclass_of,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.roles import (
    Actor,
    Manipulandum,
    ManipulationRole,
    RoleSlot,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.ontology import CORE, FLANAGAN
from llm_reasoner.knowledge_updated.ontological.documentation import (
    DocumentationReader,
)
from llm_reasoner.knowledge_updated.ontological.entities import KnowledgeEntity
from llm_reasoner.knowledge_updated.ontological.properties import PropertyMeta
from llm_reasoner.knowledge_updated.ontological.validation import (
    Check,
    OntologyValidator,
)
from llm_reasoner.knowledge_updated.ontological.vocabulary import (
    ConceptLevel,
    PropertyKind,
    Rigidity,
)

# %% bound roles


class TestBoundRoles:
    """
    A role is held by a real entity, so it exists only once a slot has been bound.
    """

    def test_manipulation_role_is_an_anti_rigid_occurrence_level_role(self) -> None:
        described = FLANAGAN.find_concept(ManipulationRole)

        assert described.module == CORE.name
        assert described.rigidity is Rigidity.ANTI_RIGID
        assert described.level is ConceptLevel.OCCURRENCE
        assert described.role_taker is Symbol
        assert described.abstract is True
        assert described.axioms.disjoint_siblings is True

    @pytest.mark.parametrize("role", [Manipulandum, Actor])
    def test_specific_roles_are_anti_rigid_occurrence_level_roles(
        self, role: type
    ) -> None:
        described = FLANAGAN.find_concept(role)

        assert issubclass(role, ManipulationRole)
        assert issubclass(role, Role)
        assert described.rigidity is Rigidity.ANTI_RIGID
        assert described.level is ConceptLevel.OCCURRENCE
        assert described.role_taker is Symbol
        assert described.abstract is False

    def test_roles_are_aligned_by_what_their_definitions_entail(self) -> None:
        assert FLANAGAN.find_concept(ManipulationRole).alignments == (
            subclass_of(DUL, "Role"),
        )
        assert FLANAGAN.find_concept(Manipulandum).alignments == (
            subclass_of(SOMA, "Patient"),
        )
        assert FLANAGAN.find_concept(Actor).alignments == (
            subclass_of(SOMA, "AgentRole"),
        )

    def test_actor_and_manipulandum_share_no_instance(self) -> None:
        report = OntologyValidator(FLANAGAN).validate(CORE.name)

        flagged = {violation.subject for violation in report.of(Check.SIBLINGS_OVERLAP)}

        assert flagged.isdisjoint({Actor.__name__, Manipulandum.__name__})


# %% unbound slots


class TestRoleSlot:
    """
    A slot names the kind of role that will be borne, without naming who bears it.
    """

    def test_slot_is_description_level_knowledge_with_an_identity(self) -> None:
        described = FLANAGAN.find_concept(RoleSlot)

        assert issubclass(RoleSlot, KnowledgeEntity)
        assert described.level is ConceptLevel.DESCRIPTION
        assert described.rigidity is Rigidity.RIGID

    def test_slot_names_the_role_concept_it_becomes(self) -> None:
        binds_as = PropertyMeta.declared_on(RoleSlot)["binds_as"]

        assert binds_as.kind is PropertyKind.REFERENCE
        assert binds_as.functional is True
        assert binds_as.minimum_count == 1
        assert binds_as.concept_root is ManipulationRole

    def test_slot_is_filled_by_one_entity_unless_the_role_is_collective(self) -> None:
        minimum_bindings = PropertyMeta.declared_on(RoleSlot)["min_bindings"]

        assert minimum_bindings.kind is PropertyKind.ATTRIBUTE
        assert minimum_bindings.functional is True
        assert RoleSlot().min_bindings == 1

    def test_slot_links_to_no_instance(self) -> None:
        kinds = {meta.kind for meta in PropertyMeta.declared_on(RoleSlot).values()}

        assert PropertyKind.RELATION not in kinds


# %% documentation


class TestRolesAreDocumented:
    """
    Every role concept and slot property says what it means.
    """

    @pytest.mark.parametrize("cls", [ManipulationRole, Manipulandum, Actor, RoleSlot])
    def test_concept_is_documented(self, cls: type) -> None:
        assert DocumentationReader().read_class(cls) is not None

    def test_slot_properties_are_documented(self) -> None:
        assert DocumentationReader().find_undocumented_fields(RoleSlot) == ()
