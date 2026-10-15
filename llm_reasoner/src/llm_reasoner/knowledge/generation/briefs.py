"""
What a generation asks the model to describe: the phases of predictive control for a
class of the predictive manipulation modules, or what an action perspective module
describes for a class of the action core or of a perspective.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from typing_extensions import Tuple

from llm_reasoner.knowledge.ontologies import ActionPerspectiveFile
from llm_reasoner.knowledge.ontologies.terms import DescriptionEntity
from llm_reasoner.knowledge.ontologies.vocabulary import Vocabulary
from llm_reasoner.prompt_structure import PromptRule


class BriefRuleName(StrEnum):
    """
    The rules about what an answer describes, each named for what it governs.
    """

    CONTENT = "content"
    ROLES = "roles"
    QUANTITIES = "quantities"
    SLOTS = "slots"
    GIVEN = "given"


@dataclass(frozen=True)
class GenerationBrief:
    """
    What the model is asked to describe, and the rules about what its answer describes.
    """

    role: str
    """
    What the model is asked to do.
    """

    rules: Tuple[PromptRule, ...]
    """
    The rules about what the answer describes, stated before the rules about its form.
    """

    @classmethod
    def for_root(cls, root: type[DescriptionEntity]) -> GenerationBrief:
        """
        :return: The brief of a generation rooted in a class: an action perspective
            module's for a class of one, the predictive manipulation brief otherwise.
        """
        module_file = Vocabulary().module_file_of(root)
        if isinstance(module_file, ActionPerspectiveFile):
            return cls.of_action_perspective(module_file)
        return PREDICTIVE_MANIPULATION_BRIEF

    @classmethod
    def of_action_perspective(
        cls, module_file: ActionPerspectiveFile
    ) -> GenerationBrief:
        """
        :return: The brief of a generation rooted in a class of an action perspective
            module: its description, the rule that slots stand for kinds of
            participant, and for a perspective, that the core it describes is given.
        """
        return cls(
            role=ACTION_PERSPECTIVE_ROLE + "\n\n" + module_file.description,
            rules=(
                ACTION_CORE_RULES
                if module_file is ActionPerspectiveFile.ACTION_CORE
                else PERSPECTIVE_RULES
            ),
        )


PREDICTIVE_MANIPULATION_BRIEF = GenerationBrief(
    role=(
        "Describe how a manipulation action is organised, as one structure of typed "
        "fragments drawn from an ontology of predictive manipulation."
    ),
    rules=(
        PromptRule(
            BriefRuleName.CONTENT,
            "describe the phases of the action, the subgoal each phase pursues and the "
            "events that end it, the participants by their roles, the forces and "
            "conditions involved, what the controller predicts at each event, how it "
            "compares and corrects, and the quantities.",
        ),
        PromptRule(
            BriefRuleName.ROLES,
            'participants are roles such as "the source container", never particular '
            "objects.",
        ),
        PromptRule(
            BriefRuleName.QUANTITIES,
            "quantities are ranges with units, never single measurements you have not "
            "made.",
        ),
    ),
)
"""
The brief of a generation rooted in a class of the predictive manipulation modules.
"""

ACTION_PERSPECTIVE_ROLE = (
    "Describe a kind of action from one perspective, as typed fragments drawn from an "
    "ontology of actions linked to CRAM-AICOR and DUL. The perspective:"
)
"""
What the model is asked to do when generating for an action perspective module, before
that module's description.
"""

SLOTS_RULE = PromptRule(
    BriefRuleName.SLOTS,
    'participants are the slots of the action\'s frame, such as "the source", never '
    "particular objects, places or robots.",
)
"""
The rule that an action perspective names participants by the slots of the frame.
"""

ACTION_CORE_RULES = (SLOTS_RULE,)
"""
The rules about what an answer describes for the action core.
"""

PERSPECTIVE_RULES = (
    SLOTS_RULE,
    PromptRule(
        BriefRuleName.GIVEN,
        "the fragments in <given> are the action core this perspective describes, as "
        'written before: refer to them by their "id", and write only fragments of '
        "this perspective's own types, never a given fragment again.",
    ),
)
"""
The rules about what an answer describes for a perspective on the action core.
"""
