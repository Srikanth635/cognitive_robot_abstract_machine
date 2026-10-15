"""
The prompts asking a model for the vocabulary of a perspective: a system prompt stating
the role, the inputs and the rules, a request giving the perspective, the action and the
terms to build on, and a repair returning a rejected answer with the rules it broke.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from typing_extensions import Any, Dict, List, Sequence

from llm_reasoner.knowledge.vocabulary_proposal.catalogue import TermCatalogue
from llm_reasoner.knowledge.vocabulary_proposal.checks import Problem
from llm_reasoner.knowledge.vocabulary_proposal.perspective import Perspective
from llm_reasoner.prompt_structure import (
    SECTION_SEPARATOR,
    PromptBlock,
    PromptRule,
    render_blocks,
)

PROPOSAL_SCHEMA: Dict[str, Any] = json.loads(
    (Path(__file__).parent / "proposal_schema.json").read_text()
)
"""
The JSON schema an answer follows.
"""

JSON_INDENT = 2
"""
The indentation of the data the model reads, except the long list of linked terms.
"""

COMPACT_SEPARATORS = (",", ":")
"""
How the long list of linked terms is written, without spaces.
"""


class ProposalSection(StrEnum):
    """
    The sections of the proposal prompts.
    """

    ROLE = "role"
    INPUTS = "inputs"
    RULES = "rules"
    PERSPECTIVE = "perspective"
    ACTION = "action"
    CORE = "core"
    LINKED_TERMS = "linked_terms"
    OUTPUT = "output"
    REJECTED_ANSWER = "rejected_answer"
    PROBLEMS = "problems"


class ProposalAttribute(StrEnum):
    """
    The attributes of a section's tag.
    """

    NAME = "name"


class ProposalRuleName(StrEnum):
    """
    The rules a proposal follows, each stated once.
    """

    SCOPE = "scope"
    SPECIALISE = "specialise"
    REUSE = "reuse"
    ATTACH = "attach"
    TERMS = "terms"
    KINDS = "kinds"
    DEFINE = "define"
    NAMES = "names"
    QUANTITIES = "quantities"
    CONCEPTS = "concepts"
    WORLD = "world"
    SOURCES = "sources"
    ANSWER_FORMAT = "answer_format"
    REPAIR = "repair"


ROLE = (
    "You design the vocabulary of one perspective on actions, as OWL classes, object "
    "properties and value partitions that a knowledge base about actions is written in."
)
"""
What the model is asked to be.
"""

INPUTS: Dict[ProposalSection, str] = {
    ProposalSection.PERSPECTIVE: "what the perspective describes, and what its "
    "vocabulary must let a knowledge base say",
    ProposalSection.ACTION: "the action the vocabulary is first used for, as its "
    "CRAM-AICOR concept, and the action core's frame of it",
    ProposalSection.CORE: "the action core's terms (JSON): the frame, slots, objective, "
    "consequence and parameter dimension every perspective describes an action in terms of",
    ProposalSection.LINKED_TERMS: "the terms of CRAM-AICOR (aicor:) and DUL (dul:) (JSON), "
    "which the vocabulary specialises and reuses",
    ProposalSection.OUTPUT: "the JSON schema of the answer",
    ProposalSection.REJECTED_ANSWER: "in a repair, the answer that broke the rules",
    ProposalSection.PROBLEMS: "in a repair, the rules it broke, each with the term (JSON)",
}
"""
What each section of a request holds.
"""

RULES: List[PromptRule] = [
    PromptRule(
        ProposalRuleName.SCOPE,
        "Declare what the perspective's description asks for, for actions in general, "
        "and nothing the action core or another perspective covers.",
    ),
    PromptRule(
        ProposalRuleName.SPECIALISE,
        "Every class specialises a class of the action core, CRAM-AICOR or DUL, directly "
        "or through classes of the proposal.",
    ),
    PromptRule(
        ProposalRuleName.REUSE,
        "Use an existing term of the core, CRAM-AICOR or DUL wherever one means what is "
        "needed, and declare a new term only for what none of them has.",
    ),
    PromptRule(
        ProposalRuleName.ATTACH,
        "Describe the action in terms of the action core: refer to its frames, slots, "
        "objectives, consequences and parameter dimensions (core:), never to the terms "
        "of another perspective.",
    ),
    PromptRule(
        ProposalRuleName.TERMS,
        "Name a term of the core, CRAM-AICOR or DUL with its prefix (core:, aicor:, "
        "dul:) exactly as listed; name a term of the proposal by its name alone.",
    ),
    PromptRule(
        ProposalRuleName.KINDS,
        "A parent, domain, range, and the filler of a 'some' or 'only' restriction are "
        "classes; the filler of a 'value' restriction is an individual, such as a member "
        "of a value partition; a restricted property and a property's parent are "
        "properties.",
    ),
    PromptRule(
        ProposalRuleName.DEFINE,
        "Give every class, property, value partition and member a definition of one or "
        "two sentences saying what it is.",
    ),
    PromptRule(
        ProposalRuleName.NAMES,
        "Name a class, a value partition or a member as one word starting in capitals, "
        "such as FillerExpectation, and a property as one word starting in lower case, "
        "such as expectsOfSlot.",
    ),
    PromptRule(
        ProposalRuleName.QUANTITIES,
        "Declare no numeric property: a quantity is one of the core's parameter "
        "dimensions, which the vocabulary refers to.",
    ),
    PromptRule(
        ProposalRuleName.CONCEPTS,
        "A property whose values are CRAM-AICOR concepts, such as an action category "
        "(aicor:Grasping) or a state type (aicor:ContainedState), names them by their "
        "IRI as the core's framesAction and aimsAtState do: give it no range, and say in "
        "its definition which concepts it names.",
    ),
    PromptRule(
        ProposalRuleName.WORLD,
        "The vocabulary is about kinds of action: it names no object, place, robot or "
        "episode, and states what a filler has or affords as a condition a world can "
        "check rather than as the class the filler belongs to.",
    ),
    PromptRule(
        ProposalRuleName.SOURCES,
        "When a term is taken from FrameNet, an image schema or another published "
        "resource, give its IRI in see_also.",
    ),
    PromptRule(
        ProposalRuleName.ANSWER_FORMAT,
        "Answer with one JSON object following the output schema.",
    ),
    PromptRule(
        ProposalRuleName.REPAIR,
        "In a repair, answer the whole vocabulary again, correcting every problem listed "
        "and keeping what was right.",
    ),
]
"""
The rules a proposal follows.
"""


@dataclass(frozen=True)
class ProposalPrompt:
    """
    Writes the prompts asking for the vocabulary of a perspective.
    """

    catalogue: TermCatalogue
    """
    The terms the vocabulary builds on.
    """

    def system(self) -> str:
        """
        :return: The system prompt: the role, what the inputs hold and the rules.
        """
        return render_blocks(
            [
                PromptBlock(ProposalSection.ROLE, ROLE),
                PromptBlock(
                    ProposalSection.INPUTS,
                    "\n".join(
                        f"- {section}: {text}" for section, text in INPUTS.items()
                    ),
                ),
                PromptBlock(
                    ProposalSection.RULES, "\n".join(rule.render() for rule in RULES)
                ),
            ]
        )

    def request(self, perspective: Perspective, action: str, frame: str) -> str:
        """
        :param perspective: The perspective whose vocabulary is asked for.
        :param action: The CRAM-AICOR concept of the action, such as ``aicor:Pouring``.
        :param frame: The core's frame of the action, such as ``core:PouringFrame``.
        :return: The request.
        """
        return render_blocks(
            [
                PromptBlock(
                    ProposalSection.PERSPECTIVE,
                    perspective.description,
                    {ProposalAttribute.NAME: perspective.value},
                ),
                PromptBlock(
                    ProposalSection.ACTION,
                    json.dumps({"action": action, "frame": frame}, indent=JSON_INDENT),
                ),
                PromptBlock(
                    ProposalSection.CORE,
                    json.dumps(
                        [entry.to_json() for entry in self.catalogue.core_terms()],
                        indent=JSON_INDENT,
                    ),
                ),
                PromptBlock(
                    ProposalSection.LINKED_TERMS,
                    json.dumps(
                        [entry.to_json() for entry in self.catalogue.linked_terms()],
                        separators=COMPACT_SEPARATORS,
                    ),
                ),
                PromptBlock(
                    ProposalSection.OUTPUT,
                    json.dumps(PROPOSAL_SCHEMA, indent=JSON_INDENT),
                ),
            ]
        )

    @staticmethod
    def repair(request: str, answer: str, problems: Sequence[Problem]) -> str:
        """
        :return: The request again, with the rejected answer and the rules it broke.
        """
        return (
            request
            + SECTION_SEPARATOR
            + render_blocks(
                [
                    PromptBlock(ProposalSection.REJECTED_ANSWER, answer),
                    PromptBlock(
                        ProposalSection.PROBLEMS,
                        json.dumps(
                            [problem.to_json() for problem in problems],
                            indent=JSON_INDENT,
                        ),
                    ),
                ]
            )
        )
