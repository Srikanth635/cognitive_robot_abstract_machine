"""
The rules every proposed vocabulary follows: its terms are named as the generator writes
them, defined, new, and specialise the action core, CRAM-AICOR or DUL; it refers only to
those and to its own terms, each where a term of that kind belongs; it attaches to the
action core; the vocabulary generator can write it as Python; and, when a reasoner is
given, it is consistent with the core and the linked ontologies.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum

from rdflib import OWL
from typing_extensions import Any, Dict, Iterator, List, Optional, Tuple

from llm_reasoner.knowledge.admission.consistency import ConsistencyCheck
from llm_reasoner.knowledge.ontologies import ActionPerspectiveFile
from llm_reasoner.knowledge.ontology_translation.reading import (
    ConflictingPythonNames,
    FieldTypeOutOfReach,
    MissingPropertyRange,
    OntologyReader,
    UnsupportedRestriction,
)
from llm_reasoner.knowledge.vocabulary_proposal.catalogue import (
    PREFIX_SEPARATOR,
    Prefix,
    TermCatalogue,
    TermKind,
    is_prefixed,
)
from llm_reasoner.knowledge.vocabulary_proposal.perspective import Perspective
from llm_reasoner.knowledge.vocabulary_proposal.proposal import (
    RestrictionKind,
    VocabularyProposal,
)
from llm_reasoner.knowledge.vocabulary_proposal.writing import ProposalWriter

CAPITALISED_NAME = re.compile(r"[A-Z][A-Za-z0-9]*")
"""
How a class, a value partition or a member is named: one word starting in capitals.
"""

PROPERTY_NAME = re.compile(r"[a-z][A-Za-z0-9]*")
"""
How a property is named: one word starting in lower case.
"""

GENERATOR_REFUSALS = (
    ConflictingPythonNames,
    FieldTypeOutOfReach,
    MissingPropertyRange,
    UnsupportedRestriction,
)
"""
What the vocabulary generator raises for a vocabulary it cannot write as Python.
"""


class ProblemKind(StrEnum):
    """
    The rules a proposal can break.
    """

    INVALID_NAME = "invalid_name"
    """
    A name is not one word, in capitals for a class and in lower case for a property.
    """

    DUPLICATE_NAME = "duplicate_name"
    """
    Two terms of the proposal share a name.
    """

    EXISTING_NAME = "existing_name"
    """
    A term is named like one the core, CRAM-AICOR or DUL already has, which is to be
    reused instead.
    """

    MISSING_MEANING = "missing_meaning"
    """
    A term has no definition.
    """

    UNKNOWN_TERM = "unknown_term"
    """
    A reference names a term neither the core, CRAM-AICOR, DUL nor the proposal declares.
    """

    WRONG_KIND = "wrong_kind"
    """
    A reference names a term of the wrong kind for where it stands.
    """

    UNGROUNDED_CLASS = "ungrounded_class"
    """
    A class does not specialise a class of the core, CRAM-AICOR or DUL, even through
    other classes of the proposal.
    """

    NOT_ATTACHED = "not_attached"
    """
    The vocabulary never refers to the action core.
    """

    PYTHON_GENERATION = "python_generation"
    """
    The vocabulary generator cannot write the vocabulary as Python.
    """

    INCONSISTENT = "inconsistent"
    """
    A reasoner finds the vocabulary inconsistent with the core, CRAM-AICOR and DUL.
    """


@dataclass(frozen=True)
class Problem:
    """
    A rule a proposal breaks, where.
    """

    kind: ProblemKind
    """
    The rule.
    """

    term: str
    """
    The term of the proposal that breaks it; empty for the vocabulary as a whole.
    """

    message: str
    """
    What is wrong, in one sentence.
    """

    def to_json(self) -> Dict[str, Any]:
        """
        :return: The problem as the model reads it.
        """
        return {"kind": self.kind, "term": self.term, "message": self.message}


@dataclass(frozen=True)
class Reference:
    """
    A place where a proposal names a term, and the kind of term that belongs there.
    """

    owner: str
    """
    The term of the proposal that names it.
    """

    name: str
    """
    The term named.
    """

    expected: TermKind
    """
    The kind of term that belongs there.
    """

    role: str
    """
    Where it stands, such as ``parent`` or ``range``.
    """


@dataclass(frozen=True)
class ProposalChecker:
    """
    Checks a proposed vocabulary against the rules every perspective follows.
    """

    catalogue: TermCatalogue
    """
    The terms a proposal may build on.
    """

    perspective: Perspective
    """
    The perspective the vocabulary is of.
    """

    consistency: Optional[ConsistencyCheck] = None
    """
    The reasoner the vocabulary is checked with against the core and the linked
    ontologies; none to leave that check out.
    """

    def problems(self, proposal: VocabularyProposal) -> List[Problem]:
        """
        :return: Every rule the proposal breaks. The generator and the reasoner are only
            asked about a proposal that breaks none of the other rules; the reasoner is
            given the module without its imports, since it has the imported files.
        """
        found = [
            *self.name_problems(proposal),
            *self.meaning_problems(proposal),
            *self.reference_problems(proposal),
            *self.grounding_problems(proposal),
            *self.attachment_problems(proposal),
        ]
        if found:
            return found
        found = self.generation_problems(proposal)
        if found or self.consistency is None:
            return found
        graph = ProposalWriter(proposal=proposal, perspective=self.perspective).graph()
        graph.remove((None, OWL.imports, None))
        if self.consistency.is_consistent(graph):
            return []
        return [
            Problem(
                ProblemKind.INCONSISTENT,
                "",
                "the reasoner finds the vocabulary inconsistent with the action core, "
                "CRAM-AICOR and DUL",
            )
        ]

    # %% names and meanings

    def name_problems(self, proposal: VocabularyProposal) -> Iterator[Problem]:
        """
        :return: The names that are malformed, repeated, or those of existing terms.
        """
        properties = {proposed.name for proposed in proposal.properties}
        for name in dict.fromkeys(proposal.names()):
            pattern = PROPERTY_NAME if name in properties else CAPITALISED_NAME
            if not pattern.fullmatch(name):
                yield Problem(
                    ProblemKind.INVALID_NAME,
                    name,
                    "a class, a value partition or a member is named as one word starting "
                    "in capitals, a property as one word starting in lower case",
                )
        for name, count in Counter(proposal.names()).items():
            if count > 1:
                yield Problem(
                    ProblemKind.DUPLICATE_NAME,
                    name,
                    f"{count} terms of the proposal are named {name}",
                )
        existing = {
            entry.name.split(PREFIX_SEPARATOR, 1)[1]: entry.name
            for entry in (*self.catalogue.core_terms(), *self.catalogue.linked_terms())
        }
        for name in dict.fromkeys(proposal.names()):
            if name in existing:
                yield Problem(
                    ProblemKind.EXISTING_NAME,
                    name,
                    f"{existing[name]} already exists; refer to it instead of declaring "
                    f"{name}",
                )

    @staticmethod
    def meaning_problems(proposal: VocabularyProposal) -> Iterator[Problem]:
        """
        :return: The terms without a definition.
        """
        for name, meaning in [
            *((proposed.name, proposed.meaning) for proposed in proposal.classes),
            *((proposed.name, proposed.meaning) for proposed in proposal.properties),
            *(
                (partition.name, partition.meaning)
                for partition in proposal.value_partitions
            ),
            *(
                (member.name, member.meaning)
                for partition in proposal.value_partitions
                for member in partition.members
            ),
        ]:
            if not meaning.strip():
                yield Problem(ProblemKind.MISSING_MEANING, name, "it has no definition")

    # %% references

    @staticmethod
    def references(proposal: VocabularyProposal) -> Iterator[Reference]:
        """
        :return: Every place the proposal names a term, with the kind that belongs there.
        """
        for proposed in proposal.classes:
            for parent in proposed.parents:
                yield Reference(proposed.name, parent, TermKind.CLASS, "parent")
            for restriction in proposed.restrictions:
                yield Reference(
                    proposed.name,
                    restriction.on_property,
                    TermKind.PROPERTY,
                    "restricted property",
                )
                yield Reference(
                    proposed.name,
                    restriction.filler,
                    (
                        TermKind.INDIVIDUAL
                        if restriction.kind is RestrictionKind.VALUE
                        else TermKind.CLASS
                    ),
                    f"filler of a '{restriction.kind}' restriction",
                )
        for partition in proposal.value_partitions:
            for parent in partition.parents:
                yield Reference(partition.name, parent, TermKind.CLASS, "parent")
        for proposed in proposal.properties:
            yield Reference(proposed.name, proposed.domain, TermKind.CLASS, "domain")
            if proposed.value_class is not None:
                yield Reference(
                    proposed.name, proposed.value_class, TermKind.CLASS, "range"
                )
            for parent in proposed.parents:
                yield Reference(proposed.name, parent, TermKind.PROPERTY, "parent")

    def kind_of(self, name: str, declared: Dict[str, TermKind]) -> Optional[TermKind]:
        """
        :return: The kind of the term a name names; ``None`` when nothing declares it.
        """
        if is_prefixed(name):
            return self.catalogue.kind_of(name)
        return declared.get(name)

    def reference_problems(self, proposal: VocabularyProposal) -> Iterator[Problem]:
        """
        :return: The references to terms nothing declares, or to terms of the wrong kind.
        """
        declared = proposal.declared()
        prefixes = ", ".join(prefix + PREFIX_SEPARATOR for prefix in Prefix)
        for reference in self.references(proposal):
            kind = self.kind_of(reference.name, declared)
            if kind is None:
                yield Problem(
                    ProblemKind.UNKNOWN_TERM,
                    reference.owner,
                    f"its {reference.role} {reference.name} is declared neither by the "
                    f"proposal nor as a term named {prefixes}",
                )
            elif kind is not reference.expected:
                yield Problem(
                    ProblemKind.WRONG_KIND,
                    reference.owner,
                    f"its {reference.role} {reference.name} is a {kind}, where a "
                    f"{reference.expected} belongs",
                )

    # %% grounding and attachment

    def grounding_problems(self, proposal: VocabularyProposal) -> Iterator[Problem]:
        """
        :return: The classes that specialise no class of the core, CRAM-AICOR or DUL,
            even through classes of the proposal.
        """
        parents: Dict[str, Tuple[str, ...]] = {
            **{proposed.name: proposed.parents for proposed in proposal.classes},
            **{
                partition.name: partition.parents
                for partition in proposal.value_partitions
            },
        }
        grounded = {
            name
            for name, given in parents.items()
            if any(
                is_prefixed(parent) and self.catalogue.kind_of(parent) is TermKind.CLASS
                for parent in given
            )
        }
        growing = True
        while growing:
            reached = {
                name
                for name, given in parents.items()
                if name not in grounded and any(parent in grounded for parent in given)
            }
            grounded |= reached
            growing = bool(reached)
        for name in parents:
            if name not in grounded:
                yield Problem(
                    ProblemKind.UNGROUNDED_CLASS,
                    name,
                    "it specialises no class of the action core, CRAM-AICOR or DUL, "
                    "directly or through the proposal's classes",
                )

    @staticmethod
    def attachment_problems(proposal: VocabularyProposal) -> Iterator[Problem]:
        """
        :return: A problem when the vocabulary never refers to the action core.
        """
        core = Prefix.CORE + PREFIX_SEPARATOR
        if not any(
            reference.name.startswith(core)
            for reference in ProposalChecker.references(proposal)
        ):
            yield Problem(
                ProblemKind.NOT_ATTACHED,
                "",
                "nothing in the vocabulary refers to the action core; a perspective "
                "describes an action in terms of the core's slots, frames and parameter "
                "dimensions",
            )

    # %% the generator

    def generation_problems(self, proposal: VocabularyProposal) -> List[Problem]:
        """
        :return: Why the vocabulary generator cannot write the vocabulary as Python, if
            it cannot; it reads the core and the proposal together as it would modules.
        """
        module_file = self.perspective.module_file
        reader = OntologyReader(
            {
                ActionPerspectiveFile.ACTION_CORE: self.catalogue.core_graph,
                module_file: ProposalWriter(
                    proposal=proposal, perspective=self.perspective
                ).graph(),
            }
        )
        try:
            reader.module_declaration(module_file)
        except GENERATOR_REFUSALS as refusal:
            return [Problem(ProblemKind.PYTHON_GENERATION, "", str(refusal))]
        return []
