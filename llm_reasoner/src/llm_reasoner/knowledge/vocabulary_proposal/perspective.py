"""
The perspectives on an action whose vocabulary a model proposes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from llm_reasoner.knowledge.ontologies import ActionPerspectiveModule

TURTLE_SUFFIX = ".ttl"
"""
The suffix of an ontology module file.
"""

WORD_START = re.compile(r"(?<!^)(?=[A-Z])")
"""
Where a new word begins in a name written in capitals, such as ``FillerExpectation``.
"""


@dataclass(frozen=True)
class ProposedModuleFile(ActionPerspectiveModule):
    """
    The module file of a perspective whose vocabulary is proposed, which is not yet one
    of the registered action perspective modules.
    """

    value: str
    """
    The file's name.
    """


class Perspective(StrEnum):
    """
    A perspective on an action, named as its module is.
    """

    ACTION_FRAME = "action-frame"
    """
    The participants an action needs, what each is normally like, its defaults, its
    preconditions and what it changes, as FrameNet and Minsky's frames describe it.
    """

    IMAGE_SCHEMAS = "image-schemas"
    """
    The spatial and force structure underneath an action, as Johnson, Lakoff and Talmy
    describe it: source-path-goal, containment, verticality, support and contact between
    the core's slots, and the parameter dimensions each constrains.
    """

    @property
    def description(self) -> str:
        """
        What the perspective covers, as the model is told.
        """
        return self.module_file.description

    @property
    def module_file(self) -> ProposedModuleFile:
        """
        The module file the perspective's vocabulary is written as.
        """
        return ProposedModuleFile(value=self.value + TURTLE_SUFFIX)

    @property
    def namespace(self) -> str:
        """
        The namespace the perspective declares its terms in.
        """
        return self.module_file.namespace

    @property
    def ontology_iri(self) -> str:
        """
        The IRI of the perspective's module.
        """
        return self.namespace.rstrip("#")

    @property
    def label(self) -> str:
        """
        The perspective's name in words.
        """
        return self.value.replace("-", " ")


def label_of(name: str) -> str:
    """
    :return: A name written in capitals as lower-case words, such as ``filler
        expectation`` for ``FillerExpectation``.
    """
    return WORD_START.sub(" ", name).lower()
