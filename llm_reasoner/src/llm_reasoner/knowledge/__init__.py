"""Knowledge generation and representation.

Stereotypical, type-level knowledge about kinds of action, kept in a scope
disjoint from the episodic world model the resolver grounds against.

LAYOUT, WITH DEPENDENCIES POINTING INWARD
-----------------------------------------
``ontological``
    Reusable machinery for describing a domain model: the metadata vocabulary,
    the structural validator, and the constraint engine. Imports no microworld,
    so a microworld can be added, changed or deleted without touching it.

``microworlds``
    Domain models declared against that machinery, each owning its own
    acceptance constraints.

``query``
    The one entry point resolution uses. ``KnowledgeService`` answers a fixed
    set of questions with cited fragments; the resolver never holds a hypothesis
    and is free to reject any answer. Everything below this line may change so
    long as those questions still have answers.

WHY THE SCOPE SEPARATION MATTERS
--------------------------------
Instances here are PROTOTYPES, never episodes. Letting an episodic entity into a
hypothesis does not raise an error -- it quietly changes what queries mean, since
closed-world quantification ranges over whatever happens to be present. The
``ConceptLevel`` metadata exists so that boundary can be checked rather than
merely intended.
"""

from llm_reasoner.knowledge.ontological.validation import (
    report as validate_declarations,
    validate,
)
from llm_reasoner.knowledge.query import Cited, HypothesisView, KnowledgeService

__all__ = [
    "Cited",
    "HypothesisView",
    "KnowledgeService",
    "validate",
    "validate_declarations",
]