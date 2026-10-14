"""
Alignments of Flanagan concepts to the foundational ontologies.

A concept is aligned as a subclass only when the foreign term's definition entails that
every instance of the concept is an instance of the term. Where the definitions merely
resemble each other, the alignment is recorded as related, so the annotation never
claims more than has been checked.
"""

from __future__ import annotations

from llm_reasoner.knowledge_updated.ontological.references import Alignment, Namespace
from llm_reasoner.knowledge_updated.ontological.vocabulary import AlignmentRelation

DUL = Namespace("http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#")
"""
DOLCE UltraLite, the upper ontology SOMA builds on.
"""

SOMA = Namespace("http://www.ease-crc.org/ont/SOMA.owl#")
"""
The Socio-physical Model of Activities.
"""


def subclass_of(ontology: Namespace, term: str) -> Alignment:
    """
    Align a concept as a subclass of a term of a foundational ontology.
    """
    return Alignment(iri=ontology.term(term), relation=AlignmentRelation.SUBCLASS_OF)


def related_to(ontology: Namespace, term: str) -> Alignment:
    """
    Align a concept as merely related to a term of a foundational ontology.
    """
    return Alignment(iri=ontology.term(term), relation=AlignmentRelation.RELATED_TO)
