"""
Regenerate the Python vocabulary modules from the ontology files.
"""

from llm_reasoner.ontology_translation.writing import VocabularyWriter

VocabularyWriter.for_predictive_manipulation().write_all()
