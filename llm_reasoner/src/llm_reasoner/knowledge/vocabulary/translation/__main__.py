"""
Regenerate the Python vocabulary modules from the ontology files.
"""

from llm_reasoner.knowledge.vocabulary.translation.writing import VocabularyWriter

VocabularyWriter.for_knowledge_modules().write_all()
