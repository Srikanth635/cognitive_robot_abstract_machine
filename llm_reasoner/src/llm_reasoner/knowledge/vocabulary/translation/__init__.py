"""
Translation of the registered knowledge ontology modules into Python vocabulary modules.

The ontology files are the source of truth. Each module is read into declarations
(:mod:`~llm_reasoner.knowledge.vocabulary.translation.reading`) and written as a Python module of
krrood symbols and property descriptors
(:mod:`~llm_reasoner.knowledge.vocabulary.translation.writing`), which is committed in
the package of its family under :mod:`~llm_reasoner.knowledge.vocabulary.generated`. Run
``python -m llm_reasoner.knowledge.vocabulary.translation`` after changing an ontology
file.
"""
