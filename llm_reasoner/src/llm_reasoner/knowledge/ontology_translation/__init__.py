"""
Translation of the predictive manipulation ontology into Python vocabulary modules.

The ontology files are the source of truth. Each module is read into declarations
(:mod:`~llm_reasoner.knowledge.ontology_translation.reading`) and written as a Python module of
krrood symbols and property descriptors
(:mod:`~llm_reasoner.knowledge.ontology_translation.writing`), which is committed next to the
ontology file it comes from. Run ``python -m llm_reasoner.knowledge.ontology_translation`` after
changing an ontology file.
"""
