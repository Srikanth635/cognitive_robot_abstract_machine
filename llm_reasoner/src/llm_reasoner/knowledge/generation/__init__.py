"""
Generating hypotheses about actions with a language model, on the vocabulary generated
from the ontology.

:mod:`~llm_reasoner.knowledge.generation.request` reads what a query asks for,
:mod:`~llm_reasoner.knowledge.generation.schema` derives the output shape from the
vocabulary classes, :mod:`~llm_reasoner.knowledge.generation.prompts` states it to the
model, :mod:`~llm_reasoner.knowledge.generation.wire` decodes the answer into vocabulary
instances, :mod:`~llm_reasoner.knowledge.generation.assessment` judges it,
:mod:`~llm_reasoner.knowledge.generation.dialogue` asks and repairs,
:mod:`~llm_reasoner.knowledge.generation.recording` records and stores the admitted
hypothesis, and :mod:`~llm_reasoner.knowledge.generation.backend` orchestrates them as a
krrood generative backend.
"""
