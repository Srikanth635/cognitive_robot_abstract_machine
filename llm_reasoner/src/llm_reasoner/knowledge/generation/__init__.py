"""
Generating hypotheses about actions with a language model, on the vocabulary generated
from the ontology.

:mod:`~llm_reasoner.knowledge.generation.schema` derives the output shape from the
vocabulary classes, :mod:`~llm_reasoner.knowledge.generation.prompts` states it to the
model, :mod:`~llm_reasoner.knowledge.generation.wire` decodes the answer into vocabulary
instances, and :mod:`~llm_reasoner.knowledge.generation.backend` runs the loop of
asking, checking with the acceptance suite and repairing, as a krrood generative
backend.
"""
