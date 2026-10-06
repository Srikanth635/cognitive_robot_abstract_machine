"""
Generating hypotheses about actions with a language model, on the vocabulary generated
from the ontology.

:mod:`~llm_reasoner.generation.schema` derives the output shape from the vocabulary
classes, :mod:`~llm_reasoner.generation.prompts` states it to the model,
:mod:`~llm_reasoner.generation.wire` decodes the answer into vocabulary instances, and
:mod:`~llm_reasoner.generation.backend` runs the loop of asking, checking with the
acceptance suite and repairing, as a krrood generative backend.
"""
