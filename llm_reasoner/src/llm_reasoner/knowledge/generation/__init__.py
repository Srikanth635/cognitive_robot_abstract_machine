"""
Generating hypotheses: an underspecified action to stereotypical structure.

WHAT THIS PRODUCES
------------------
Given an underspecified action and nothing else -- no world state, no episode --
a structure conforming to the selected microworld: phases, subgoals, boundary
events, participant slots, parameter ranges. Type-level, non-indexical,
defeasible. What pouring is like in general, not what is about to happen.

Withholding world state is a design commitment, not an oversight. It is what
makes a hypothesis cacheable across episodes, keyed on the action type and the
versions of the modules that produced it, and what keeps the resolver's job --
unifying generic structure with particular facts -- a separate, inspectable step.

THE LOOP
--------
    underspecified action
      -> module selection        which microworlds does this action need
      -> schema derivation       the output shape, from the compiled classes
      -> ask                     one completion
      -> decode                  a fragment graph, checked against the schema
      -> construct               instances, wired by id in a second pass
      -> validate                the microworld's own acceptance constraints
      -> repair, or admit with provenance

The interesting step is repair. A violated constraint is diagnostic feedback in
the ontology's own vocabulary -- "this transition fires on an event that does not
end the phase it leaves" -- naming both the fault and the fragment responsible,
which a parse error cannot do.

MODULES
-------
``schema``
    ``derive_schema`` reads the output shape back from the ``@ontological``
    metadata, so a field renamed in a microworld changes the schema without a
    second edit. A hand-written mirror would drift silently.
``prompts``
    The system contract and ``GenerationPromptBuilder``. None of the resolver's
    prompt text is reused: its vocabulary presumes one object being completed
    against a world, and generation has neither.
``assumptions``
    Extracting what a hypothesis presupposes, driven by ``marks_assumption``
    metadata rather than by field names, so it works for the next microworld.
``cached``
    ``CachingGenerator``, which consults a store before calling a model. A
    wrapper rather than a flag, so the generator stays a thing that always
    generates and the caching policy can change independently.
``backend``
    ``HypothesisGenerator``. Shaped after the resolver's pipeline but not
    subclassed from it -- there is no query to reflect over and no world to
    select from, and the result is a graph rather than one object.

ON OUTPUT SHAPE
---------------
A flat list of fragments, each with an id, cross-referenced by id. Nesting a
hypothesis as one JSON literal cannot express the cycles that phases and
conditions genuinely form, duplicates any fragment referenced twice, and makes a
single malformed bracket invalidate everything.
"""

from llm_reasoner.knowledge.generation.assumptions import extract_assumptions
from llm_reasoner.knowledge.generation.backend import (
    GenerationFailure,
    GenerationPromptVersion,
    GenerationResult,
    HypothesisGenerator,
    RepairBudgetExhausted,
    ResponseUnusable,
    UnreachableModules,
)
from llm_reasoner.knowledge.generation.cached import CachedResult, CachingGenerator
from llm_reasoner.knowledge.generation.prompts import (
    SYSTEM_PROMPT,
    GenerationPromptBuilder,
)
from llm_reasoner.knowledge.generation.schema import (
    ClassSchema,
    FieldKind,
    FieldSchema,
    GenerationSchema,
    derive_schema,
    to_json_schema,
)
from llm_reasoner.knowledge.generation.wire import (
    CanonicalFragmentIdentityAllocator,
    FragmentDeclaration,
    FragmentField,
    GenerationWireDecoder,
    HypothesisDeclaration,
    WireFormatError,
)

__all__ = [
    "CachedResult",
    "CachingGenerator",
    "CanonicalFragmentIdentityAllocator",
    "ClassSchema",
    "FieldKind",
    "FieldSchema",
    "FragmentDeclaration",
    "FragmentField",
    "GenerationFailure",
    "GenerationPromptVersion",
    "GenerationPromptBuilder",
    "GenerationResult",
    "GenerationSchema",
    "GenerationWireDecoder",
    "HypothesisGenerator",
    "HypothesisDeclaration",
    "RepairBudgetExhausted",
    "ResponseUnusable",
    "SYSTEM_PROMPT",
    "UnreachableModules",
    "WireFormatError",
    "derive_schema",
    "extract_assumptions",
    "to_json_schema",
]
