# llmr_updated_arch

`llmr_updated_arch` is an isolated, component-oriented rewrite of the existing
`llmr` action-resolution flow. It keeps the same broad behavior while splitting
the system into explicit stages:

1. Normalize a natural-language instruction or KRROOD `Match`.
2. Generate semantic artifacts such as slot descriptions.
3. Ground generated semantics into concrete Python and SymbolGraph values.
4. Materialize the resolved action instance.
5. Optionally project reasoner sidecars into a hypothesis graph.

The package does not import from the existing `llmr` package. KRROOD,
SymbolGraph, PyCRAM, and LangChain access live under `integrations/`.

```python
from llmr_updated_arch import instance_from_match, plan_from_instruction

action = instance_from_match(
    match,
    llm=llm,
    instruction="pick up the milk",
    symbol_type=Body,
)

plan = plan_from_instruction(
    "pick up the milk from the table",
    context=context,
    llm=llm,
    symbol_type=Body,
)
```
