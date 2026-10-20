---
generated:
  by: repository-knowledge-pilot/0.1.0
  at: '2026-10-20T05:56:43.080905+00:00'
sources:
- id: generic
  resource: https://github.com/cram2/cognitive_robot_abstract_machine/blob/7b27424de7768798dc561f447e11c2a8eb86c307/krrood/src/krrood/patterns/subclass_safe_generic.py#L63-L468
- id: rules
  resource: https://github.com/cram2/cognitive_robot_abstract_machine/blob/7b27424de7768798dc561f447e11c2a8eb86c307/AGENTS.md#L1-L156
type: Convention
title: Generic dataclasses
description: Generic parameter binding and dataclass field narrowing.
status: draft
cram:
  source_commit: 7b27424de7768798dc561f447e11c2a8eb86c307
  depends_on: []
  replacement: ''
  evidence:
  - source_id: generic
    path: krrood/src/krrood/patterns/subclass_safe_generic.py
    start_line: 63
    end_line: 468
    content_digest: 8a4f824e747e6df2930f2eda605859a779bfcae20aab5824fc21022a7c49d08e
    symbol: krrood.patterns.subclass_safe_generic.SubClassSafeGeneric
  - source_id: rules
    path: AGENTS.md
    start_line: 1
    end_line: 156
    content_digest: 76187033b0444e5b846b4c7b3af6d0c7944b44fad6803b376b0fd1ad85caad91
    symbol: ''
---
# Generic dataclasses

`SubClassSafeGeneric` narrows dataclass field types when subclasses bind generic parameters. Its
documented inheritance order puts `Generic[...]` before `SubClassSafeGeneric`; the helper itself
does not declare a fixed generic parameter list.[^generic]

Repository instructions require generic classes to inherit both bases, prohibit frozen dataclasses
for those classes, and require reading a bound parameter with `get_type_of_generic_parameter` rather
than `__orig_bases__`.[^rules]

Inspect the helper and applicable instructions together before generating a generic class. A static
source index does not prove that every forward reference or import cycle resolves at runtime.

[^generic]: SubClassSafeGeneric
[^rules]: Repository generic type-hint rules
