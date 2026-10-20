---
generated:
  by: repository-knowledge-pilot/0.1.0
  at: '2026-10-20T05:56:43.102780+00:00'
sources:
- id: role
  resource: https://github.com/cram2/cognitive_robot_abstract_machine/blob/7b27424de7768798dc561f447e11c2a8eb86c307/krrood/src/krrood/patterns/role.py#L32-L324
- id: guide
  resource: https://github.com/cram2/cognitive_robot_abstract_machine/blob/7b27424de7768798dc561f447e11c2a8eb86c307/krrood/doc/patterns/role.md#L1-L402
type: Convention
title: Roles and role takers
description: Composition, role identity and delegated attribute access.
status: draft
cram:
  source_commit: 7b27424de7768798dc561f447e11c2a8eb86c307
  depends_on: []
  replacement: ''
  evidence:
  - source_id: role
    path: krrood/src/krrood/patterns/role.py
    start_line: 32
    end_line: 324
    content_digest: ad2dc9cbadfc5c1a44774b5e6fce51b1b0a574c2404ece9936c364237f1a287a
    symbol: krrood.patterns.role.Role
  - source_id: guide
    path: krrood/doc/patterns/role.md
    start_line: 1
    end_line: 402
    content_digest: bb880e412033452bff5e4d8bc7ce4da7e5672c929fea3a96596dceba08037626
    symbol: ''
---
# Roles and role takers

`Role` adds contextual attributes to a role taker through composition. Its documented contract keeps
a role distinct from its taker, requires the explicit keyword-only `role_taker`, and recommends
`@dataclass(eq=False)` for role subclasses.[^role]

The role contract delegates reads of undeclared attributes to the taker, restricts writes to
declared role fields, and directs taker changes through `role.role_taker`. It distinguishes object
identity from semantic equivalence through `IsSameSemanticEntity`.[^role]

The role guide demonstrates construction, lookup, parallel roles and role chaining.[^guide] Use the
qualified source class and guide when navigating this pattern; source extraction cannot resolve all
delegated attributes.

[^role]: Role
[^guide]: Role user guide
