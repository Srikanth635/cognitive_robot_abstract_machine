---
generated:
  by: repository-knowledge-pilot/0.1.0
  at: '2026-10-20T05:56:43.052905+00:00'
sources:
- id: entry
  resource: https://github.com/cram2/cognitive_robot_abstract_machine/blob/7b27424de7768798dc561f447e11c2a8eb86c307/scripts/regenerate_all_orm.py#L1-L33
- id: registry
  resource: https://github.com/cram2/cognitive_robot_abstract_machine/blob/7b27424de7768798dc561f447e11c2a8eb86c307/cognitive_robot_abstract_machine/orm_interfaces.py#L1-L459
- id: rules
  resource: https://github.com/cram2/cognitive_robot_abstract_machine/blob/7b27424de7768798dc561f447e11c2a8eb86c307/AGENTS.md#L1-L156
type: Workflow
title: ORM generation
description: Generator entry point, registry and generated-file requirements.
status: draft
cram:
  source_commit: 7b27424de7768798dc561f447e11c2a8eb86c307
  depends_on: []
  replacement: ''
  evidence:
  - source_id: entry
    path: scripts/regenerate_all_orm.py
    start_line: 1
    end_line: 33
    content_digest: d0638e58eefe5eb635dd3df3bf182f69fa283ec89478c7691d84f6e37b8d351a
    symbol: ''
  - source_id: registry
    path: cognitive_robot_abstract_machine/orm_interfaces.py
    start_line: 1
    end_line: 459
    content_digest: 9d4d921af3bd6b856c6974065224781d14a680e197339281661cf5c399fc489d
    symbol: ''
  - source_id: rules
    path: AGENTS.md
    start_line: 1
    end_line: 156
    content_digest: 76187033b0444e5b846b4c7b3af6d0c7944b44fad6803b376b0fd1ad85caad91
    symbol: ''
---
# ORM generation

`scripts/regenerate_all_orm.py` parses its debug flag and calls
`WORKSPACE_ORM_INTERFACES.regenerate`. The workspace registry describes the interfaces built for the
packages.[^entry][^registry]

Repository instructions require generated `ormatic_interface.py` files to remain ignored and
untouched. For an ORM problem, use the configured interpreter to run
`scripts/regenerate_all_orm.py`; if regeneration does not fix it, ask the developer. Test commands
must pass `--orm-build never` and regenerate explicitly when required.[^rules]

For agent navigation, inspect the generator entry point and its registry rather than opening
generated mappings. This concept is explanatory context; the applicable repository instructions
remain authoritative.

[^entry]: Regeneration entry point
[^registry]: Workspace interface registry
[^rules]: Repository ORM and testing instructions
