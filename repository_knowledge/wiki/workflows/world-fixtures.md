---
generated:
  by: repository-knowledge-pilot/0.1.0
  at: '2026-10-20T05:56:43.049902+00:00'
sources:
- id: fixtures
  resource: https://github.com/cram2/cognitive_robot_abstract_machine/blob/7b27424de7768798dc561f447e11c2a8eb86c307/test/coraplex_test/conftest.py#L94-L134
- id: snapshot
  resource: https://github.com/cram2/cognitive_robot_abstract_machine/blob/7b27424de7768798dc561f447e11c2a8eb86c307/test/coraplex_test/world_snapshot.py#L1-L55
type: Workflow
title: World fixtures
description: Shared context fixtures and world model/state restoration.
status: draft
cram:
  source_commit: 7b27424de7768798dc561f447e11c2a8eb86c307
  depends_on: []
  replacement: ''
  evidence:
  - source_id: fixtures
    path: test/coraplex_test/conftest.py
    start_line: 94
    end_line: 134
    content_digest: 6537a3ef888904cf1e4c5d7b9a40d507e5c7bda43bfc5100b38da0cf44dc3c58
    symbol: ''
  - source_id: snapshot
    path: test/coraplex_test/world_snapshot.py
    start_line: 1
    end_line: 55
    content_digest: 4a548313a66fc7a5cc943886b12fbc862a16af49662936b0c07d737f4dad2fbf
    symbol: ''
---
# World fixtures

The CoraPlex fixture `pr2_apartment_context` captures a `WorldSnapshot`, yields the shared world,
PR2 and a seeded context, then restores the snapshot. `simple_pr2_context` and
`stretch_apartment_context` use the same capture/yield/restore pattern for their respective
worlds.[^fixtures]

`WorldSnapshot.capture` records the world model version and degree-of-freedom states. `restore`
rolls the world model back, restores the recorded states and notifies a state change.[^snapshot]

For agent-generated tests, inspect the applicable fixture and its dependencies before choosing it.
These citations describe restoration of model/state; they do not establish restoration of arbitrary
process globals or external services.

[^fixtures]: CoraPlex context fixtures
[^snapshot]: WorldSnapshot
