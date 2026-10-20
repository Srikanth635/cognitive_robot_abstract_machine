---
generated:
  by: repository-knowledge-pilot/0.1.0
  at: '2026-10-20T05:56:43.021902+00:00'
sources:
- id: environment
  resource: https://github.com/cram2/cognitive_robot_abstract_machine/blob/7b27424de7768798dc561f447e11c2a8eb86c307/coraplex/src/coraplex/execution_environment.py#L1-L140
- id: executable
  resource: https://github.com/cram2/cognitive_robot_abstract_machine/blob/7b27424de7768798dc561f447e11c2a8eb86c307/coraplex/src/coraplex/plans/executables.py#L369-L400
type: Workflow
title: Execution environments
description: Execution modes, nested settings and simulation context.
status: draft
cram:
  source_commit: 7b27424de7768798dc561f447e11c2a8eb86c307
  depends_on: []
  replacement: ''
  evidence:
  - source_id: environment
    path: coraplex/src/coraplex/execution_environment.py
    start_line: 1
    end_line: 140
    content_digest: 434f85cdea4bde00b29e7127988f8c331c8845d1c0659dc67483b26146918b81
    symbol: ''
  - source_id: executable
    path: coraplex/src/coraplex/plans/executables.py
    start_line: 369
    end_line: 400
    content_digest: 3441ed1903cafeb7254bbeb2a31e5bb798a4559a61c342c48022c86c8f730722
    symbol: coraplex.plans.executables.GiskardExecutable.execute
---
# Execution environments

`ExecutionEnvironment.__enter__` saves the previous execution type and collision-avoidance setting,
then assigns the requested values to `GiskardExecutable`. Its exit method restores both values.
`PhysicallySimulatedRobot` additionally saves, assigns and restores the executable's
simulation.[^environment]

The physical environment's documented precondition is an already started stepped simulation.
`GiskardExecutable.execute` distinguishes no execution, simulated execution and real execution; it
raises for a mode outside its handled cases.[^environment][^executable]

For a change to environment handling, inspect both classes together. This explanation describes
static contracts and does not demonstrate that a simulator or robot was run.

[^environment]: Execution environment definitions
[^executable]: GiskardExecutable.execute
