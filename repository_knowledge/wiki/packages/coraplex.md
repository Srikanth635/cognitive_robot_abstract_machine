---
generated:
  by: repository-knowledge-pilot/0.1.0
  at: '2026-10-20T05:56:42.998904+00:00'
sources:
- id: plan
  resource: https://github.com/cram2/cognitive_robot_abstract_machine/blob/7b27424de7768798dc561f447e11c2a8eb86c307/coraplex/src/coraplex/plans/plan.py#L338-L346
- id: action-node
  resource: https://github.com/cram2/cognitive_robot_abstract_machine/blob/7b27424de7768798dc561f447e11c2a8eb86c307/coraplex/src/coraplex/plans/plan_node.py#L562-L676
- id: executable
  resource: https://github.com/cram2/cognitive_robot_abstract_machine/blob/7b27424de7768798dc561f447e11c2a8eb86c307/coraplex/src/coraplex/plans/executables.py#L369-L400
type: Package
title: 'CoraPlex: plan to motion execution'
description: Plan entry points, action expansion and motion execution dispatch.
status: draft
cram:
  source_commit: 7b27424de7768798dc561f447e11c2a8eb86c307
  depends_on: []
  replacement: ''
  evidence:
  - source_id: plan
    path: coraplex/src/coraplex/plans/plan.py
    start_line: 338
    end_line: 346
    content_digest: 460549bba51a45d4b3ac32ff2479956db19f43fe1356428eb6ff6c97b9be5a88
    symbol: coraplex.plans.plan.Plan.perform
  - source_id: action-node
    path: coraplex/src/coraplex/plans/plan_node.py
    start_line: 562
    end_line: 676
    content_digest: 9a532e859dce72e2d6627a5d3f4f0d27de57d618ddc42b49e047efc51adb5bce
    symbol: coraplex.plans.plan_node.ActionNode
  - source_id: executable
    path: coraplex/src/coraplex/plans/executables.py
    start_line: 369
    end_line: 400
    content_digest: 3441ed1903cafeb7254bbeb2a31e5bb798a4559a61c342c48022c86c8f730722
    symbol: coraplex.plans.executables.GiskardExecutable.execute
---
# Plan to motion execution

`Plan.perform` saves a copy of the initial world and calls the root node's `perform` method.[^plan]

`ActionNode.notify` expands an action when it has no children, applies plan transformations, and
notifies its children. `ActionNode.parse` parses the body, then attaches the action's precondition
to its first Giskard executable and its postcondition to the last one; an action with no Giskard
executables returns the parsed executable directly.[^action-node]

`GiskardExecutable.execute` returns without executing when it has no motion mappings or the
environment selects `NO_EXECUTION`. Otherwise it prepares execution, dispatches simulated execution
for the kinematic and physical modes, and dispatches real execution for `REAL`.[^executable]

For agent navigation, start at these qualified symbols and inspect their current definitions before
proposing a change. Related explanations cover [execution
environments](/workflows/execution-environments.md) and [adding an
action](/workflows/adding-an-action.md).

[^plan]: Plan.perform
[^action-node]: ActionNode
[^executable]: GiskardExecutable.execute
