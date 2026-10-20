---
generated:
  by: repository-knowledge-pilot/0.1.0
  at: '2026-10-20T05:56:43.046903+00:00'
sources:
- id: action-base
  resource: https://github.com/cram2/cognitive_robot_abstract_machine/blob/7b27424de7768798dc561f447e11c2a8eb86c307/coraplex/src/coraplex/robot_plans/actions/base.py#L36-L157
- id: navigation
  resource: https://github.com/cram2/cognitive_robot_abstract_machine/blob/7b27424de7768798dc561f447e11c2a8eb86c307/coraplex/src/coraplex/robot_plans/actions/core/navigation.py#L42-L84
- id: condition-tests
  resource: https://github.com/cram2/cognitive_robot_abstract_machine/blob/7b27424de7768798dc561f447e11c2a8eb86c307/test/coraplex_test/test_designator/test_action_conditions.py#L1-L178
type: Workflow
title: Adding an action
description: The action contract, navigation example and condition tests.
status: draft
cram:
  source_commit: 7b27424de7768798dc561f447e11c2a8eb86c307
  depends_on:
  - workflows/world-fixtures.md
  replacement: ''
  evidence:
  - source_id: action-base
    path: coraplex/src/coraplex/robot_plans/actions/base.py
    start_line: 36
    end_line: 157
    content_digest: d6296076c61874f97cc344c812b9282e4698e06f1d8309f9c3cded2e12bc0175
    symbol: coraplex.robot_plans.actions.base.ActionDescription
  - source_id: navigation
    path: coraplex/src/coraplex/robot_plans/actions/core/navigation.py
    start_line: 42
    end_line: 84
    content_digest: 6e29122549488972b57dae3aee1081d6f70d96b9a32df3f3c017383fe960fbce
    symbol: coraplex.robot_plans.actions.core.navigation.NavigateAction
  - source_id: condition-tests
    path: test/coraplex_test/test_designator/test_action_conditions.py
    start_line: 1
    end_line: 178
    content_digest: 915f3e21fd6485156b7e4656d427e4c783aef6832554bd153513bbc843977b22
    symbol: ''
---
# Adding an action

`ActionDescription` defines the abstract `_action_plan` property. Its `action_plan` property wraps
the returned subplan with an action node, a precondition node and a postcondition node. The default
condition methods return true, and `execute` mounts the action plan.[^action-base]

`NavigateAction` is a concrete example: it declares `target_location`, builds a plan with
`execute_single(MoveMotion(...))`, and overrides both condition methods. Use its current
implementation as a navigation example, rather than copying it without checking the new action's
contract.[^navigation]

The existing action-condition tests exercise condition behavior and provide a relevant starting test
module.[^condition-tests] The [world fixtures](/workflows/world-fixtures.md) explanation identifies
context restoration behavior.

[^action-base]: ActionDescription
[^navigation]: NavigateAction
[^condition-tests]: Action condition tests
