// Tests of what the page shows of one NEEM: the tree of the steps of its plan, and for each
// step what it was asked to do and the things and places it refers to.
import test from 'node:test';
import assert from 'node:assert';
import { BLOCK } from '../../../llm_reasoner_viz/src/llm_reasoner_viz/web/inspector.mjs';
import {
  NOT_SET, describeNode, displayValue, firstStep, neemAddress, planRows, referenceText, trailOf,
} from '../../../llm_reasoner_viz/src/llm_reasoner_viz/web/neem.mjs';
import { loadFixture } from './support.mjs';

const NAVIGATION = loadFixture('neem_navigation.json').plan;
const PICK_UP = loadFixture('neem_pick_up.json').plan;

function node(nodes, id) {
  return nodes.find(function (candidate) { return candidate.id === id; });
}

function block(view, title) {
  return view.blocks.find(function (candidate) { return candidate.title === title; });
}

// %% where to ask

test('one NEEM is asked for by its number', function () {
  assert.strictEqual(neemAddress(7), '/api/neems/7');
});

// %% the tree

test('the rows of the tree follow the plan, each as deep as its step', function () {
  const rows = planRows(PICK_UP);
  assert.deepStrictEqual(rows.map(function (row) { return row.id; }), [1, 2, 3, 4, 5, 6]);
  assert.deepStrictEqual(rows.map(function (row) { return row.depth; }), [0, 1, 1, 2, 2, 1]);
});

test('an action or a motion is named for what it was asked to do', function () {
  const labels = planRows(PICK_UP).map(function (row) { return row.label; });
  assert.strictEqual(labels[1], 'ParkArmsAction');
  assert.strictEqual(labels[2], 'PickUpAction');
  assert.strictEqual(labels[4], 'MoveToolCenterPointMotion');
});

test('the steps that are not an action or a motion are named for what they are', function () {
  const labels = planRows(PICK_UP).map(function (row) { return row.label; });
  assert.strictEqual(labels[0], 'sequence');
  assert.strictEqual(labels[3], 'condition before');
  assert.strictEqual(labels[5], 're attach');
});

test('a condition checked after is named so', function () {
  const after = Object.assign({}, node(PICK_UP, 4), {
    parameters: [{ name: 'pre_condition', value: false }],
  });
  assert.strictEqual(planRows([after])[0].label, 'condition after');
});

test('conditions are set apart from the steps that do the work', function () {
  const muted = planRows(PICK_UP).map(function (row) { return row.muted; });
  assert.deepStrictEqual(muted, [false, false, false, true, false, false]);
});

test('the trail of a step is the steps above it, from the root', function () {
  assert.deepStrictEqual(trailOf(PICK_UP, 5), [{ id: 1, label: 'sequence' }, { id: 3, label: 'PickUpAction' }]);
  assert.deepStrictEqual(trailOf(PICK_UP, 1), []);
});

test('the view opens on the first action, which says most of what was done', function () {
  assert.strictEqual(firstStep(PICK_UP).id, 2);
  assert.strictEqual(firstStep(NAVIGATION).id, 12);
});

test('a plan with no action opens on its root', function () {
  const motions = PICK_UP.filter(function (candidate) { return candidate.type !== 'ActionNode'; });
  assert.strictEqual(firstStep(motions).id, 1);
});

// %% values written for reading

test('an enumeration is written as its member, whatever its module', function () {
  assert.strictEqual(displayValue('coraplex.datastructures.enums.Arms.RIGHT'), 'RIGHT');
  assert.strictEqual(displayValue('coraplex.datastructures.enums.VerticalAlignment.NoAlignment'), 'NoAlignment');
});

test('a class is written as its name', function () {
  assert.strictEqual(displayValue('giskardpy.motion_statechart.goals.templates.Sequence'), 'Sequence');
});

test('a name with a dot that is no path stays as it is', function () {
  assert.strictEqual(displayValue('bowl.stl'), 'bowl.stl');
  assert.strictEqual(displayValue('Static_MilkPitcher.stl'), 'Static_MilkPitcher.stl');
});

test('a number keeps what matters of it', function () {
  assert.strictEqual(displayValue(1.7514786941281228), '1.75148');
  assert.strictEqual(displayValue(0.10000000000000009), '0.1');
  assert.strictEqual(displayValue(33), '33');
  assert.strictEqual(displayValue(0), '0');
});

test('a flag is true or false', function () {
  assert.strictEqual(displayValue(true), 'true');
  assert.strictEqual(displayValue(false), 'false');
});

test('a value that was not set says so', function () {
  assert.strictEqual(displayValue(null), NOT_SET);
});

test('text that is no path stays as it is', function () {
  assert.strictEqual(displayValue('["torso_lift_joint"]'), '["torso_lift_joint"]');
});

// %% references written for reading

test('a thing of the world is its kind and the name the world gives it', function () {
  const pour = node(NAVIGATION, 14);
  assert.strictEqual(referenceText(pour.references[0]), 'Body · bowl.stl');
  assert.strictEqual(referenceText(pour.references[1]), 'PouringCup · PouringCup');
});

test('a place is its frame, its position and the way it faces', function () {
  const navigation = node(NAVIGATION, 13);
  assert.strictEqual(referenceText(navigation.references[0]), 'Pose · map · (0.5, -1, 0) · facing (0, 0, 0, 1)');
});

test('a place that faces no way is its frame and its position', function () {
  const motion = node(PICK_UP, 5);
  assert.strictEqual(referenceText(motion.references[0]), 'Pose · map · (1.5, 2.25, 0)');
});

test('an object that is only values is its kind', function () {
  assert.strictEqual(referenceText(node(PICK_UP, 3).references[0]), 'GraspDescription');
});

// %% what a step is

test('a step is titled for what it was asked to do, and says its number and kind', function () {
  const view = describeNode(node(NAVIGATION, 14));
  assert.strictEqual(view.title, 'PouringAction');
  assert.deepStrictEqual(view.facts, [['step', '#14'], ['type', 'ActionNode']]);
});

test('a step that was asked for nothing in particular is titled for its kind', function () {
  assert.strictEqual(describeNode(node(PICK_UP, 1)).title, 'sequence');
});

test('a step has no status of its own, since the plan stores only its root as ended', function () {
  assert.strictEqual(describeNode(node(PICK_UP, 3)).status, '');
});

test('a step of a NEEM leads to no other NEEM', function () {
  assert.strictEqual(describeNode(node(PICK_UP, 3)).neem, null);
});

test('the values of a step are written for reading, those not set as such', function () {
  const values = block(describeNode(node(NAVIGATION, 14)), 'Values');
  assert.strictEqual(values.kind, BLOCK.FACTS);
  const asked = new Map(values.facts);
  assert.strictEqual(asked.get('tilt_angle'), '1.6');
  assert.strictEqual(asked.get('arm'), 'RIGHT');
  assert.strictEqual(asked.get('position_threshold'), NOT_SET);
});

test('the things a step refers to are listed by what it uses them as', function () {
  const things = block(describeNode(node(NAVIGATION, 14)), 'Things and places');
  assert.deepStrictEqual(things.facts, [
    ['target_container', 'Body · bowl.stl'],
    ['source_container', 'PouringCup · PouringCup'],
  ]);
});

test('a step that moves a body names the body and its new parent', function () {
  const things = block(describeNode(node(PICK_UP, 6)), 'Things and places');
  assert.deepStrictEqual(things.facts, [['body', 'Body · bowl.stl'], ['new_parent', 'Body · map']]);
});

test('an object that is only values has a block of its own with them', function () {
  const grasp = block(describeNode(node(PICK_UP, 3)), 'Values of grasp_description');
  assert.strictEqual(grasp.kind, BLOCK.FACTS);
  assert.deepStrictEqual(grasp.facts, [
    ['rotate_gripper', 'false'],
    ['manipulation_offset', '0.1'],
    ['approach_direction', 'FRONT'],
    ['vertical_alignment', 'NoAlignment'],
  ]);
});

test('a step that refers to nothing has no block of things', function () {
  assert.strictEqual(block(describeNode(node(PICK_UP, 4)), 'Things and places'), undefined);
});

test('a step with no values has no block of values', function () {
  assert.strictEqual(block(describeNode(node(PICK_UP, 6)), 'Values'), undefined);
});

test('a condition says whether it is checked before', function () {
  const values = block(describeNode(node(PICK_UP, 4)), 'Values');
  assert.deepStrictEqual(values.facts, [['pre_condition', 'true']]);
});

test('the rows a step is kept in are given as they are stored, to copy', function () {
  const rows = block(describeNode(node(NAVIGATION, 14)), 'Rows it is kept in');
  assert.strictEqual(rows.kind, BLOCK.JSON);
  assert.deepStrictEqual(rows.value, node(NAVIGATION, 14).rows);
});
