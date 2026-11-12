// Tests of what the page shows of one NEEM: the tree of the nodes of the statechart its plan
// was performed as, and for each step how it ended, what it was asked to do and the things
// and places it refers to.
import test from 'node:test';
import assert from 'node:assert';
import { BLOCK } from '../../../llm_reasoner_viz/src/llm_reasoner_viz/web/inspector.mjs';
import { formatOffset } from '../../../llm_reasoner_viz/src/llm_reasoner_viz/web/format.mjs';
import { STATUS } from '../../../llm_reasoner_viz/src/llm_reasoner_viz/web/model.mjs';
import {
  NEVER_STARTED, NOT_ENDED, NOT_SET, describeNode, displayValue, firstStep, neemAddress, planRows, referenceText,
  trailOf,
} from '../../../llm_reasoner_viz/src/llm_reasoner_viz/web/neem.mjs';
import { loadFixture } from './support.mjs';

const POUR = loadFixture('neem_pour.json').plan;
const NAVIGATION = loadFixture('neem_navigation.json').plan;

function node(nodes, id) {
  return nodes.find(function (candidate) { return candidate.id === id; });
}

function block(view, title) {
  return view.blocks.find(function (candidate) { return candidate.title === title; });
}

function fact(view, name) {
  return new Map(view.facts).get(name);
}

// %% where to ask

test('one NEEM is asked for by its number', function () {
  assert.strictEqual(neemAddress(7), '/api/neems/7');
});

// %% the tree

test('the rows of the tree follow the plan, each as deep as its step', function () {
  const rows = planRows(POUR);
  assert.deepStrictEqual(rows.map(function (row) { return row.id; }), [0, 1, 2, 3, 4, 5, 6]);
  assert.deepStrictEqual(rows.map(function (row) { return row.depth; }), [0, 1, 2, 1, 2, 0, 0]);
});

test('an action is named for what it was asked to do', function () {
  const labels = planRows(POUR).map(function (row) { return row.label; });
  assert.strictEqual(labels[1], 'PickUpAction');
  assert.strictEqual(labels[3], 'PouringAction');
});

test('the steps that are no action are named for their class', function () {
  const labels = planRows(POUR).map(function (row) { return row.label; });
  assert.strictEqual(labels[0], 'sequence');
  assert.strictEqual(labels[2], 'cartesian pose');
  assert.strictEqual(labels[5], 'end motion');
});

test('what the executor added to watch the plan is set apart from it', function () {
  const muted = planRows(POUR).map(function (row) { return row.muted; });
  assert.deepStrictEqual(muted, [false, false, false, false, false, true, true]);
});

test('a row stands as its step ended', function () {
  const standing = planRows(POUR).map(function (row) { return row.standing; });
  assert.strictEqual(standing[1], STATUS.OK);
  assert.strictEqual(standing[3], STATUS.FAILED);
});

test('a step not begun stands nowhere', function () {
  const waiting = Object.assign({}, node(POUR, 3), { status: 'NOT_STARTED' });
  assert.strictEqual(planRows([waiting])[0].standing, null);
});

test('a step interrupted because the step above it ended stands nowhere', function () {
  const interrupted = Object.assign({}, node(POUR, 4), { status: 'INTERRUPTED' });
  assert.strictEqual(planRows([interrupted])[0].standing, null);
});

test('the trail of a step is the steps above it, from the root', function () {
  assert.deepStrictEqual(trailOf(POUR, 2), [{ id: 0, label: 'sequence' }, { id: 1, label: 'PickUpAction' }]);
  assert.deepStrictEqual(trailOf(POUR, 0), []);
});

test('the view opens on the first action, which says most of what was done', function () {
  assert.strictEqual(firstStep(POUR).id, 1);
  assert.strictEqual(firstStep(NAVIGATION).id, 1);
});

test('a plan with no action opens on its root', function () {
  const motions = POUR.filter(function (candidate) { return candidate.designator === null; });
  assert.strictEqual(firstStep(motions).id, 0);
});

// %% values written for reading

test('an enumeration is written as its member, whatever its module', function () {
  assert.strictEqual(displayValue('coraplex.datastructures.enums.Arms.RIGHT'), 'RIGHT');
  assert.strictEqual(displayValue('coraplex.datastructures.enums.VerticalAlignment.NoAlignment'), 'NoAlignment');
});

test('a class is written as its name', function () {
  assert.strictEqual(displayValue('cramph.composites.Sequence'), 'Sequence');
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
  const pour = node(POUR, 3);
  assert.strictEqual(referenceText(pour.references[0]), 'Bowl · Bowl');
  assert.strictEqual(referenceText(pour.references[1]), 'PouringCup · PouringCup');
});

test('a place is its frame, its position and the way it faces', function () {
  const navigation = node(NAVIGATION, 3);
  assert.strictEqual(referenceText(navigation.references[0]), 'Pose · map · (0.5, -1, 0) · facing (0, 0, 0, 1)');
});

test('a place that faces no way is its frame and its position', function () {
  const reference = Object.assign({}, node(NAVIGATION, 3).references[0]);
  reference.pose = Object.assign({}, reference.pose, { orientation: null });
  assert.strictEqual(referenceText(reference), 'Pose · map · (0.5, -1, 0)');
});

test('an object that is only values is its kind', function () {
  assert.strictEqual(referenceText(node(POUR, 1).references[0]), 'GraspCandidate');
});

// %% what a step is

test('a step is titled for what it was asked to do, and says its number, kind and how it went', function () {
  const view = describeNode(node(POUR, 3));
  assert.strictEqual(view.title, 'PouringAction');
  assert.deepStrictEqual(view.facts, [
    ['step', '#3'], ['type', 'PouringAction'], ['status', 'FAILED'],
    ['started', formatOffset(node(POUR, 3).started)], ['ended', formatOffset(node(POUR, 3).ended)],
  ]);
});

test('a step that was asked for nothing in particular is titled for its kind', function () {
  assert.strictEqual(describeNode(node(POUR, 0)).title, 'sequence');
});

test('a step stands as it ended', function () {
  assert.strictEqual(describeNode(node(POUR, 3)).status, STATUS.FAILED);
  assert.strictEqual(describeNode(node(POUR, 1)).status, STATUS.OK);
});

test('a step that never began or has not ended says so', function () {
  const waiting = Object.assign({}, node(POUR, 3), { status: 'NOT_STARTED', started: null, ended: null });
  const view = describeNode(waiting);
  assert.strictEqual(view.status, '');
  assert.strictEqual(fact(view, 'started'), NEVER_STARTED);
  assert.strictEqual(fact(view, 'ended'), NOT_ENDED);
});

test('a step of a NEEM leads to no other NEEM', function () {
  assert.strictEqual(describeNode(node(POUR, 3)).neem, null);
});

test('the values of a step are written for reading, those not set as such', function () {
  const values = block(describeNode(node(POUR, 3)), 'Values');
  assert.strictEqual(values.kind, BLOCK.FACTS);
  const asked = new Map(values.facts);
  assert.strictEqual(asked.get('tilt_angle'), '1.85');
  assert.strictEqual(asked.get('pour_side'), NOT_SET);
});

test('a choice from an enumeration is written as its member', function () {
  const values = block(describeNode(node(NAVIGATION, 2)), 'Values');
  assert.deepStrictEqual(values.facts, [['torso_state', 'HIGH']]);
});

test('the things a step refers to are listed by what it uses them as', function () {
  const things = block(describeNode(node(POUR, 3)), 'Things and places');
  assert.deepStrictEqual(things.facts, [
    ['target_container', 'Bowl · Bowl'],
    ['source_container', 'PouringCup · PouringCup'],
    ['arm', 'PR2RightArm · PR2RightArm'],
  ]);
});

test('each thing of a list is listed', function () {
  const things = block(describeNode(node(NAVIGATION, 1)), 'Things and places');
  assert.deepStrictEqual(things.facts, [['arms', 'PR2LeftArm · PR2LeftArm'], ['arms', 'PR2RightArm · PR2RightArm']]);
});

test('an object that is only values has a block of its own with them', function () {
  const grasp = block(describeNode(node(POUR, 1)), 'Values of grasp');
  assert.strictEqual(grasp.kind, BLOCK.FACTS);
  assert.deepStrictEqual(grasp.facts, [['approach_direction', 'FRONT'], ['manipulation_offset', '0.1']]);
});

test('a step that refers to nothing has no block of things', function () {
  assert.strictEqual(block(describeNode(node(POUR, 4)), 'Things and places'), undefined);
});

test('a step with no values has no block of values', function () {
  assert.strictEqual(block(describeNode(node(POUR, 0)), 'Values'), undefined);
});

test('a step is given as it is kept, to copy', function () {
  const kept = block(describeNode(node(POUR, 3)), 'As it is kept');
  assert.strictEqual(kept.kind, BLOCK.JSON);
  assert.deepStrictEqual(kept.value, node(POUR, 3).stored);
});
