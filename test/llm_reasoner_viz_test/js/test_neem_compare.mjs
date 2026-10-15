// Tests of comparing two NEEMs: which steps of one are those of the other, and what differs
// in the steps that are.
import test from 'node:test';
import assert from 'node:assert';
import { BLOCK } from '../../../llm_reasoner_viz/src/llm_reasoner_viz/web/inspector.mjs';
import { describeNode } from '../../../llm_reasoner_viz/src/llm_reasoner_viz/web/neem.mjs';
import {
  MARK, compareNeems, comparisonSummary, describeComparedNode, markOf,
} from '../../../llm_reasoner_viz/src/llm_reasoner_viz/web/neem_compare.mjs';
import { loadFixture } from './support.mjs';

const POUR = loadFixture('neem_pour.json').plan;
const NAVIGATION = loadFixture('neem_navigation.json').plan;

function node(nodes, id) {
  return nodes.find(function (candidate) { return candidate.id === id; });
}

function block(view, title) {
  return view.blocks.find(function (candidate) { return candidate.title === title; });
}

// A step written by hand, for what the fixtures do not hold.
function stepOf(id, parent, designator, parameters) {
  return {
    id: id, parent: parent, depth: parent === null ? 0 : 1, type: 'ActionNode', designator: designator,
    parameters: parameters || [], references: [], rows: [],
  };
}

const COMPARED = compareNeems(POUR, NAVIGATION, '#2');

// %% which steps are which

test('the steps of one plan are those of the other that have the same names the same way down', function () {
  assert.ok(!COMPARED.onlyHere.has(2));
  assert.ok(!COMPARED.onlyHere.has(3));
  assert.deepStrictEqual([...COMPARED.onlyHere], [4, 5]);
});

test('the steps only the other has are named', function () {
  assert.deepStrictEqual(COMPARED.onlyThere, [{ id: 13, label: 'NavigateAction' }]);
});

test('steps with the same name under the same step are matched in order', function () {
  const here = [stepOf(1, null, 'Root'), stepOf(2, 1, 'Move'), stepOf(3, 1, 'Move')];
  const there = [stepOf(10, null, 'Root'), stepOf(11, 10, 'Move')];
  const compared = compareNeems(here, there, '#other');
  assert.deepStrictEqual([...compared.onlyHere], [3]);
  assert.deepStrictEqual(compared.onlyThere, []);
});

test('a plan compared with itself differs nowhere', function () {
  const compared = compareNeems(POUR, POUR, '#1');
  assert.strictEqual(compared.changed.size, 0);
  assert.strictEqual(compared.onlyHere.size, 0);
  assert.deepStrictEqual(compared.onlyThere, []);
});

// %% what differs

test('a step whose values differ lists them, as it has them and as the other has them', function () {
  const pour = COMPARED.changed.get(3);
  assert.deepStrictEqual(pour.parameters, [
    { name: 'tilt_angle', here: '1.85', other: '1.6' },
    { name: 'pour_side_offset', here: '0', other: '-0.188' },
  ]);
  assert.deepStrictEqual(pour.references, []);
});

test('a step whose values are the same is not listed as changed', function () {
  assert.ok(!COMPARED.changed.has(2));
  assert.ok(!COMPARED.changed.has(1));
});

test('a thing the step refers to that differs is listed too', function () {
  const other = JSON.parse(JSON.stringify(NAVIGATION));
  node(other, 14).references[0].label = 'plate.stl';
  const compared = compareNeems(NAVIGATION, other, '#other');
  assert.deepStrictEqual(compared.changed.get(14).references, [
    { name: 'target_container', here: 'Body · bowl.stl', other: 'Body · plate.stl' },
  ]);
  assert.deepStrictEqual(compared.changed.get(14).parameters, []);
});

test('a value the other step does not have is said to be absent', function () {
  const here = [stepOf(1, null, 'Root', [{ name: 'speed', value: 1 }])];
  const there = [stepOf(2, null, 'Root', [])];
  const compared = compareNeems(here, there, '#other');
  assert.deepStrictEqual(compared.changed.get(1).parameters, [{ name: 'speed', here: '1', other: 'absent' }]);
});

// %% how it is shown

test('each step is marked as changed, as only here, or not at all', function () {
  assert.strictEqual(markOf(COMPARED, 3), MARK.CHANGED);
  assert.strictEqual(markOf(COMPARED, 4), MARK.ONLY_HERE);
  assert.strictEqual(markOf(COMPARED, 2), null);
});

test('a step that is no different says it is the same as the other', function () {
  const view = describeComparedNode(node(POUR, 2), COMPARED);
  assert.deepStrictEqual(block(view, 'Compared with #2').text, 'The same as in #2.');
});

test('a step that differs shows each value as it was and as it is in the other', function () {
  const view = describeComparedNode(node(POUR, 3), COMPARED);
  const compared = block(view, 'Compared with #2');
  assert.strictEqual(compared.kind, BLOCK.FACTS);
  assert.deepStrictEqual(compared.facts, [['tilt_angle', '1.85 → 1.6'], ['pour_side_offset', '0 → -0.188']]);
});

test('a step the other does not have says so', function () {
  const view = describeComparedNode(node(POUR, 5), COMPARED);
  assert.strictEqual(block(view, 'Compared with #2').text, 'Not in #2.');
});

test('a step is described as it is, with the comparison after it', function () {
  const alone = describeNode(node(POUR, 3));
  const compared = describeComparedNode(node(POUR, 3), COMPARED);
  assert.deepStrictEqual(compared.blocks.slice(0, alone.blocks.length), alone.blocks);
  assert.strictEqual(compared.blocks.length, alone.blocks.length + 1);
});

test('the comparison is summed up by how many steps differ and are not in the other', function () {
  assert.strictEqual(comparisonSummary(COMPARED), '1 step differs · 2 steps only here · 1 step only in #2');
});

test('plans that do not differ are said to be the same', function () {
  assert.strictEqual(comparisonSummary(compareNeems(POUR, POUR, '#1')), 'the same as #1');
});
