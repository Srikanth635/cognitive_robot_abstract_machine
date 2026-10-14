// Tests of what the map of the components shows: how much each did and what is going on.
import test from 'node:test';
import assert from 'node:assert';
import { LANE } from '../../../llm_reasoner_viz/src/llm_reasoner_viz/web/model.mjs';
import { mapState, NODES, EDGES } from '../../../llm_reasoner_viz/src/llm_reasoner_viz/web/map.mjs';
import { loadFixture, modelOf } from './support.mjs';

const resolved = loadFixture('resolved_run.json');

function node(state, lane) { return state.nodes.filter(function (n) { return n.lane === lane; })[0]; }

test('every component of the system is a node', function () {
  const state = mapState(modelOf(resolved), 0);
  assert.deepStrictEqual(state.nodes.map(function (n) { return n.lane; }).sort(),
    [LANE.KNOWLEDGE, LANE.MODEL, LANE.RESOLVER, LANE.SIMULATION, LANE.WORLD].sort());
  assert.strictEqual(NODES.length, 5);
});

test('a node counts the spans of its component', function () {
  const model = modelOf(resolved);
  const state = mapState(model, 0);
  assert.strictEqual(node(state, LANE.MODEL).calls, model.spans.filter(function (s) { return s.lane === LANE.MODEL; }).length);
  assert.strictEqual(node(state, LANE.SIMULATION).calls, 2);
});

test('a run that has ended has no component at work', function () {
  const state = mapState(modelOf(resolved), 0);
  assert.ok(state.nodes.every(function (n) { return n.open === 0 && n.activity === null; }));
});

test('a component at work says what it is doing', function () {
  const state = mapState(modelOf(resolved.slice(0, 7)), 0);
  const model = node(state, LANE.MODEL);
  assert.strictEqual(model.open, 1);
  assert.strictEqual(model.activity, LANE.RESOLVER + ' asks');
});

test('an edge has the number of calls between its components', function () {
  const state = mapState(modelOf(resolved), 0);
  const edge = state.edges.filter(function (e) { return e.from === LANE.RESOLVER && e.to === LANE.MODEL; })[0];
  assert.strictEqual(edge.count, 3);
});

test('every edge joins two nodes', function () {
  const lanes = NODES.map(function (n) { return n.lane; });
  assert.ok(EDGES.every(function (e) { return lanes.includes(e.from) && lanes.includes(e.to); }));
});

test('an edge pulses while a call over it has just been made', function () {
  const model = modelOf(resolved);
  const last = model.lastAt;
  const recent = mapState(model, last + 0.3);
  const calm = mapState(model, last + 30);
  assert.ok(recent.edges.some(function (e) { return e.pulsing; }));
  assert.ok(calm.edges.every(function (e) { return !e.pulsing; }));
});
