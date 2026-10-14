// Tests of drawing where pours ended: a plan of the floor around the opening, seen from the
// robot, with the opening as a circle and each pour as a point.
import test from 'node:test';
import assert from 'node:assert';
import { EVENT, OUTCOME } from '../../../llm_reasoner_viz/src/llm_reasoner_viz/web/model.mjs';
import { pourPlot, triedPours } from '../../../llm_reasoner_viz/src/llm_reasoner_viz/web/pour.mjs';
import { eventOf, modelOf, pourEvents, pourOutcome } from './support.mjs';

const SIZE = 200;
const RADIUS = 0.04;

function trialOf(label, where) {
  return { id: label, label: label, outcome: pourOutcome(Object.assign({ radius: RADIUS, above: 0.09 }, where)) };
}

test('the opening is a circle of its radius in the middle of the plan', function () {
  const plot = pourPlot([trialOf('a', { ahead: 0.01, right: 0.01 })], SIZE);
  assert.strictEqual(plot.opening.x, SIZE / 2);
  assert.strictEqual(plot.opening.y, SIZE / 2);
  assert.strictEqual(plot.opening.radius, RADIUS * plot.scale);
});

test('a pour that ended to the right of the opening is drawn to its right', function () {
  const [point] = pourPlot([trialOf('a', { ahead: 0, right: 0.05 })], SIZE).points;
  assert.ok(point.x > SIZE / 2);
  assert.strictEqual(point.y, SIZE / 2);
});

test('a pour that ended ahead of the opening is drawn above it', function () {
  const [point] = pourPlot([trialOf('a', { ahead: 0.05, right: 0 })], SIZE).points;
  assert.ok(point.y < SIZE / 2);
  assert.strictEqual(point.x, SIZE / 2);
});

test('a pour that landed is drawn inside the opening, and one that missed outside it', function () {
  const plot = pourPlot([trialOf('landed', { ahead: 0.01, right: 0.01 }), trialOf('missed', { ahead: 0.02, right: 0.18 })], SIZE);
  const [landed, missed] = plot.points;
  const distance = function (point) { return Math.hypot(point.x - plot.opening.x, point.y - plot.opening.y); };
  assert.ok(landed.lands && distance(landed) <= plot.opening.radius);
  assert.ok(!missed.lands && distance(missed) > plot.opening.radius);
});

test('the plan reaches far enough to hold the furthest pour', function () {
  const plot = pourPlot([trialOf('near', { ahead: 0.01, right: 0 }), trialOf('far', { ahead: -0.3, right: 0.4 })], SIZE);
  plot.points.forEach(function (point) {
    assert.ok(point.x >= 0 && point.x <= SIZE && point.y >= 0 && point.y <= SIZE, point.label);
  });
});

test('the plan holds the whole opening even when every pour landed in its middle', function () {
  const plot = pourPlot([trialOf('a', { ahead: 0, right: 0 })], SIZE);
  assert.ok(plot.opening.radius < SIZE / 2);
});

test('a pour keeps its name, its height above the opening and whether it landed', function () {
  const [point] = pourPlot([trialOf('trial 1', { ahead: 0.01, right: 0, above: 0.12 })], SIZE).points;
  assert.strictEqual(point.label, 'trial 1');
  assert.strictEqual(point.id, 'trial 1');
  assert.strictEqual(point.above, 0.12);
  assert.strictEqual(point.lands, true);
});

test('a pour with no place, because its motion failed or it was not measured, is left out', function () {
  const unplaced = [{ id: 'x', label: 'x', outcome: { [OUTCOME.LANDS_IN_OPENING]: false, [OUTCOME.MOTION_FAILED]: 'out of reach' } },
    { id: 'y', label: 'y', outcome: { [OUTCOME.LANDS_IN_OPENING]: true } }];
  const plot = pourPlot(unplaced.concat([trialOf('a', { ahead: 0, right: 0.02 })]), SIZE);
  assert.deepStrictEqual(plot.points.map(function (point) { return point.id; }), ['a']);
});

test('when no pour has a place there is no plan', function () {
  assert.strictEqual(pourPlot([{ id: 'y', label: 'y', outcome: { [OUTCOME.LANDS_IN_OPENING]: true } }], SIZE), null);
  assert.strictEqual(pourPlot([], SIZE), null);
});

test('the pours tried during a resolution are those of its attempts, in the order they were tried', function () {
  const outcomes = [{ ahead: 0.02, right: 0.18 }, { ahead: 0.005, right: 0.01 }].map(function (where) {
    return pourOutcome(Object.assign({ radius: RADIUS, above: 0.09 }, where));
  });
  const model = modelOf([
    eventOf(EVENT.RESOLUTION_STARTED, 0, { resolution: 'r', purpose: 'action', action: 'a.B', slots: [], max_repairs: 1 }),
    eventOf(EVENT.ATTEMPT_STARTED, 0.5, { resolution: 'r', attempt: 0 }),
    ...pourEvents(outcomes),
    eventOf(EVENT.RESOLUTION_FINISHED, 30, { resolution: 'r', attempts: 1, instances: [], justifications: [] }),
  ]);
  const tried = triedPours(model, model.byId.get('resolution:r'));
  assert.deepStrictEqual(tried.map(function (span) { return span.data.finished.outcome; }), outcomes);
});

test('a pour that has not ended is not among the pours tried', function () {
  const model = modelOf([
    eventOf(EVENT.RESOLUTION_STARTED, 0, { resolution: 'r', purpose: 'action', action: 'a.B', slots: [], max_repairs: 1 }),
    eventOf(EVENT.ATTEMPT_STARTED, 0.5, { resolution: 'r', attempt: 0 }),
    eventOf(EVENT.TRIAL_STARTED, 1, { trial: 'pour', values: {} }),
  ]);
  assert.deepStrictEqual(triedPours(model, model.byId.get('resolution:r')), []);
});

test('there are no pours tried without a resolution', function () {
  assert.deepStrictEqual(triedPours(modelOf([]), null), []);
});
