// Tests on the events of a run that failed in the middle of its second attempt, taken from a
// real resolution: how the failure shows from the model call up to the run, and what the
// attempt before it left.
import test from 'node:test';
import assert from 'node:assert';
import { EVENT, KIND, STATUS, USAGE, inputTokens } from '../../../llm_reasoner_viz/src/llm_reasoner_viz/web/model.mjs';
import { attemptCards } from '../../../llm_reasoner_viz/src/llm_reasoner_viz/web/attempts.mjs';
import { BLOCK, describeSpan } from '../../../llm_reasoner_viz/src/llm_reasoner_viz/web/inspector.mjs';
import { plotOfTrials, triedPours } from '../../../llm_reasoner_viz/src/llm_reasoner_viz/web/pour.mjs';
import { actionResolution, loadFixture, modelOf, spansOfKind } from './support.mjs';

const failed = loadFixture('failed_run.json');

function eventsOf(kind) {
  return failed.filter(function (event) { return event.event === kind; });
}

test('a model call that failed fails the attempt it was made in, the resolution and the run', function () {
  const model = modelOf(failed);
  const [failure] = eventsOf(EVENT.MODEL_FAILED);
  const call = model.byId.get('model:' + failure.call);
  const attempt = model.byId.get(call.parent);
  const resolution = model.byId.get(attempt.parent);
  assert.strictEqual(call.status, STATUS.FAILED);
  assert.strictEqual(attempt.kind, KIND.ATTEMPT);
  assert.strictEqual(attempt.status, STATUS.FAILED);
  assert.strictEqual(resolution.status, STATUS.FAILED);
  assert.strictEqual(spansOfKind(model, KIND.RUN)[0].status, STATUS.FAILED);
});

test('the inspector says why the model call failed', function () {
  const model = modelOf(failed);
  const [failure] = eventsOf(EVENT.MODEL_FAILED);
  const view = describeSpan(model.byId.get('model:' + failure.call), model);
  assert.strictEqual(view.status, STATUS.FAILED);
  assert.strictEqual(view.facts.find(function (fact) { return fact[0] === 'failed with'; })[1], failure.error_type);
  assert.strictEqual(view.blocks.find(function (block) { return block.title === 'Failure'; }).text, failure.reason);
});

test('the attempt before the failure was rejected, and the one that failed chose nothing', function () {
  const model = modelOf(failed);
  const [rejected, broken] = attemptCards(model, actionResolution(model, failed));
  assert.strictEqual(rejected.status, STATUS.REJECTED);
  assert.ok(rejected.chose.length > 0);
  assert.ok(rejected.outcome.startsWith('missed'));
  assert.strictEqual(broken.status, STATUS.FAILED);
  assert.deepStrictEqual(broken.chose, []);
  assert.strictEqual(broken.outcome, null);
});

test('the pour that missed is on the plan with the opening it missed, as the simulation measured it', function () {
  const model = modelOf(failed);
  const [trial] = eventsOf(EVENT.TRIAL_FINISHED);
  const plot = plotOfTrials(triedPours(model, actionResolution(model, failed)));
  assert.strictEqual(plot.points.length, 1);
  assert.strictEqual(plot.points[0].lands, trial.outcome.lands_in_opening);
  assert.strictEqual(plot.opening.radius, trial.outcome.opening_radius * plot.scale);
});

test('the run totals the tokens of the answers it got, those of the cache included, and what they cost', function () {
  const answered = eventsOf(EVENT.MODEL_ANSWERED);
  const model = modelOf(failed);
  assert.strictEqual(model.totals.modelCalls, answered.length);
  assert.strictEqual(model.totals.inputTokens, answered.reduce(function (sum, event) { return sum + inputTokens(event.usage); }, 0));
  assert.strictEqual(model.totals.outputTokens, answered.reduce(function (sum, event) { return sum + event.usage[USAGE.OUTPUT]; }, 0));
  assert.ok(model.totals.inputTokens > answered.reduce(function (sum, event) { return sum + event.usage[USAGE.INPUT]; }, 0));
});

test('a tool the model called is a span of the component that owns the tool', function () {
  const model = modelOf(failed);
  const tools = spansOfKind(model, KIND.TOOL);
  assert.ok(tools.length > 0);
  tools.forEach(function (tool) { assert.strictEqual(tool.lane, tool.data.called.owner); });
});

test('what the model asked for during an attempt is listed with the attempt, whether or not it was answered', function () {
  const model = modelOf(failed);
  const [, broken] = attemptCards(model, actionResolution(model, failed));
  const view = describeSpan(model.byId.get(broken.id), model);
  const during = view.blocks.find(function (block) { return block.title === 'During the attempt'; });
  assert.strictEqual(during.kind, BLOCK.LIST);
  assert.ok(during.items.some(function (item) { return item.label.startsWith('model call'); }));
});
