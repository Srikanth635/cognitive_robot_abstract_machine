// Tests of filtering the spans of a run: by what they say, and by whether they went wrong.
import test from 'node:test';
import assert from 'node:assert';
import { KIND, LANE, STATUS } from '../../../llm_reasoner_viz/src/llm_reasoner_viz/web/model.mjs';
import { SpanFilter } from '../../../llm_reasoner_viz/src/llm_reasoner_viz/web/filters.mjs';
import { actionResolution, loadFixture, modelOf, spansOfKind } from './support.mjs';

const resolved = loadFixture('resolved_run.json');

function filterOf(settings) {
  return Object.assign(new SpanFilter(), settings);
}

test('a filter that asks for nothing is not active and lets every span through', function () {
  const model = modelOf(resolved);
  const filter = new SpanFilter();
  assert.strictEqual(filter.active, false);
  assert.strictEqual(filter.matching(model).length, model.spans.length);
});

test('a search of nothing but spaces asks for nothing', function () {
  assert.strictEqual(filterOf({ text: '   ' }).active, false);
});

test('a search finds the spans whose events say the words', function () {
  const model = modelOf(resolved);
  const matching = filterOf({ text: 'tilt_angle' }).matching(model);
  const attempts = model.childrenOf(actionResolution(model, resolved)).filter(function (span) { return span.kind === KIND.ATTEMPT; });
  const callsAboutTheAction = attempts.flatMap(function (attempt) {
    return model.childrenOf(attempt).filter(function (span) { return span.kind === KIND.MODEL; });
  });
  assert.strictEqual(callsAboutTheAction.length, attempts.length);
  callsAboutTheAction.forEach(function (call) { assert.ok(matching.includes(call), call.id); });
  const toWorld = spansOfKind(model, KIND.CALL).find(function (span) { return span.lane === LANE.WORLD; });
  assert.ok(!matching.includes(toWorld));
});

test('a search finds what an attempt was rejected with', function () {
  const model = modelOf(resolved);
  const rejected = spansOfKind(model, KIND.ATTEMPT).find(function (span) { return span.status === STATUS.REJECTED; });
  assert.ok(filterOf({ text: 'tilt_angle' }).matches(rejected));
});

test('a search does not mind capital letters', function () {
  const model = modelOf(resolved);
  assert.deepStrictEqual(filterOf({ text: 'TILT_ANGLE' }).matching(model), filterOf({ text: 'tilt_angle' }).matching(model));
});

test('a search finds the spans by what they are called', function () {
  const model = modelOf(resolved);
  const matching = filterOf({ text: 'trial' }).matching(model);
  spansOfKind(model, KIND.TRIAL).forEach(function (trial) { assert.ok(matching.includes(trial)); });
});

test('a search for words nobody wrote finds nothing', function () {
  assert.deepStrictEqual(filterOf({ text: 'zzz-not-written-anywhere' }).matching(modelOf(resolved)), []);
});

test('only the problems are the spans that were rejected or failed', function () {
  const model = modelOf(resolved);
  const matching = filterOf({ problemsOnly: true }).matching(model);
  assert.ok(matching.length > 0);
  assert.ok(matching.every(function (span) { return span.status === STATUS.REJECTED || span.status === STATUS.FAILED; }));
  const rejectedAttempt = spansOfKind(model, KIND.ATTEMPT).find(function (span) { return span.status === STATUS.REJECTED; });
  assert.ok(matching.includes(rejectedAttempt));
  assert.ok(!matching.includes(spansOfKind(model, KIND.RUN)[0]));
});

test('a search of only the problems finds what is both', function () {
  const model = modelOf(resolved);
  const matching = filterOf({ text: 'tilt_angle', problemsOnly: true }).matching(model);
  assert.ok(matching.length > 0);
  assert.ok(matching.every(function (span) { return span.status !== STATUS.OK; }));
  assert.ok(matching.length < filterOf({ text: 'tilt_angle' }).matching(model).length);
});

test('the next match is the first one that starts after the span asked from', function () {
  const model = modelOf(resolved);
  const filter = filterOf({ problemsOnly: true });
  const matching = filter.matching(model).sort(function (a, b) { return a.start - b.start; });
  assert.strictEqual(filter.next(model, null), matching[0].id);
  assert.strictEqual(filter.next(model, matching[0].id), matching[1].id);
});

test('the next match after the last one is the first again', function () {
  const model = modelOf(resolved);
  const filter = filterOf({ problemsOnly: true });
  const matching = filter.matching(model).sort(function (a, b) { return a.start - b.start; });
  assert.strictEqual(filter.next(model, matching[matching.length - 1].id), matching[0].id);
});

test('there is no next match when nothing matches', function () {
  assert.strictEqual(filterOf({ text: 'zzz-not-written-anywhere' }).next(modelOf(resolved), null), null);
});
