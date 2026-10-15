// Tests of what the page knows about the NEEMs before it draws them: where to ask for them,
// which are shown, and how each is worded.
import test from 'node:test';
import assert from 'node:assert';
import { NEEM_KIND, OUTCOME, STATUS } from '../../../llm_reasoner_viz/src/llm_reasoner_viz/web/model.mjs';
import {
  NEEM_FILTER, NEEM_SCOPE, PLAN_STATUS, describeNeem, kindCounts, neemsAddress, shownNeems,
} from '../../../llm_reasoner_viz/src/llm_reasoner_viz/web/neems.mjs';

// A plan as the server lists it.
function planOf(neem, origin, overrides) {
  return Object.assign({
    neem: neem, status: PLAN_STATUS.SUCCEEDED, started: '2026-10-14T16:48:50', ended: '2026-10-14T16:48:52.500000',
    seconds: 2.5, nodes: 33, steps: 5, origin: origin,
  }, overrides || {});
}

function originOf(kind, overrides) {
  return Object.assign({
    kind: kind, episode: 'episode-1', run: 'run-a', recorded_at: 110.0, outcome: null, result: null,
  }, overrides || {});
}

const LANDED = { [OUTCOME.LANDS_IN_OPENING]: true, [OUTCOME.DISTANCE]: 0.005 };
const MISSED = { [OUTCOME.LANDS_IN_OPENING]: false, [OUTCOME.DISTANCE]: 0.189 };
const FAILED_MOTION = { [OUTCOME.LANDS_IN_OPENING]: false, [OUTCOME.MOTION_FAILED]: 'no solution' };

// %% where to ask

test('the NEEMs of the run shown are asked for by its name', function () {
  assert.strictEqual(neemsAddress(NEEM_SCOPE.RUN, 'run-a'), '/api/neems?run=run-a');
});

test('a name that is not plain is written so that the server reads it back', function () {
  const address = new URL(neemsAddress(NEEM_SCOPE.RUN, 'a b&c'), 'http://host');
  assert.strictEqual(address.searchParams.get('run'), 'a b&c');
});

test('all the NEEMs are asked for without a run', function () {
  assert.strictEqual(neemsAddress(NEEM_SCOPE.ALL, 'run-a'), '/api/neems');
});

// %% which are shown

const TRIAL = planOf(3, originOf(NEEM_KIND.TRIAL, { outcome: LANDED }));
const OTHER_TRIAL = planOf(2, originOf(NEEM_KIND.TRIAL, { outcome: MISSED }));
const EXECUTION = planOf(1, originOf(NEEM_KIND.EXECUTION, { result: 'finished' }));
const UNKNOWN = planOf(0, null);
const ALL = [TRIAL, OTHER_TRIAL, EXECUTION, UNKNOWN];

test('every NEEM is shown unless a kind is asked for', function () {
  assert.deepStrictEqual(shownNeems(ALL, NEEM_FILTER.ALL), ALL);
});

test('only the NEEMs of the kind asked for are shown', function () {
  assert.deepStrictEqual(shownNeems(ALL, NEEM_FILTER.TRIALS), [TRIAL, OTHER_TRIAL]);
  assert.deepStrictEqual(shownNeems(ALL, NEEM_FILTER.EXECUTIONS), [EXECUTION]);
});

test('a NEEM no log names is of neither kind', function () {
  assert.ok(!shownNeems(ALL, NEEM_FILTER.TRIALS).includes(UNKNOWN));
  assert.ok(!shownNeems(ALL, NEEM_FILTER.EXECUTIONS).includes(UNKNOWN));
});

test('each kind is counted, and all of them', function () {
  assert.deepStrictEqual(kindCounts(ALL), {
    [NEEM_FILTER.ALL]: 4, [NEEM_FILTER.TRIALS]: 2, [NEEM_FILTER.EXECUTIONS]: 1,
  });
});

// %% how each is worded

test('a NEEM is named by its number', function () {
  assert.strictEqual(describeNeem(TRIAL).title, '#3');
});

test('a pour that landed is done, and says how far from the centre of the opening', function () {
  const described = describeNeem(TRIAL);
  assert.strictEqual(described.origin, NEEM_KIND.TRIAL);
  assert.strictEqual(described.verdict, 'landed');
  assert.strictEqual(described.standing, STATUS.OK);
  assert.ok(described.facts.includes('0.005 m from the centre'));
});

test('a pour that missed was rejected', function () {
  const described = describeNeem(OTHER_TRIAL);
  assert.strictEqual(described.verdict, 'missed');
  assert.strictEqual(described.standing, STATUS.REJECTED);
  assert.ok(described.facts.includes('0.189 m from the centre'));
});

test('a pour whose motion failed is failed and has no distance to say', function () {
  const described = describeNeem(planOf(4, originOf(NEEM_KIND.TRIAL, { outcome: FAILED_MOTION })));
  assert.strictEqual(described.verdict, 'motion failed');
  assert.strictEqual(described.standing, STATUS.FAILED);
  assert.ok(!described.facts.some(function (fact) { return fact.includes('centre'); }));
});

test('a pour whose trial has not ended says nothing of how it went', function () {
  const described = describeNeem(planOf(5, originOf(NEEM_KIND.TRIAL)));
  assert.strictEqual(described.verdict, null);
  assert.strictEqual(described.standing, null);
});

test('an execution is done when it finished and failed when it failed', function () {
  assert.strictEqual(describeNeem(EXECUTION).verdict, 'finished');
  assert.strictEqual(describeNeem(EXECUTION).standing, STATUS.OK);
  const failed = describeNeem(planOf(6, originOf(NEEM_KIND.EXECUTION, { result: 'failed' })));
  assert.strictEqual(failed.verdict, 'failed');
  assert.strictEqual(failed.standing, STATUS.FAILED);
});

test('a NEEM no log names has no origin, and goes by how its plan ended', function () {
  const described = describeNeem(UNKNOWN);
  assert.strictEqual(described.origin, null);
  assert.strictEqual(described.verdict, 'succeeded');
  assert.strictEqual(described.standing, STATUS.OK);
  assert.strictEqual(described.run, null);
  assert.strictEqual(described.spanId, null);
  const failed = describeNeem(planOf(7, null, { status: PLAN_STATUS.FAILED }));
  assert.strictEqual(failed.standing, STATUS.FAILED);
});

test('the size and the duration of the plan are facts of it', function () {
  assert.deepStrictEqual(describeNeem(EXECUTION).facts, ['5 steps', '33 nodes', '2.50 s']);
});

test('one step is not many', function () {
  const single = describeNeem(planOf(8, null, { steps: 1, nodes: 1 }));
  assert.ok(single.facts.includes('1 step'));
  assert.ok(single.facts.includes('1 node'));
});

test('a plan that has not ended has no duration', function () {
  const described = describeNeem(planOf(9, null, { seconds: null, ended: null }));
  assert.deepStrictEqual(described.facts, ['5 steps', '33 nodes']);
});

test('where a NEEM came from is the run and the bar of its trial or execution', function () {
  const described = describeNeem(TRIAL);
  assert.strictEqual(described.run, 'run-a');
  assert.strictEqual(described.spanId, 'trial:episode-1');
  assert.strictEqual(describeNeem(EXECUTION).spanId, 'execution:episode-1');
  assert.strictEqual(described.recordedAt, 110.0);
});
