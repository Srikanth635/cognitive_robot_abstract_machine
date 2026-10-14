// Tests of building the spans of a run from its events: what each is, where it lies, and
// how it ended.
import test from 'node:test';
import assert from 'node:assert';
import { RunModel, LANE, STATUS, KIND, EVENT, NEEM_KIND, USAGE, edgeKey, spanEnd, humanize } from '../../../llm_reasoner_viz/src/llm_reasoner_viz/web/model.mjs';
import { actionResolution, eventOf, loadFixture, modelOf, situationResolution, spansOfKind } from './support.mjs';

const of = spansOfKind;

const resolved = loadFixture('resolved_run.json');
const executed = loadFixture('executed_run.json');

// How the events of a piece of work that failed say it did.
const FAILURE = { error_type: 'ScriptExhausted', reason: 'none left', seconds: 1 };

test('a run is one span from its start to its end', function () {
  const model = modelOf(resolved);
  const [run] = of(model, KIND.RUN);
  assert.strictEqual(run.status, STATUS.OK);
  assert.strictEqual(run.start, resolved[0].at);
  assert.strictEqual(run.end, resolved[resolved.length - 1].at);
  assert.ok(run.data.started.query.startsWith('the(PouringAction)'));
});

test('a run still going has a span with no end', function () {
  const model = modelOf(resolved.slice(0, 10));
  const [run] = of(model, KIND.RUN);
  assert.strictEqual(run.status, STATUS.RUNNING);
  assert.strictEqual(run.end, null);
});

test('a run that failed has a failed span', function () {
  const model = modelOf([eventOf(EVENT.RUN_STARTED, 1, { query: 'q', scene: 's' }), eventOf(EVENT.RUN_FAILED, 5, FAILURE)]);
  assert.strictEqual(of(model, KIND.RUN)[0].status, STATUS.FAILED);
});

test('a model call is on the language model lane, from asked to answered, named for who asked', function () {
  const model = modelOf(resolved);
  const calls = of(model, KIND.MODEL);
  assert.strictEqual(calls.length, 4);
  assert.ok(calls.every(function (call) { return call.lane === LANE.MODEL && call.status === STATUS.OK; }));
  assert.deepStrictEqual(calls.map(function (call) { return call.data.asked.source; }), [LANE.RESOLVER, LANE.KNOWLEDGE, LANE.RESOLVER, LANE.RESOLVER]);
  const asked = resolved.filter(function (e) { return e.event === EVENT.MODEL_ASKED; })[0];
  const answered = resolved.filter(function (e) { return e.event === EVENT.MODEL_ANSWERED; })[0];
  assert.strictEqual(calls[0].start, asked.at);
  assert.strictEqual(calls[0].end, answered.at);
});

test('a model call that failed is failed', function () {
  const model = modelOf([
    eventOf(EVENT.MODEL_ASKED, 1, { call: 'c', source: LANE.RESOLVER, style: 'plain', identity: 'i', system: 's', user: 'u', tools: [] }),
    eventOf(EVENT.MODEL_FAILED, 2, Object.assign({ call: 'c' }, FAILURE)),
  ]);
  const [call] = of(model, KIND.MODEL);
  assert.strictEqual(call.status, STATUS.FAILED);
  assert.strictEqual(call.data.failure.error_type, FAILURE.error_type);
});

test('a call between components is on the lane of the component called', function () {
  const model = modelOf(resolved);
  const calls = of(model, KIND.CALL);
  const toWorld = calls.filter(function (call) { return call.lane === LANE.WORLD; });
  const toKnowledge = calls.filter(function (call) { return call.lane === LANE.KNOWLEDGE; });
  assert.ok(toWorld.length >= 1);
  assert.ok(toKnowledge.length >= 1);
  assert.ok(calls.every(function (call) { return call.status === STATUS.OK && call.end !== null; }));
  const firstCallToKnowledge = resolved.find(function (e) { return e.event === EVENT.COMPONENT_CALLED && e.target === LANE.KNOWLEDGE; });
  assert.strictEqual(toKnowledge[0].label, firstCallToKnowledge.operation);
  assert.strictEqual(toKnowledge[0].data.called.source, LANE.RESOLVER);
});

test('a resolution has its attempts, the last accepted and the ones before it rejected', function () {
  const model = modelOf(resolved);
  const action = actionResolution(model, resolved);
  const attempts = of(model, KIND.ATTEMPT).filter(function (a) { return a.parent === action.id; });
  assert.deepStrictEqual(attempts.map(function (a) { return a.status; }), [STATUS.REJECTED, STATUS.OK]);
  assert.strictEqual(action.status, STATUS.OK);
  assert.deepStrictEqual(attempts.map(function (a) { return a.label; }), ['attempt 1', 'attempt 2']);
  const rejection = resolved.find(function (e) { return e.event === EVENT.ATTEMPT_REJECTED; });
  assert.strictEqual(attempts[0].data.rejected.kind, rejection.kind);
});

test('the choice of the situation is a resolution of its own purpose', function () {
  const model = modelOf(resolved);
  const situation = situationResolution(model, resolved);
  assert.notStrictEqual(situation.data.started.purpose, actionResolution(model, resolved).data.started.purpose);
  assert.strictEqual(situation.status, STATUS.OK);
  assert.strictEqual(model.childrenOf(situation).length, 1);
});

test('what the resolver asked the model during an attempt belongs to that attempt', function () {
  const model = modelOf(resolved);
  const calls = of(model, KIND.MODEL).filter(function (c) { return c.data.asked.source === LANE.RESOLVER; });
  const parents = calls.map(function (c) { return model.byId.get(c.parent).kind; });
  assert.ok(parents.every(function (kind) { return kind === KIND.ATTEMPT; }));
});

test('the consultation of the knowledge is kept with the resolution that made it', function () {
  const model = modelOf(resolved);
  const consulted = resolved.find(function (e) { return e.event === EVENT.KNOWLEDGE_CONSULTED; });
  assert.strictEqual(actionResolution(model, resolved).data.consulted.record_id, consulted.record_id);
});

test('a hypothesis has the attempt that generated it and is served', function () {
  const model = modelOf(resolved);
  const [hypothesis] = of(model, KIND.HYPOTHESIS);
  assert.strictEqual(hypothesis.lane, LANE.KNOWLEDGE);
  assert.strictEqual(hypothesis.status, STATUS.OK);
  assert.strictEqual(hypothesis.data.served.from_store, false);
  const attempts = of(model, KIND.GENERATION_ATTEMPT);
  assert.strictEqual(attempts.length, 1);
  assert.strictEqual(attempts[0].parent, hypothesis.id);
  assert.strictEqual(attempts[0].status, STATUS.OK);
});

test('what the knowledge component asked the model belongs to its generation attempt', function () {
  const model = modelOf(resolved);
  const call = of(model, KIND.MODEL).filter(function (c) { return c.data.asked.source === LANE.KNOWLEDGE; })[0];
  assert.strictEqual(model.byId.get(call.parent).kind, KIND.GENERATION_ATTEMPT);
});

test('a generation attempt that was rejected is rejected, and a failed generation fails its hypothesis', function () {
  const model = modelOf([
    eventOf(EVENT.HYPOTHESIS_REQUESTED, 1, { root_type: 'a.B', fixed_values: {}, instruction: 'i' }),
    eventOf(EVENT.GENERATION_ATTEMPT_STARTED, 2, { attempt: 0 }),
    eventOf(EVENT.GENERATION_ATTEMPT_REJECTED, 3, { attempt: 0, problems: ['loops'] }),
    eventOf(EVENT.GENERATION_FAILED, 4, { attempts: 1, problems: ['loops'] }),
  ]);
  assert.strictEqual(of(model, KIND.GENERATION_ATTEMPT)[0].status, STATUS.REJECTED);
  assert.strictEqual(of(model, KIND.HYPOTHESIS)[0].status, STATUS.FAILED);
});

test('a pour tried in the simulation is ok when it lands and rejected when it misses', function () {
  const model = modelOf(resolved);
  const trials = of(model, KIND.TRIAL);
  assert.deepStrictEqual(trials.map(function (t) { return t.status; }), [STATUS.REJECTED, STATUS.OK]);
  assert.ok(trials.every(function (t) { return t.lane === LANE.SIMULATION; }));
  assert.deepStrictEqual(trials.map(function (t) { return t.label; }), ['trial 1', 'trial 2']);
});

test('a trial whose simulation failed is failed', function () {
  const model = modelOf([
    eventOf(EVENT.TRIAL_STARTED, 1, { trial: 't', values: {} }),
    eventOf(EVENT.TRIAL_FINISHED, 2, { trial: 't', outcome: { lands_in_opening: false, motion_failed: 'out of reach' }, seconds: 1 }),
  ]);
  assert.strictEqual(of(model, KIND.TRIAL)[0].status, STATUS.FAILED);
});

test('a tool call is on the lane of the component that owns the tool', function () {
  const model = modelOf([
    eventOf(EVENT.MODEL_ASKED, 1, { call: 'm', source: LANE.RESOLVER, style: 'with_tools', identity: 'i', system: '', user: '', tools: ['find_individuals'] }),
    eventOf(EVENT.TOOL_CALLED, 2, { call: 't', parent: 'm', tool: 'find_individuals', arguments: { kind: 'Body' }, owner: LANE.WORLD }),
    eventOf(EVENT.TOOL_RETURNED, 3, { call: 't', content: '{}', is_error: false, seconds: 1 }),
    eventOf(EVENT.MODEL_ANSWERED, 4, { call: 'm', answer: 'a', seconds: 3 }),
  ]);
  const [tool] = of(model, KIND.TOOL);
  assert.strictEqual(tool.lane, LANE.WORLD);
  assert.strictEqual(tool.status, STATUS.OK);
  assert.strictEqual(tool.parent, 'model:m');
  assert.strictEqual(tool.label, 'find_individuals');
});

test('a tool nobody owns is on the language model lane', function () {
  const model = modelOf([
    eventOf(EVENT.TOOL_CALLED, 2, { call: 't', parent: 'm', tool: 'x', arguments: {}, owner: null }),
    eventOf(EVENT.TOOL_RETURNED, 3, { call: 't', content: '{}', is_error: false, seconds: 1 }),
  ]);
  assert.strictEqual(of(model, KIND.TOOL)[0].lane, LANE.MODEL);
});

test('a tool that answered with an error is rejected, and one that raised is failed', function () {
  const model = modelOf([
    eventOf(EVENT.TOOL_CALLED, 1, { call: 'a', parent: 'm', tool: 'x', arguments: {}, owner: LANE.WORLD }),
    eventOf(EVENT.TOOL_RETURNED, 2, { call: 'a', content: '{"error": "UnknownHandle"}', is_error: true, seconds: 1 }),
    eventOf(EVENT.TOOL_CALLED, 3, { call: 'b', parent: 'm', tool: 'y', arguments: {}, owner: LANE.WORLD }),
    eventOf(EVENT.TOOL_FAILED, 4, { call: 'b', error_type: 'RuntimeError', reason: 'broke', seconds: 1 }),
  ]);
  const tools = of(model, KIND.TOOL);
  assert.deepStrictEqual(tools.map(function (t) { return t.status; }), [STATUS.REJECTED, STATUS.FAILED]);
});

test('a refused tool call is a rejected moment', function () {
  const model = modelOf([eventOf(EVENT.TOOL_REFUSED, 5, { call: 'r', parent: 'm', tool: 'x', arguments: {}, content: '{"error": "InputBreaksSchema"}' })]);
  const [refused] = of(model, KIND.REFUSED);
  assert.strictEqual(refused.status, STATUS.REJECTED);
  assert.strictEqual(refused.start, refused.end);
});

test('a step ends when the next one starts, and the last when the execution ends', function () {
  const model = modelOf(executed);
  const steps = of(model, KIND.STEP);
  assert.deepStrictEqual(steps.map(function (s) { return s.label; }), ['park arms', 'move torso', 'navigate', 'pick up', 'pouring']);
  assert.ok(steps.every(function (s) { return s.status === STATUS.OK && s.end !== null; }));
  for (let index = 1; index < steps.length; index += 1) {
    assert.strictEqual(steps[index - 1].end, steps[index].start);
  }
  const [execution] = of(model, KIND.EXECUTION);
  assert.strictEqual(steps[steps.length - 1].end, execution.end);
  assert.strictEqual(execution.status, STATUS.OK);
  assert.ok(steps.every(function (s) { return s.parent === execution.id && s.lane === LANE.SIMULATION; }));
});

test('an execution that failed fails the step it was in', function () {
  const model = modelOf([
    eventOf(EVENT.EXECUTION_STARTED, 1, { run: 'r' }),
    eventOf(EVENT.STEP_CHANGED, 2, { run: 'r', step: 'NavigateAction', status: 'RUNNING' }),
    eventOf(EVENT.EXECUTION_FAILED, 3, { run: 'r', error_type: 'RuntimeError', reason: 'stuck', seconds: 2 }),
  ]);
  assert.strictEqual(of(model, KIND.STEP)[0].status, STATUS.FAILED);
  assert.strictEqual(of(model, KIND.EXECUTION)[0].status, STATUS.FAILED);
});

test('an open span lasts until now', function () {
  const model = modelOf(resolved.slice(0, 7));
  const [call] = of(model, KIND.MODEL);
  assert.strictEqual(call.end, null);
  assert.strictEqual(spanEnd(call, 99), 99);
  assert.strictEqual(spanEnd(of(modelOf(resolved), KIND.MODEL)[0], 99), of(modelOf(resolved), KIND.MODEL)[0].end);
});

test('calls between components are counted by the pair that made them', function () {
  const model = modelOf(resolved);
  assert.strictEqual(model.edges.get(edgeKey(LANE.RESOLVER, LANE.MODEL)), 3);
  assert.strictEqual(model.edges.get(edgeKey(LANE.KNOWLEDGE, LANE.MODEL)), 1);
  assert.ok(model.edges.get(edgeKey(LANE.RESOLVER, LANE.WORLD)) >= 1);
  assert.ok(model.edges.get(edgeKey(LANE.RESOLVER, LANE.KNOWLEDGE)) >= 1);
  assert.strictEqual(model.edges.get(edgeKey(LANE.RESOLVER, LANE.SIMULATION)), 2);
});

test('the answers of the models are summed', function () {
  const model = modelOf([
    eventOf(EVENT.MODEL_ASKED, 1, { call: 'a', source: LANE.RESOLVER, style: 'plain', identity: 'i', system: '', user: '', tools: [] }),
    eventOf(EVENT.MODEL_ANSWERED, 2, { call: 'a', answer: 'x', seconds: 1, usage: { input_tokens: 100, output_tokens: 20 }, cost: 0.5 }),
    eventOf(EVENT.MODEL_ASKED, 3, { call: 'b', source: LANE.KNOWLEDGE, style: 'plain', identity: 'i', system: '', user: '', tools: [] }),
    eventOf(EVENT.MODEL_ANSWERED, 4, { call: 'b', answer: 'y', seconds: 1, usage: { input_tokens: 10, output_tokens: 5 }, cost: null }),
  ]);
  assert.deepStrictEqual(model.totals, { modelCalls: 2, cost: 0.5, inputTokens: 110, outputTokens: 25 });
});

test('an event of a kind nobody knows is kept and makes no span', function () {
  const model = modelOf([eventOf('SomethingNew', 1, { a: 1 })]);
  assert.strictEqual(model.events.length, 1);
  assert.strictEqual(model.spans.length, 0);
});

test('adding the events one at a time gives the spans adding them all at once does', function () {
  const together = modelOf(resolved);
  const apart = new RunModel();
  resolved.forEach(function (e) { apart.add(e); });
  assert.deepStrictEqual(apart.spans.map(function (s) { return [s.id, s.status, s.start, s.end]; }),
    together.spans.map(function (s) { return [s.id, s.status, s.start, s.end]; }));
});

test('a run is dated from its first event to its last', function () {
  const model = modelOf(resolved);
  assert.strictEqual(model.firstAt, resolved[0].at);
  assert.strictEqual(model.lastAt, resolved[resolved.length - 1].at);
});

test('the name of an action is written for reading', function () {
  assert.strictEqual(humanize('ParkArmsAction'), 'park arms');
  assert.strictEqual(humanize('PouringAction'), 'pouring');
  assert.strictEqual(humanize('MoveTorsoAction'), 'move torso');
});

test('a line of the log that is not an event is kept and gives the run no date', function () {
  const model = modelOf([{ event: 'Unreadable', line: 'half a line' }, eventOf(EVENT.RUN_STARTED, 5, { run: 'r', query: 'q', scene: 's' })]);
  assert.strictEqual(model.events.length, 2);
  assert.strictEqual(model.firstAt, 5);
  assert.strictEqual(model.lastAt, 5);
});

test('the tokens read from and written to the cache are input tokens too', function () {
  const model = modelOf([
    eventOf(EVENT.MODEL_ASKED, 1, { call: 'a', source: LANE.RESOLVER, style: 'plain', identity: 'i', system: '', user: '', tools: [] }),
    eventOf(EVENT.MODEL_ANSWERED, 2, { call: 'a', answer: 'x', seconds: 1, cost: 0.1,
      usage: { [USAGE.INPUT]: 6, [USAGE.OUTPUT]: 389, [USAGE.CACHE_WRITTEN]: 13572, [USAGE.CACHE_READ]: 7745 } }),
  ]);
  assert.strictEqual(model.totals.inputTokens, 6 + 13572 + 7745);
  assert.strictEqual(model.totals.outputTokens, 389);
});

// %% NEEMs kept of what the simulation performed

const neemRun = loadFixture('neem_run.json');

function neemKept(kind, episode, fields) {
  return eventOf(EVENT.NEEM_RECORDED, 3, Object.assign({ kind: kind, episode: episode, database: '/data/neems.sqlite', database_id: 7, seconds: 5.5 }, fields || {}));
}

test('the NEEM kept of an execution is kept with the execution', function () {
  const model = modelOf(neemRun);
  const [execution] = of(model, KIND.EXECUTION);
  const kept = neemRun.find(function (event) { return event.event === EVENT.NEEM_RECORDED; });
  assert.strictEqual(execution.data.neem, kept);
  assert.ok(execution.events.includes(kept));
});

test('the NEEM kept of a pour tried is kept with the trial', function () {
  const model = modelOf([
    eventOf(EVENT.TRIAL_STARTED, 1, { trial: 't1', values: {} }),
    neemKept(NEEM_KIND.TRIAL, 't1'),
    eventOf(EVENT.TRIAL_FINISHED, 5, { trial: 't1', outcome: { lands_in_opening: true }, seconds: 4 }),
  ]);
  const [trial] = of(model, KIND.TRIAL);
  assert.strictEqual(trial.data.neem.database_id, 7);
  assert.strictEqual(trial.status, STATUS.OK);
});

test('the NEEMs kept are counted with the run', function () {
  const model = modelOf([
    eventOf(EVENT.TRIAL_STARTED, 1, { trial: 't1', values: {} }),
    neemKept(NEEM_KIND.TRIAL, 't1'),
    eventOf(EVENT.TRIAL_STARTED, 4, { trial: 't2', values: {} }),
    neemKept(NEEM_KIND.TRIAL, 't2'),
  ]);
  assert.strictEqual(model.neems.length, 2);
  assert.strictEqual(model.neemFailures.length, 0);
});

test('a NEEM that could not be kept is kept with its span and counted apart', function () {
  const failure = eventOf(EVENT.NEEM_NOT_RECORDED, 3, { kind: NEEM_KIND.TRIAL, episode: 't1', error_type: 'DiskIsFull', reason: 'no space left' });
  const model = modelOf([eventOf(EVENT.TRIAL_STARTED, 1, { trial: 't1', values: {} }), failure]);
  const [trial] = of(model, KIND.TRIAL);
  assert.strictEqual(trial.data.neemFailure, failure);
  assert.strictEqual(trial.data.neem, undefined);
  assert.strictEqual(model.neems.length, 0);
  assert.strictEqual(model.neemFailures.length, 1);
});

test('a NEEM of an episode the log does not hold is counted and attached to nothing', function () {
  const model = modelOf([neemKept(NEEM_KIND.EXECUTION, 'a-run-the-log-never-started')]);
  assert.strictEqual(model.neems.length, 1);
  assert.strictEqual(model.spans.length, 0);
});

test('a run that kept no NEEM has none', function () {
  assert.strictEqual(modelOf(resolved).neems.length, 0);
});
