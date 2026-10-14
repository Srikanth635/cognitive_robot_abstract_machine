// Tests of what the inspector shows of a span: its facts and the blocks of detail.
import test from 'node:test';
import assert from 'node:assert';
import { KIND, LANE, STATUS, EVENT, NEEM_KIND, OUTCOME, USAGE } from '../../../llm_reasoner_viz/src/llm_reasoner_viz/web/model.mjs';
import { BLOCK, describeSpan, trialText } from '../../../llm_reasoner_viz/src/llm_reasoner_viz/web/inspector.mjs';
import { actionResolution, loadFixture, modelOf, pourEvents, pourOutcome, spansOfKind } from './support.mjs';

const resolved = loadFixture('resolved_run.json');

function spanOf(model, kind, test) { return model.spans.filter(function (s) { return s.kind === kind && (!test || test(s)); })[0]; }
function block(view, title) { return view.blocks.filter(function (b) { return b.title === title; })[0]; }
function fact(view, name) { return (view.facts.filter(function (f) { return f[0] === name; })[0] || [])[1]; }

test('every view says what the span is, and how it ended', function () {
  const model = modelOf(resolved);
  const view = describeSpan(spanOf(model, KIND.MODEL), model);
  assert.ok(view.title);
  assert.strictEqual(view.status, STATUS.OK);
});

test('a model call shows who asked, the way, and what it took', function () {
  const model = modelOf(resolved);
  const call = spanOf(model, KIND.MODEL);
  const view = describeSpan(call, model);
  assert.strictEqual(fact(view, 'asked by'), call.data.asked.source);
  assert.strictEqual(fact(view, 'style'), call.data.asked.style);
  assert.ok(fact(view, 'took').endsWith('s'));
});

test('a model call shows its prompts as the sections they are made of', function () {
  const model = modelOf(resolved);
  const view = describeSpan(spanOf(model, KIND.MODEL), model);
  const system = block(view, 'System prompt');
  assert.strictEqual(system.kind, BLOCK.SECTIONS);
  assert.ok(system.sections.map(function (s) { return s.name; }).includes('role'));
  assert.strictEqual(block(view, 'User prompt').kind, BLOCK.SECTIONS);
});

test('a model call shows its answer, the tools it was offered and the schema it had to meet', function () {
  const model = modelOf(resolved);
  const structured = spanOf(model, KIND.MODEL, function (s) { return s.data.asked.source === LANE.KNOWLEDGE; });
  const view = describeSpan(structured, model);
  assert.strictEqual(block(view, 'Answer').kind, BLOCK.TEXT);
  assert.strictEqual(block(view, 'Schema').kind, BLOCK.JSON);
  const tooled = describeSpan(spanOf(model, KIND.MODEL, function (s) { return s.data.asked.source === LANE.RESOLVER; }), model);
  assert.strictEqual(block(tooled, 'Tools offered').kind, BLOCK.LIST);
});

test('a model call lists the tool calls made during it', function () {
  const model = modelOf([
    { event: EVENT.MODEL_ASKED, at: 1, call: 'm', source: LANE.RESOLVER, style: 'with_tools', identity: 'i', system: '', user: '', tools: ['x'] },
    { event: EVENT.TOOL_CALLED, at: 2, call: 't', parent: 'm', tool: 'x', arguments: { a: 1 }, owner: LANE.WORLD },
    { event: EVENT.TOOL_RETURNED, at: 3, call: 't', content: '{"ok": true}', is_error: false, seconds: 1 },
    { event: EVENT.MODEL_ANSWERED, at: 4, call: 'm', answer: 'a', seconds: 3 },
  ]);
  const view = describeSpan(spanOf(model, KIND.MODEL), model);
  const calls = block(view, 'Tool calls');
  assert.deepStrictEqual(calls.items.map(function (i) { return i.label; }), ['x']);
  assert.strictEqual(calls.items[0].id, 'tool:t');
});

test('a tool call shows its input and what it returned', function () {
  const model = modelOf([
    { event: EVENT.TOOL_CALLED, at: 2, call: 't', parent: 'm', tool: 'x', arguments: { a: 1 }, owner: LANE.WORLD },
    { event: EVENT.TOOL_RETURNED, at: 3, call: 't', content: '{"ok": true}', is_error: false, seconds: 1 },
  ]);
  const view = describeSpan(spanOf(model, KIND.TOOL), model);
  assert.strictEqual(block(view, 'Input').kind, BLOCK.JSON);
  assert.strictEqual(block(view, 'Returned').kind, BLOCK.JSON);
  assert.strictEqual(fact(view, 'belongs to'), LANE.WORLD);
});

test('a call between components shows what it asked and what it was told', function () {
  const model = modelOf(resolved);
  const view = describeSpan(spanOf(model, KIND.CALL, function (s) { return s.label === 'serve'; }), model);
  assert.strictEqual(fact(view, 'from'), LANE.RESOLVER);
  assert.strictEqual(fact(view, 'to'), LANE.KNOWLEDGE);
  assert.strictEqual(block(view, 'Asked').kind, BLOCK.JSON);
  assert.strictEqual(block(view, 'Answered').kind, BLOCK.TEXT);
});

test('a resolution shows its slots, what the knowledge served and how it ended', function () {
  const model = modelOf(resolved);
  const action = actionResolution(model, resolved);
  const view = describeSpan(action, model);
  const slots = block(view, 'Slots');
  assert.strictEqual(slots.kind, BLOCK.TABLE);
  assert.ok(slots.rows.length >= 1);
  assert.strictEqual(block(view, 'Knowledge consulted').kind, BLOCK.FACTS);
  assert.strictEqual(block(view, 'Attempts').items.length, 2);
  assert.deepStrictEqual(block(view, 'Attempts').items.map(function (i) { return i.status; }), [STATUS.REJECTED, STATUS.OK]);
});

test('an attempt rejected shows why, what the failure found and the answer', function () {
  const model = modelOf(resolved);
  const rejected = spanOf(model, KIND.ATTEMPT, function (s) { return s.status === STATUS.REJECTED; });
  const view = describeSpan(rejected, model);
  const rejection = resolved.find(function (e) { return e.event === EVENT.ATTEMPT_REJECTED; });
  assert.strictEqual(fact(view, 'rejected for'), rejection.kind);
  assert.strictEqual(block(view, 'What the failure found').kind, BLOCK.JSON);
  assert.strictEqual(block(view, 'Answer rejected').kind, BLOCK.TEXT);
});

test('a hypothesis shows what was asked for, its attempts and how it was served', function () {
  const model = modelOf(resolved);
  const view = describeSpan(spanOf(model, KIND.HYPOTHESIS), model);
  assert.ok(fact(view, 'kind'));
  assert.strictEqual(fact(view, 'from the store'), 'no');
  assert.ok(fact(view, 'record'));
  assert.strictEqual(block(view, 'Fixed values').kind, BLOCK.JSON);
});

test('a pour tried in the simulation says where it ended', function () {
  const model = modelOf(resolved);
  const trial = spanOf(model, KIND.TRIAL);
  const view = describeSpan(trial, model);
  assert.strictEqual(block(view, 'Values tried').kind, BLOCK.JSON);
  assert.strictEqual(block(view, 'Outcome').kind, BLOCK.JSON);
  assert.ok(fact(view, 'result'));
});

test('a pour is written as landed or missed with where the mouth ended', function () {
  const where = {
    [OUTCOME.LANDS_IN_OPENING]: true, [OUTCOME.DISTANCE]: 0.005, [OUTCOME.AHEAD]: 0.001,
    [OUTCOME.RIGHT]: 0.004, [OUTCOME.ABOVE]: 0.09,
  };
  assert.strictEqual(trialText(where), 'landed: the mouth 0.005 m from the opening\'s centre (0.001 m ahead, 0.004 m right), 0.09 m above it');
  assert.ok(trialText(Object.assign({}, where, { [OUTCOME.LANDS_IN_OPENING]: false })).startsWith('missed'));
  assert.strictEqual(trialText({ [OUTCOME.LANDS_IN_OPENING]: false, [OUTCOME.MOTION_FAILED]: 'out of reach' }), 'the motion failed: out of reach');
  assert.strictEqual(trialText({ [OUTCOME.LANDS_IN_OPENING]: false }), 'missed');
});

test('a run shows its query and scene', function () {
  const model = modelOf(resolved);
  const view = describeSpan(spanOf(model, KIND.RUN), model);
  const started = resolved.find(function (e) { return e.event === EVENT.RUN_STARTED; });
  assert.strictEqual(fact(view, 'scene'), started.scene);
  assert.strictEqual(block(view, 'Query').kind, BLOCK.TEXT);
});

test('a span of a kind nobody described is shown as its events', function () {
  const model = modelOf(resolved);
  const odd = { id: 'odd', kind: 'odd', lane: LANE.RUN, label: 'odd', start: 1, end: 2, status: STATUS.OK, parent: null, events: [{ event: 'X', at: 1 }], data: {} };
  const view = describeSpan(odd, model);
  assert.strictEqual(block(view, 'Events').kind, BLOCK.JSON);
});

test('what a resolution rests on is named by its fragment, not by the address it is found at', function () {
  const model = modelOf(resolved);
  const action = actionResolution(model, resolved);
  const why = block(describeSpan(action, model), 'Why');
  assert.strictEqual(why.items[0].note, 'knowledge: pour');
});

test('what happened during an attempt is listed with the kind of work it was', function () {
  const model = modelOf(resolved);
  const rejected = spanOf(model, KIND.ATTEMPT, function (s) { return s.status === STATUS.REJECTED; });
  const during = block(describeSpan(rejected, model), 'During the attempt');
  assert.deepStrictEqual(during.items.map(function (item) { return item.label; }),
    ['model call · resolver', 'call · citation', 'trial 1']);
});

// %% where pours ended

const POURS = [{ ahead: 0.02, right: 0.18 }, { ahead: 0.005, right: 0.01 }];

function modelOfPours() {
  return modelOf(pourEvents(POURS.map(function (where) { return pourOutcome(Object.assign({ radius: 0.04, above: 0.09 }, where)); })));
}

test('a pour shows where the mouth ended on a plan of the opening, with the pours before it', function () {
  const model = modelOfPours();
  const [first, second] = spansOfKind(model, KIND.TRIAL);
  const view = describeSpan(second, model);
  const plan = block(view, 'Where the mouth ended');
  assert.strictEqual(plan.kind, BLOCK.PLOT);
  assert.deepStrictEqual(plan.plot.points.map(function (point) { return point.id; }), [first.id, second.id]);
  assert.strictEqual(plan.emphasised, second.id);
});

test('a pour with no place has no plan', function () {
  const model = modelOf(pourEvents([{ [OUTCOME.LANDS_IN_OPENING]: false, [OUTCOME.MOTION_FAILED]: 'out of reach' }]));
  assert.strictEqual(block(describeSpan(spansOfKind(model, KIND.TRIAL)[0], model), 'Where the mouth ended'), undefined);
});

test('a resolution shows the plan of the pours tried during its attempts', function () {
  const events = [
    { event: EVENT.RESOLUTION_STARTED, at: 0, resolution: 'r', purpose: 'action', action: 'a.B', slots: [], max_repairs: 1 },
    { event: EVENT.ATTEMPT_STARTED, at: 0.5, resolution: 'r', attempt: 0 },
    ...pourEvents(POURS.map(function (where) { return pourOutcome(Object.assign({ radius: 0.04, above: 0.09 }, where)); })),
    { event: EVENT.RESOLUTION_FINISHED, at: 30, resolution: 'r', attempts: 1, instances: [], justifications: [] },
  ];
  const model = modelOf(events);
  const plan = block(describeSpan(model.byId.get('resolution:r'), model), 'Pours tried');
  assert.strictEqual(plan.kind, BLOCK.PLOT);
  assert.strictEqual(plan.plot.points.length, 2);
  assert.strictEqual(plan.emphasised, null);
});

test('a resolution that tried no pour has no plan of them', function () {
  const model = modelOf(resolved);
  assert.strictEqual(block(describeSpan(actionResolution(model, resolved), model), 'Pours tried'), undefined);
});

// %% what a model call read

test('a model call shows all the tokens it read, and how many came from the cache and went to it', function () {
  const usage = { [USAGE.INPUT]: 6, [USAGE.OUTPUT]: 389, [USAGE.CACHE_WRITTEN]: 13572, [USAGE.CACHE_READ]: 7745 };
  const model = modelOf([
    { event: EVENT.MODEL_ASKED, at: 1, call: 'm', source: LANE.RESOLVER, style: 'plain', identity: 'i', system: '', user: '', tools: [] },
    { event: EVENT.MODEL_ANSWERED, at: 2, call: 'm', answer: 'a', seconds: 1, usage: usage, cost: 0.04 },
  ]);
  const view = describeSpan(spanOf(model, KIND.MODEL), model);
  assert.strictEqual(fact(view, 'input tokens'), usage[USAGE.INPUT] + usage[USAGE.CACHE_WRITTEN] + usage[USAGE.CACHE_READ]);
  assert.strictEqual(fact(view, 'read from the cache'), usage[USAGE.CACHE_READ]);
  assert.strictEqual(fact(view, 'written to the cache'), usage[USAGE.CACHE_WRITTEN]);
  assert.strictEqual(fact(view, 'output tokens'), usage[USAGE.OUTPUT]);
});

test('a model call that used no cache says nothing of it', function () {
  const model = modelOf([
    { event: EVENT.MODEL_ASKED, at: 1, call: 'm', source: LANE.RESOLVER, style: 'plain', identity: 'i', system: '', user: '', tools: [] },
    { event: EVENT.MODEL_ANSWERED, at: 2, call: 'm', answer: 'a', seconds: 1, usage: { [USAGE.INPUT]: 100, [USAGE.OUTPUT]: 20 }, cost: 0.01 },
  ]);
  const view = describeSpan(spanOf(model, KIND.MODEL), model);
  assert.strictEqual(fact(view, 'input tokens'), 100);
  assert.strictEqual(fact(view, 'read from the cache'), undefined);
  assert.strictEqual(fact(view, 'written to the cache'), undefined);
});

// %% the NEEM kept of what the simulation performed

const neemRun = loadFixture('neem_run.json');

function trialModel(kept) {
  return modelOf([
    { event: EVENT.TRIAL_STARTED, at: 1, trial: 't1', values: {} },
    kept,
    { event: EVENT.TRIAL_FINISHED, at: 5, trial: 't1', outcome: { [OUTCOME.LANDS_IN_OPENING]: true }, seconds: 4 },
  ]);
}

const keptOfTrial = {
  event: EVENT.NEEM_RECORDED, at: 3, kind: NEEM_KIND.TRIAL, episode: 't1', database: '/data/neems.sqlite', database_id: 7, seconds: 5.5,
};

test('a pour whose NEEM was kept says so, and where', function () {
  const model = trialModel(keptOfTrial);
  const view = describeSpan(spanOf(model, KIND.TRIAL), model);
  assert.ok(fact(view, 'NEEM').includes(String(keptOfTrial.database_id)));
  const neem = block(view, 'NEEM');
  assert.strictEqual(neem.kind, BLOCK.FACTS);
  assert.deepStrictEqual(neem.facts.find(function (row) { return row[0] === 'database'; })[1], keptOfTrial.database);
  assert.deepStrictEqual(neem.facts.find(function (row) { return row[0] === 'plan'; })[1], keptOfTrial.database_id);
});

test('a NEEM that was kept comes with a command that reads it', function () {
  const model = trialModel(keptOfTrial);
  const read = block(describeSpan(spanOf(model, KIND.TRIAL), model), 'Read it');
  assert.strictEqual(read.kind, BLOCK.TEXT);
  assert.ok(read.text.includes(keptOfTrial.database));
  assert.ok(read.text.includes('database_id = ' + keptOfTrial.database_id));
});

test('the NEEM kept of an execution is shown the same way', function () {
  const model = modelOf(neemRun);
  const kept = neemRun.find(function (event) { return event.event === EVENT.NEEM_RECORDED; });
  const view = describeSpan(spanOf(model, KIND.EXECUTION), model);
  assert.ok(fact(view, 'NEEM').includes(String(kept.database_id)));
  assert.strictEqual(block(view, 'NEEM').kind, BLOCK.FACTS);
});

test('a NEEM that could not be kept says why', function () {
  const failure = { event: EVENT.NEEM_NOT_RECORDED, at: 3, kind: NEEM_KIND.TRIAL, episode: 't1', error_type: 'DiskIsFull', reason: 'no space left on device' };
  const model = trialModel(failure);
  const view = describeSpan(spanOf(model, KIND.TRIAL), model);
  assert.ok(fact(view, 'NEEM').includes(failure.error_type));
  assert.strictEqual(block(view, 'NEEM not saved').text, failure.reason);
  assert.strictEqual(block(view, 'Read it'), undefined);
});

test('a pour that kept no NEEM says nothing of one', function () {
  const model = modelOfPours();
  const view = describeSpan(spansOfKind(model, KIND.TRIAL)[0], model);
  assert.strictEqual(fact(view, 'NEEM'), undefined);
  assert.strictEqual(block(view, 'NEEM'), undefined);
});
