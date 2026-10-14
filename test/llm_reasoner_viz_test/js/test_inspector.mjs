// Tests of what the inspector shows of a span: its facts and the blocks of detail.
import test from 'node:test';
import assert from 'node:assert';
import { KIND, LANE, STATUS, EVENT } from '../../../llm_reasoner_viz/src/llm_reasoner_viz/web/model.mjs';
import { BLOCK, describeSpan, trialText } from '../../../llm_reasoner_viz/src/llm_reasoner_viz/web/inspector.mjs';
import { actionResolution, loadFixture, modelOf } from './support.mjs';

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
  assert.strictEqual(trialText({ lands_in_opening: true, mouth_distance_from_opening_centre: 0.005,
    mouth_ahead_of_opening_centre: 0.001, mouth_right_of_opening_centre: 0.004, mouth_above_opening: 0.09 }),
  'landed: the mouth 0.005 m from the opening\'s centre (0.001 m ahead, 0.004 m right), 0.09 m above it');
  assert.ok(trialText({ lands_in_opening: false, mouth_distance_from_opening_centre: 0.188,
    mouth_ahead_of_opening_centre: 0.002, mouth_right_of_opening_centre: 0.188, mouth_above_opening: 0.087 }).startsWith('missed'));
  assert.strictEqual(trialText({ lands_in_opening: false, motion_failed: 'out of reach' }), 'the motion failed: out of reach');
  assert.strictEqual(trialText({ lands_in_opening: false }), 'missed');
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
