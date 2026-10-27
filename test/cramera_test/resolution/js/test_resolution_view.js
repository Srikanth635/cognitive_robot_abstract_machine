// Tests what a resolution, its trials and the runs kept read as: the markup of the answer
// area, and the text of a designator's hint and of a run in the list.
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const WEB_ADDON = path.join(
  __dirname, '..', '..', '..', '..', 'cramera', 'src', 'cramera', 'web', 'panels', 'resolution',
);

function loadView() {
  global.window = {};
  new Function(fs.readFileSync(path.join(WEB_ADDON, 'resolution_view.js'), 'utf8'))();
  return global.window.ResolutionView;
}

const view = loadView();

function missedOutcome() {
  return {
    lands_in_opening: false, mouth_distance_from_opening_centre: 0.188,
    mouth_ahead_of_opening_centre: 0.002, mouth_right_of_opening_centre: 0.188,
    mouth_above_opening: 0.087,
  };
}

function resolution(overrides) {
  return Object.assign({
    query: 'the(PouringAction)(\n    arm=variable(Arm),\n)\n',
    designator: 'PouringAction',
    actions: [{
      slots: [
        { name: 'target_container', value: 'bowl.stl', given: true },
        { name: 'tilt_angle', value: '1.85', given: false },
      ],
      why: 'The default tilt lets the milk flow.',
      grounds: [{ source: 'knowledge', name: 'ph_tilt' }],
    }],
    situation: { has_material_regime: 'LIQUID' },
    record_id: 'abc', perspective_record_ids: [], cost: 0.0512, run: 'run-1', trials: [],
  }, overrides || {});
}

// %% escaping
test('markup characters of a value are escaped', function () {
  assert.strictEqual(view.esc('a<b>&c'), 'a&lt;b&gt;&amp;c');
});

test('an attribute value also has its quotes escaped', function () {
  assert.strictEqual(view.attributeText('x"y'), 'x&quot;y');
});

// %% trials
test('a pour that landed says where the mouth was against the opening', function () {
  const outcome = Object.assign(missedOutcome(), { lands_in_opening: true });
  assert.ok(view.trialText(outcome).startsWith('landed: the mouth 0.188 m from'));
});

test('a pour that missed says so with the offsets in each direction', function () {
  assert.strictEqual(
    view.trialText(missedOutcome()),
    "missed: the mouth 0.188 m from the opening's centre (0.002 m ahead, 0.188 m right), 0.087 m above it",
  );
});

test('a pour whose motion failed says why instead of where it ended', function () {
  assert.strictEqual(
    view.trialText({ lands_in_opening: false, motion_failed: 'MotionMadeNoProgress: out of reach' }),
    'the motion failed: MotionMadeNoProgress: out of reach',
  );
});

test('the trials are numbered from one in the order they were tried', function () {
  const html = view.trialsHtml([{ outcome: missedOutcome() }, { outcome: missedOutcome() }]);
  assert.ok(html.indexOf('<span class="tag">1</span>') < html.indexOf('<span class="tag">2</span>'));
});

test('no trials write nothing', function () {
  assert.strictEqual(view.trialsHtml([]), '');
});

// %% the resolution
test('a given slot is tagged given and a chosen one chosen', function () {
  const html = view.resolutionHtml(resolution(), null);
  assert.ok(html.includes('<span class="tag">given</span><div class="body"><span class="name">target_container = bowl.stl'));
  assert.ok(html.includes('<span class="tag">chosen</span><div class="body"><span class="name">tilt_angle = 1.85'));
});

test('the answer says why and what it rests on', function () {
  const html = view.resolutionHtml(resolution(), null);
  assert.ok(html.includes('why: The default tilt lets the milk flow.'));
  assert.ok(html.includes('rests on knowledge: ph_tilt'));
});

test('the answer names the situation, the hypothesis, the cost and the run', function () {
  const html = view.resolutionHtml(resolution(), null);
  assert.ok(html.includes('knowledge consulted for has_material_regime = LIQUID (hypothesis abc)'));
  assert.ok(html.includes('resolver cost $0.051'));
  assert.ok(html.includes('run run-1'));
});

test('a situation nothing is known of is said to be unknown', function () {
  const html = view.resolutionHtml(resolution({ situation: {} }), null);
  assert.ok(html.includes('knowledge consulted for an unknown situation'));
});

test('the perspectives consulted are named beside the action core', function () {
  const html = view.resolutionHtml(resolution({ perspective_record_ids: ['p1', 'p2'] }), null);
  assert.ok(html.includes('with the action core and perspectives p1, p2'));
});

test('the viewer of the run is linked when the answer has one', function () {
  const html = view.resolutionHtml(resolution(), 'http://localhost:8721/?run=run-1');
  assert.ok(html.includes('<a href="http://localhost:8721/?run=run-1" target="_blank" rel="noopener">'));
});

test('no link is written when the answer has no viewer', function () {
  assert.ok(!view.resolutionHtml(resolution(), null).includes('<a href'));
});

test('a value that holds markup cannot inject any', function () {
  const hostile = resolution();
  hostile.actions[0].slots[0].value = '<img src=x onerror=alert(1)>';
  assert.ok(!view.resolutionHtml(hostile, null).includes('<img'));
});

test('the trials the resolution tried are part of the answer', function () {
  const html = view.resolutionHtml(resolution({ trials: [{ outcome: missedOutcome() }] }), null);
  assert.ok(html.includes('tried in the simulation'));
});

// %% designators and runs
test('an available designator is told how to use', function () {
  const hint = view.designatorHint({ identifier: 'objects_given', available: true, unbound: [], without_candidates: [] });
  assert.strictEqual(hint, 'objects_given: click to copy it into the search bar, then Resolve');
});

test('an unavailable designator says what the scene lacks', function () {
  const hint = view.designatorHint({
    identifier: 'objects_given', available: false,
    unbound: ['pouring_cup'], without_candidates: ['source_container'],
  });
  assert.strictEqual(
    hint,
    'objects_given needs an object named pouring_cup, a candidate for source_container, which this scene lacks;' +
      ' click to copy it into the box anyway, but Resolve will refuse it here',
  );
});

test('a run is read as its scene, designator and query', function () {
  const run = { name: 'r', scene: 'simple_pouring', designator: 'PouringAction', query: 'the(PouringAction)()' };
  assert.strictEqual(view.runText(run), 'simple_pouring · PouringAction · the(PouringAction)()');
});

test('a run says when it was kept, what it cost and how many pours it tried', function () {
  const run = { name: 'r', at: '2026-10-21T10:00:00', cost: 0.1368, trials: 2 };
  assert.strictEqual(
    view.runTitle(run),
    'run r · 2026-10-21T10:00:00 · resolver cost $0.137 · 2 pours tried — click to show it',
  );
});

test('a single pour is not written in the plural', function () {
  const run = { name: 'r', at: 'x', cost: 0, trials: 1 };
  assert.ok(view.runTitle(run).includes('1 pour tried'));
});
