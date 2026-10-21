// Tests the designator panel: chips reading the scene's designators as krrood match queries
// and copying them into the query box; the choices of knowledge to consult; Resolve
// resolving the query in the box in the scene with llm_reasoner and showing the answer;
// the runs kept earlier, listed and shown again; and Execute performing the run shown.
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const WEB_ADDON = path.join(
  __dirname, '..', '..', '..', 'cramera_resolution', 'src', 'cramera_resolution', 'web', 'addon',
);

const PANEL_ID = 'resolution-designators';

const ROUTES = {
  designators: '/api/resolution/designators?scene=simple_pouring',
  runs: '/api/resolution/runs',
  run: '/api/resolution/run?name=run-1&scene=simple_pouring',
  resolve: '/api/resolution/resolve',
  execute: '/api/resolution/execute',
};

// %% a page to mount the panel into
function element() {
  const created = {
    innerHTML: '', textContent: '', value: '', disabled: false, checked: undefined, type: '',
    style: {}, className: '', title: '', children: [], listeners: {}, classes: new Set(),
    classList: {
      toggle(name, force) {
        const on = force === undefined ? !this.owner.classes.has(name) : force;
        if (on) this.owner.classes.add(name); else this.owner.classes.delete(name);
      },
    },
    addEventListener(event, listener) { this.listeners[event] = listener; },
    appendChild(child) { this.children.push(child); },
    scrollTop: 0,
  };
  created.classList.owner = created;
  return created;
}

function designatorsPayload() {
  return {
    ok: true,
    scene: 'simple_pouring',
    designators: [
      {
        identifier: 'objects_given',
        query: 'the(PouringAction)(\n    source_container=pouring_cup,\n    arm=variable(Arm),\n)\n',
        query_on_one_line: 'the(PouringAction)( source_container=pouring_cup, arm=variable(Arm), )',
        available: true, unbound: [], without_candidates: [],
      },
      {
        identifier: 'needs_the_cup',
        query: 'the(PouringAction)(\n    source_container=pouring_cup,\n)\n',
        query_on_one_line: 'the(PouringAction)( source_container=pouring_cup, )',
        available: false, unbound: ['pouring_cup'], without_candidates: [],
      },
      {
        identifier: 'objects_open',
        query: 'the(PouringAction)(\n    source_container=variable(Tool),\n)\n',
        query_on_one_line: 'the(PouringAction)( source_container=variable(Tool), )',
        available: false, unbound: [], without_candidates: ['source_container'],
      },
    ],
    phase_structure: { label: 'phase structure', summary: 'How a kind of manipulation task is organised.' },
    geometry_adjustment: { label: 'adjust the pour geometry', summary: 'A miss may be corrected with the pour geometry.' },
    perspectives: [
      { name: 'action-frame.ttl', label: 'action frame', summary: 'The action frame describes an action.' },
      { name: 'failures.ttl', label: 'failures', summary: 'The failures describe how an action goes wrong.' },
    ],
  };
}

function resolvedPayload() {
  return {
    ok: true,
    viewer: 'http://localhost:8721/?run=run-1',
    resolution: {
      query: 'the(PouringAction)(\n    source_container=pouring_cup,\n)\n',
      designator: 'PouringAction',
      actions: [{
        slots: [
          { name: 'target_container', value: 'bowl.stl', given: true },
          { name: 'tilt_angle', value: '1.85', given: false },
        ],
        why: 'The default tilt lets the milk flow.',
        grounds: [{ source: 'knowledge', name: 'ph_tilt' }],
      }],
      situation: { has_material_regime: 'LIQUID' }, record_id: 'abc', cost: 0.05, run: 'run-1',
      trials: [
        { outcome: { lands_in_opening: false, mouth_distance_from_opening_centre: 0.188, mouth_ahead_of_opening_centre: 0.002, mouth_right_of_opening_centre: 0.188, mouth_above_opening: 0.087 } },
        { outcome: { lands_in_opening: true, mouth_distance_from_opening_centre: 0.005, mouth_ahead_of_opening_centre: 0.003, mouth_right_of_opening_centre: 0.004, mouth_above_opening: 0.089 } },
      ],
    },
    hypothesis: { ok: true, crumb: 'hypothesis graph', root: 'g0', nodes: [], edges: [], details: {} },
  };
}

function runsPayload() {
  return {
    ok: true,
    runs: [
      { name: 'run-2', scene: 'simple_pouring', designator: 'PouringAction', query: 'the(PouringAction)( arm=variable(Arm), )', cost: 0.12, trials: 2, at: '2026-10-21T11:00:00' },
      { name: 'run-1', scene: 'simple_pouring', designator: 'PouringAction', query: 'the(PouringAction)( )', cost: 0.05, trials: 1, at: '2026-10-21T10:00:00' },
    ],
  };
}

function loadPanel(options) {
  const settings = options || {};
  const elements = {};
  [
    '#resolution-answer', '#resolution-query', '#resolution-resolve', '#resolution-execute',
    '#resolution-designators', '#resolution-runs', '#resolution-phase-structure',
    '#resolution-perspectives', '#resolution-adjustment', '#resolution-tab-designators',
    '#resolution-tab-runs',
  ].forEach(function (name) { elements[name] = element(); });

  const root = { innerHTML: '', querySelector(selector) { return elements[selector]; } };
  const handlers = {};
  const emitted = [];
  const bus = {
    on(event, listener) { (handlers[event] = handlers[event] || []).push(listener); },
    emit(event, payload) { emitted.push({ event: event, payload: payload }); },
  };

  const requests = [];
  function answer(payload) { return { status: 200, json: async function () { return payload; } }; }
  global.fetch = async function (url, init) {
    requests.push({ url: url, init: init });
    if (url === ROUTES.designators) return answer(settings.designators || designatorsPayload());
    if (url === ROUTES.runs) return answer(settings.runs || runsPayload());
    if (url === ROUTES.resolve) {
      return answer(settings.resolutionFails
        ? { ok: false, error: 'NoFittingReferent: nothing fits' }
        : (settings.resolved || resolvedPayload()));
    }
    if (url === ROUTES.run) {
      return answer(settings.openFails
        ? { ok: false, error: 'UnknownRun: no run of that name is kept' }
        : resolvedPayload());
    }
    if (url === ROUTES.execute) {
      return answer(settings.executeFails
        ? { ok: false, error: 'UnknownRun: no run of that name is kept' }
        : { ok: true });
    }
    throw new Error('Unexpected fetch: ' + url);
  };

  global.document = { createElement() { return element(); } };
  global.SceneContext = {
    name() { return 'simple_pouring'; },
    withScene(url) { return url + (url.indexOf('?') >= 0 ? '&' : '?') + 'scene=simple_pouring'; },
  };
  const layout = settings.layout || { left: ['robot-scene'], right: ['eql', 'graph'] };
  global.window = { location: { search: settings.search || '' }, CRAMERA_CONFIG: { layout: layout } };

  const defined = {};
  global.Panels = { define(identifier, factory) { defined[identifier] = factory; } };
  ['resolution_view.js', 'designator_panel.js'].forEach(function (file) {
    new Function(fs.readFileSync(path.join(WEB_ADDON, file), 'utf8'))();
  });
  return { root, bus, elements, requests, emitted, defined, layout, mount() { return defined[PANEL_ID](root, bus); } };
}

async function settle() {
  for (let i = 0; i < 8; i += 1) {
    await new Promise(function (resolve) { setImmediate(resolve); });
  }
}

async function mounted(options) {
  const panel = loadPanel(options);
  panel.mount();
  await settle();
  return panel;
}

function post(panel, route) {
  return panel.requests.filter(function (r) { return r.url === route && r.init; });
}

function bodyOf(request) { return JSON.parse(request.init.body); }

function chips(panel) { return panel.elements['#resolution-designators'].children; }
function runEntries(panel) { return panel.elements['#resolution-runs'].children; }

function resolve(panel, query) {
  panel.elements['#resolution-query'].value = query;
  panel.elements['#resolution-resolve'].listeners.click();
}

// %% the panel in the page
test('the panel is defined under its identifier and added below the shipped panels', function () {
  const panel = loadPanel();
  assert.deepStrictEqual(Object.keys(panel.defined), [PANEL_ID]);
  assert.deepStrictEqual(panel.layout.right, ['eql', 'graph', PANEL_ID]);
});

test('a layout with no right slot is left as it is', function () {
  const panel = loadPanel({ layout: { left: ['robot-scene'] } });
  assert.deepStrictEqual(panel.layout, { left: ['robot-scene'] });
});

test('every id the panel reads is in its markup, none of them an id cramera already uses', async function () {
  const panel = await mounted();
  const html = panel.root.innerHTML;
  Object.keys(panel.elements).forEach(function (selector) {
    assert.ok(html.includes('id="' + selector.slice(1) + '"'), selector);
  });
  ['query-input', 'query-run', 'presets', 'answer'].forEach(function (shipped) {
    assert.ok(!html.includes('id="' + shipped + '"'), shipped);
  });
});

// %% the designators
test('each designator of the scene becomes a chip reading its match query on one line', async function () {
  const panel = await mounted();
  assert.deepStrictEqual(
    chips(panel).map(function (chip) { return chip.textContent; }),
    designatorsPayload().designators.map(function (d) { return d.query_on_one_line; }),
  );
});

test('the designators are asked for in the scene the page shows', async function () {
  const panel = await mounted();
  assert.ok(panel.requests.some(function (r) { return r.url === ROUTES.designators; }));
});

test('a chip shows its whole match query on hover', async function () {
  const panel = await mounted();
  assert.ok(chips(panel)[0].title.includes(designatorsPayload().designators[0].query));
});

test('a designator naming an object the scene lacks says so and resolves nothing', async function () {
  const panel = await mounted();
  const unavailable = chips(panel)[1];
  assert.ok(unavailable.className.includes('unavailable'));
  assert.ok(unavailable.title.includes(designatorsPayload().designators[1].unbound[0]));
  assert.strictEqual(unavailable.listeners.click, undefined);
});

test('a designator with nothing in the scene to take for a slot says so', async function () {
  const panel = await mounted();
  const unavailable = chips(panel)[2];
  assert.ok(unavailable.className.includes('unavailable'));
  assert.ok(unavailable.title.includes(designatorsPayload().designators[2].without_candidates[0]));
});

test('clicking an available designator copies its query into the box and resolves nothing', async function () {
  const panel = await mounted();
  chips(panel)[0].listeners.click();
  await settle();
  assert.strictEqual(panel.elements['#resolution-query'].value, designatorsPayload().designators[0].query);
  assert.deepStrictEqual(post(panel, ROUTES.resolve), []);
});

test('designators that cannot be loaded say why instead of showing none', async function () {
  const panel = await mounted({ designators: { ok: false, error: 'NoSuchScene: no scene bundle' } });
  assert.ok(panel.elements['#resolution-designators'].textContent.includes('NoSuchScene'));
});

// %% the two lists
test('the designators show first and the runs are a tab away', async function () {
  const panel = await mounted();
  assert.ok(!panel.elements['#resolution-designators'].classes.has('hidden'));
  assert.ok(panel.elements['#resolution-runs'].classes.has('hidden'));
  assert.ok(panel.elements['#resolution-tab-designators'].classes.has('on'));
});

test('the runs tab shows the runs in place of the designators', async function () {
  const panel = await mounted();
  panel.elements['#resolution-tab-runs'].listeners.click();
  assert.ok(panel.elements['#resolution-designators'].classes.has('hidden'));
  assert.ok(!panel.elements['#resolution-runs'].classes.has('hidden'));
  assert.ok(panel.elements['#resolution-tab-runs'].classes.has('on'));
  assert.ok(!panel.elements['#resolution-tab-designators'].classes.has('on'));
});

// %% what Resolve consults
test('the phase structure, the perspectives and the pour geometry are offered ticked', async function () {
  const panel = await mounted();
  const boxes = ['#resolution-phase-structure', '#resolution-adjustment']
    .map(function (id) { return panel.elements[id].children[0].children[0]; })
    .concat(panel.elements['#resolution-perspectives'].children.map(function (c) { return c.children[0]; }));
  assert.strictEqual(boxes.length, 4);
  assert.ok(boxes.every(function (box) { return box.checked === true; }));
});

test('each perspective is offered by the file the knowledge registers it by', async function () {
  const panel = await mounted();
  assert.deepStrictEqual(
    panel.elements['#resolution-perspectives'].children.map(function (c) { return c.children[0].value; }),
    designatorsPayload().perspectives.map(function (p) { return p.name; }),
  );
});

test('a request names the scene, the query and the perspectives ticked', async function () {
  const panel = await mounted();
  resolve(panel, 'the(PouringAction)()');
  await settle();
  assert.deepStrictEqual(bodyOf(post(panel, ROUTES.resolve)[0]), {
    query: 'the(PouringAction)()', scene: 'simple_pouring',
    perspectives: ['action-frame.ttl', 'failures.ttl'],
  });
});

test('a perspective unticked is left out of the request', async function () {
  const panel = await mounted();
  panel.elements['#resolution-perspectives'].children[0].children[0].checked = false;
  resolve(panel, 'the(PouringAction)()');
  await settle();
  assert.deepStrictEqual(bodyOf(post(panel, ROUTES.resolve)[0]).perspectives, ['failures.ttl']);
});

test('every perspective unticked leaves the request with none', async function () {
  const panel = await mounted();
  panel.elements['#resolution-perspectives'].children.forEach(function (c) { c.children[0].checked = false; });
  resolve(panel, 'the(PouringAction)()');
  await settle();
  assert.strictEqual('perspectives' in bodyOf(post(panel, ROUTES.resolve)[0]), false);
});

test('the phase structure unticked is left out of the request', async function () {
  const panel = await mounted();
  panel.elements['#resolution-phase-structure'].children[0].children[0].checked = false;
  resolve(panel, 'the(PouringAction)()');
  await settle();
  assert.strictEqual(bodyOf(post(panel, ROUTES.resolve)[0]).phase_structure, false);
});

test('the pour geometry unticked is kept as the query has it', async function () {
  const panel = await mounted();
  panel.elements['#resolution-adjustment'].children[0].children[0].checked = false;
  resolve(panel, 'the(PouringAction)()');
  await settle();
  assert.strictEqual(bodyOf(post(panel, ROUTES.resolve)[0]).adjust_geometry, false);
});

test('the choices left as they are add nothing to the request', async function () {
  const panel = await mounted();
  resolve(panel, 'the(PouringAction)()');
  await settle();
  const body = bodyOf(post(panel, ROUTES.resolve)[0]);
  assert.strictEqual('phase_structure' in body || 'adjust_geometry' in body, false);
});

// %% resolving
test('an empty query resolves nothing', async function () {
  const panel = await mounted();
  resolve(panel, '   ');
  await settle();
  assert.deepStrictEqual(post(panel, ROUTES.resolve), []);
});

test('Execute waits for a resolution', async function () {
  const panel = await mounted();
  assert.strictEqual(panel.elements['#resolution-execute'].disabled, true);
});

test('while it resolves the answer area says so and Resolve cannot be asked again', async function () {
  const panel = await mounted();
  resolve(panel, 'the(PouringAction)()');
  assert.ok(panel.elements['#resolution-answer'].innerHTML.includes('resolving'));
  assert.strictEqual(panel.elements['#resolution-resolve'].textContent, '…');
  panel.elements['#resolution-resolve'].listeners.click();
  await settle();
  assert.strictEqual(post(panel, ROUTES.resolve).length, 1);
});

test('the answer shows what was chosen, why, the trials and the link to the run', async function () {
  const panel = await mounted();
  resolve(panel, 'the(PouringAction)()');
  await settle();
  const html = panel.elements['#resolution-answer'].innerHTML;
  assert.ok(html.includes('tilt_angle = 1.85'));
  assert.ok(html.includes('why: The default tilt lets the milk flow.'));
  assert.ok(html.includes('landed:'));
  assert.ok(html.includes('http://localhost:8721/?run=run-1'));
});

test('the hypothesis the resolution consulted is published on the bus', async function () {
  const panel = await mounted();
  resolve(panel, 'the(PouringAction)()');
  await settle();
  const published = panel.emitted.filter(function (e) { return e.event === 'hypothesis:generated'; });
  assert.deepStrictEqual(published.map(function (e) { return e.payload; }), [resolvedPayload().hypothesis]);
});

test('after a resolution Resolve is ready again and Execute is enabled', async function () {
  const panel = await mounted();
  resolve(panel, 'the(PouringAction)()');
  await settle();
  assert.strictEqual(panel.elements['#resolution-resolve'].textContent, 'Resolve');
  assert.strictEqual(panel.elements['#resolution-execute'].disabled, false);
});

test('a resolution that fails says why and leaves Execute disabled', async function () {
  const panel = await mounted({ resolutionFails: true });
  resolve(panel, 'the(PouringAction)()');
  await settle();
  assert.ok(panel.elements['#resolution-answer'].innerHTML.includes('NoFittingReferent: nothing fits'));
  assert.strictEqual(panel.elements['#resolution-execute'].disabled, true);
});

test('the runs are listed again once a resolution has made another', async function () {
  const panel = await mounted();
  resolve(panel, 'the(PouringAction)()');
  await settle();
  assert.strictEqual(panel.requests.filter(function (r) { return r.url === ROUTES.runs; }).length, 2);
});

// %% the runs kept
test('each run kept is listed as its scene, designator and query, newest first', async function () {
  const panel = await mounted();
  assert.deepStrictEqual(
    runEntries(panel).map(function (entry) { return entry.textContent; }),
    runsPayload().runs.map(function (run) { return run.scene + ' · ' + run.designator + ' · ' + run.query; }),
  );
});

test('no run kept says so', async function () {
  const panel = await mounted({ runs: { ok: true, runs: [] } });
  assert.strictEqual(panel.elements['#resolution-runs'].textContent, 'no run kept yet');
});

test('runs that cannot be listed say why', async function () {
  const panel = await mounted({ runs: { ok: false, error: 'ValueError: unreadable' } });
  assert.ok(panel.elements['#resolution-runs'].textContent.includes('ValueError'));
});

test('clicking a run shows it again with its query in the box, resolving nothing', async function () {
  const panel = await mounted();
  runEntries(panel)[1].listeners.click();
  await settle();
  assert.strictEqual(panel.elements['#resolution-query'].value, resolvedPayload().resolution.query);
  assert.ok(panel.elements['#resolution-answer'].innerHTML.includes('tilt_angle = 1.85'));
  assert.deepStrictEqual(post(panel, ROUTES.resolve), []);
});

test('a run shown again can be executed', async function () {
  const panel = await mounted();
  runEntries(panel)[1].listeners.click();
  await settle();
  assert.strictEqual(panel.elements['#resolution-execute'].disabled, false);
});

test('a run that cannot be opened says why', async function () {
  const panel = await mounted({ openFails: true });
  runEntries(panel)[1].listeners.click();
  await settle();
  assert.ok(panel.elements['#resolution-answer'].innerHTML.includes('UnknownRun'));
  assert.strictEqual(panel.elements['#resolution-execute'].disabled, true);
});

test('a page opened with a run in its address shows that run', async function () {
  const panel = await mounted({ search: '?scene=simple_pouring&run=run-1' });
  assert.ok(panel.elements['#resolution-answer'].innerHTML.includes('tilt_angle = 1.85'));
  assert.strictEqual(panel.elements['#resolution-query'].value, resolvedPayload().resolution.query);
});

test('a page opened with no run in its address shows none', async function () {
  const panel = await mounted();
  assert.ok(!panel.requests.some(function (r) { return r.url === ROUTES.run; }));
});

// %% executing
test('Execute performs the run that was resolved', async function () {
  const panel = await mounted();
  resolve(panel, 'the(PouringAction)()');
  await settle();
  panel.elements['#resolution-execute'].listeners.click();
  await settle();
  assert.deepStrictEqual(bodyOf(post(panel, ROUTES.execute)[0]), { run: 'run-1' });
  assert.ok(panel.elements['#resolution-answer'].innerHTML.includes('executing run run-1'));
});

test('Execute performs nothing before a run is shown', async function () {
  const panel = await mounted();
  panel.elements['#resolution-execute'].listeners.click();
  await settle();
  assert.deepStrictEqual(post(panel, ROUTES.execute), []);
});

test('an execution that fails says why', async function () {
  const panel = await mounted({ executeFails: true });
  resolve(panel, 'the(PouringAction)()');
  await settle();
  panel.elements['#resolution-execute'].listeners.click();
  await settle();
  assert.ok(panel.elements['#resolution-answer'].innerHTML.includes('execution failed'));
  assert.strictEqual(panel.elements['#resolution-execute'].disabled, false);
});

test('asking to execute twice at once performs the run once', async function () {
  const panel = await mounted();
  resolve(panel, 'the(PouringAction)()');
  await settle();
  panel.elements['#resolution-execute'].listeners.click();
  panel.elements['#resolution-execute'].listeners.click();
  await settle();
  assert.strictEqual(post(panel, ROUTES.execute).length, 1);
});
