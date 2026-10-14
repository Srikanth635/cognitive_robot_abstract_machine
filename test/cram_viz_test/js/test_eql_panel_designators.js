// Tests the predefined designators in the EQL panel: chips reading the scene's
// underspecified pouring designators as krrood match queries, copied into the search bar
// on a click; Generate resolving the query in the search bar with llm_reasoner, the
// resolution shown in the answer panel and the consulted hypothesis published on the
// bus; Execute running the resolved designator in the simulation; and the fixed demo.
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const WEB_ROOT = path.join(
  __dirname, '..', '..', '..', 'cram_viz', 'src', 'cram_viz', 'web',
);

function element() {
  const created = {
    innerHTML: '',
    textContent: '',
    value: '',
    disabled: false,
    style: {},
    className: '',
    title: '',
    children: [],
    listeners: {},
    classes: new Set(),
    classList: {
      add(name) { this.owner.classes.add(name); },
      remove(name) { this.owner.classes.delete(name); },
      toggle(name, force) {
        const on = force === undefined ? !this.owner.classes.has(name) : force;
        if (on) this.owner.classes.add(name); else this.owner.classes.delete(name);
      },
      contains(name) { return this.owner.classes.has(name); },
    },
    addEventListener(event, listener) { this.listeners[event] = listener; },
    appendChild(child) { this.children.push(child); },
    scrollTop: 0,
  };
  created.classList.owner = created;
  return created;
}

function kbPayload() {
  return {
    ok: true,
    status: 'ready',
    presets: [{ text: 'what phases?', code: 'an(entity(pouring_phase))' }],
    details: {},
  };
}

function designatorsPayload() {
  return {
    ok: true,
    scene: 'simple_pouring',
    designators: [
      {
        identifier: 'pouring_cup_into_bowl',
        query: 'the(PouringAction)(\n    source_container=variable(Tool, domain=[pouring_cup]),\n)\n',
        query_on_one_line: 'the(PouringAction)( source_container=variable(Tool, domain=[pouring_cup]), )',
        available: true, unbound: [], without_candidates: [],
      },
      {
        identifier: 'from_the_pouring_cup',
        query: 'the(PouringAction)(\n    source_container=variable(Tool, domain=[pouring_cup]),\n)\n',
        query_on_one_line: 'the(PouringAction)( source_container=variable(Tool, domain=[pouring_cup]), )',
        available: false, unbound: ['pouring_cup'], without_candidates: [],
      },
      {
        identifier: 'everything_open',
        query: 'the(PouringAction)(\n    source_container=variable(Tool),\n)\n',
        query_on_one_line: 'the(PouringAction)( source_container=variable(Tool), )',
        available: false, unbound: [], without_candidates: ['source_container'],
      },
    ],
  };
}

function resolvedPayload() {
  return {
    ok: true,
    viewer: 'http://localhost:8721/?run=run-1',
    resolution: {
      identifier: 'pouring_cup_into_bowl',
      query: 'the(PouringAction)(\n    source_container=variable(Tool, domain=[pouring_cup]),\n)\n',
      designator: 'PouringAction',
      actions: [{
        slots: [
          { name: 'target_container', value: 'bowl.stl', given: true },
          { name: 'tilt_angle', value: '1.85', given: false },
        ],
        why: 'The default tilt lets the milk flow.',
        grounds: [{ source: 'knowledge', name: 'ph_tilt' }],
      }],
      situation: { has_material_regime: 'LIQUID' }, record_id: 'abc', cost: 0.05,
      run: 'run-1',
      trials: [
        {
          values: { pour_side_offset: 0.0 },
          outcome: {
            lands_in_opening: false, mouth_distance_from_opening_centre: 0.188,
            mouth_ahead_of_opening_centre: 0.002, mouth_right_of_opening_centre: 0.188,
            mouth_above_opening: 0.087,
          },
        },
        {
          values: { pour_side_offset: -0.188 },
          outcome: {
            lands_in_opening: true, mouth_distance_from_opening_centre: 0.005,
            mouth_ahead_of_opening_centre: 0.003, mouth_right_of_opening_centre: 0.004,
            mouth_above_opening: 0.089,
          },
        },
        { values: { pour_side_offset: -0.5 }, outcome: { lands_in_opening: false, motion_failed: 'MotionDidNotFinish: out of reach' } },
      ],
    },
    hypothesis: {
      ok: true, crumb: 'hypothesis graph', root: 'g0',
      nodes: [{ id: 'g0', label: 'PouringAction', kind: 'instruction' }], edges: [], details: {},
    },
  };
}

function loadEqlPanel(options) {
  const settings = options || {};
  const elements = {};
  const names = [
    '#kb-status', '#answer', '#query-input', '#query-run', '#presets',
    '#designators', '#query-generate', '#designator-execute', '#demo-execute',
    '#source-designators', '#source-questions',
  ];
  names.forEach(function (name) { elements[name] = element(); });

  const root = {
    innerHTML: '',
    querySelector(selector) { return elements[selector]; },
  };
  const handlers = {};
  const emitted = [];
  const bus = {
    on(event, listener) { (handlers[event] = handlers[event] || []).push(listener); },
    emit(event, payload) {
      emitted.push({ event: event, payload: payload });
      (handlers[event] || []).forEach(function (listener) { listener(payload); });
    },
  };

  const requests = [];
  global.fetch = async function (url, init) {
    requests.push({ url: url, init: init });
    if (url === '/api/kb') {
      return { status: 200, json: async function () { return kbPayload(); } };
    }
    if (url === '/api/designators') {
      return { status: 200, json: async function () { return designatorsPayload(); } };
    }
    if (url === '/api/designator/resolve') {
      if (settings.resolutionFails) {
        return {
          status: 200,
          json: async function () {
            return { ok: false, error: 'NoFittingReferent: nothing fits' };
          },
        };
      }
      return {
        status: 200,
        json: async function () { return settings.resolved || resolvedPayload(); },
      };
    }
    if (url === '/api/designator/run?name=run-1') {
      if (settings.openFails) {
        return {
          status: 200,
          json: async function () { return { ok: false, error: 'UnknownRun: no run of that name' }; },
        };
      }
      return {
        status: 200,
        json: async function () { return settings.resolved || resolvedPayload(); },
      };
    }
    if (url === '/api/execute') {
      if (settings.executeFails && JSON.parse(init.body).run === undefined) {
        return {
          status: 200,
          json: async function () {
            return { ok: false, error: 'DemoAlreadyRunningError: already running' };
          },
        };
      }
      return { status: 200, json: async function () { return { ok: true }; } };
    }
    throw new Error('Unexpected fetch: ' + url);
  };

  global.document = { createElement() { return element(); } };
  global.SceneSelection = {
    apiUrl(url) { return url; },
    nameFromSearch() { return 'simple_pouring'; },
  };
  global.window = { location: { search: settings.search || '' } };

  let factory = null;
  global.Panels = {
    define(identifier, panelFactory) {
      assert.strictEqual(identifier, 'eql');
      factory = panelFactory;
    },
  };
  const source = fs.readFileSync(path.join(WEB_ROOT, 'panels/eql/panel.js'), 'utf8');
  new Function(source)();
  factory(root, bus);
  return { root, bus, elements, requests, emitted };
}

async function settle() {
  for (let i = 0; i < 6; i += 1) {
    await new Promise(function (resolve) { setImmediate(resolve); });
  }
}

function emittedEvents(panel, name) {
  return panel.emitted.filter(function (entry) { return entry.event === name; });
}

// %% markup
test('the panel offers designators below the EQL query box', async function () {
  const panel = loadEqlPanel();
  await settle();

  assert.ok(panel.root.innerHTML.includes('id="designators"'));
  assert.ok(panel.root.innerHTML.includes('id="query-input"'));
  assert.ok(panel.root.innerHTML.includes('id="query-generate"'));
  assert.ok(!panel.root.innerHTML.includes('id="nl-input"'));
});

function chips(panel) {
  return panel.elements['#designators'].children;
}

test('each designator of the scene becomes a chip reading its match query on one line', async function () {
  const panel = loadEqlPanel();
  await settle();

  assert.deepStrictEqual(
    chips(panel).map(function (chip) { return chip.textContent; }),
    designatorsPayload().designators.map(function (d) { return d.query_on_one_line; }),
  );
});

test('a chip shows its whole match query on hover', async function () {
  const panel = loadEqlPanel();
  await settle();

  assert.ok(chips(panel)[0].title.includes(designatorsPayload().designators[0].query));
});

test('a designator naming an object the scene lacks says so and resolves nothing', async function () {
  const panel = loadEqlPanel();
  await settle();

  const unavailable = chips(panel)[1];
  assert.ok(unavailable.className.includes('unavailable'));
  assert.ok(unavailable.title.includes(designatorsPayload().designators[1].unbound[0]));
  assert.strictEqual(unavailable.listeners.click, undefined);
});

test('a designator with nothing in the scene to take for a slot says so', async function () {
  const panel = loadEqlPanel();
  await settle();

  const unavailable = chips(panel)[2];
  assert.ok(unavailable.className.includes('unavailable'));
  assert.ok(unavailable.title.includes(
    designatorsPayload().designators[2].without_candidates[0],
  ));
});

// %% the two lists: designators and questions
test('the designators show first and the questions are a tab away', async function () {
  const panel = loadEqlPanel();
  await settle();

  assert.ok(!panel.elements['#designators'].classes.has('hidden'));
  assert.ok(panel.elements['#presets'].classes.has('hidden'));
  assert.ok(panel.elements['#source-designators'].classes.has('on'));
});

test('the questions tab shows the questions in place of the designators', async function () {
  const panel = loadEqlPanel();
  await settle();

  panel.elements['#source-questions'].listeners.click();

  assert.ok(panel.elements['#designators'].classes.has('hidden'));
  assert.ok(!panel.elements['#presets'].classes.has('hidden'));
  assert.ok(panel.elements['#source-questions'].classes.has('on'));
  assert.ok(!panel.elements['#source-designators'].classes.has('on'));
});

// %% from a chip to the search bar
test('clicking a designator copies its query into the search bar and resolves nothing', async function () {
  const panel = loadEqlPanel();
  await settle();

  chips(panel)[0].listeners.click();
  await settle();

  assert.strictEqual(panel.elements['#query-input'].value, designatorsPayload().designators[0].query);
  assert.ok(!panel.requests.some(function (r) { return r.url === '/api/designator/resolve'; }));
});

// %% generating: resolving the query in the search bar
function generate(panel, query) {
  panel.elements['#query-input'].value = query;
  panel.elements['#query-generate'].listeners.click();
}

test('Generate resolves the query in the search bar in the scene', async function () {
  const panel = loadEqlPanel();
  await settle();

  generate(panel, designatorsPayload().designators[0].query);
  await settle();

  const call = panel.requests.find(function (r) { return r.url === '/api/designator/resolve'; });
  assert.ok(call);
  assert.deepStrictEqual(JSON.parse(call.init.body), {
    query: designatorsPayload().designators[0].query, scene: 'simple_pouring',
  });
});

test('a resolution publishes the consulted hypothesis and shows why', async function () {
  const panel = loadEqlPanel();
  await settle();

  generate(panel, designatorsPayload().designators[0].query);
  await settle();

  const published = emittedEvents(panel, 'hypothesis:generated');
  assert.strictEqual(published.length, 1);
  assert.strictEqual(published[0].payload.root, 'g0');
  const answer = panel.elements['#answer'].innerHTML;
  assert.ok(answer.includes(resolvedPayload().resolution.actions[0].why));
  assert.ok(answer.includes('ph_tilt'));
});

test('a resolution is shown from the top of the answer', async function () {
  const panel = loadEqlPanel();
  await settle();
  panel.elements['#answer'].scrollTop = 240;

  generate(panel, designatorsPayload().designators[0].query);
  await settle();

  assert.strictEqual(panel.elements['#answer'].scrollTop, 0);
});

test('a resolution shows the match query it resolved', async function () {
  const panel = loadEqlPanel();
  await settle();

  generate(panel, designatorsPayload().designators[0].query);
  await settle();

  const answer = panel.elements['#answer'].innerHTML;
  assert.ok(answer.includes(resolvedPayload().resolution.query.split('\n')[0]));
});

test('a resolution shows every pour it tried in the simulation', async function () {
  const panel = loadEqlPanel();
  await settle();

  generate(panel, designatorsPayload().designators[0].query);
  await settle();

  const answer = panel.elements['#answer'].innerHTML;
  const [missed, landed, failed] = resolvedPayload().resolution.trials;
  assert.ok(answer.includes('missed'));
  assert.ok(answer.includes(String(missed.outcome.mouth_distance_from_opening_centre)));
  assert.ok(answer.includes('landed'));
  assert.ok(answer.includes(String(landed.outcome.mouth_distance_from_opening_centre)));
  assert.ok(answer.includes(failed.outcome.motion_failed));
});

test('a resolution names the run it was made in', async function () {
  const panel = loadEqlPanel();
  await settle();

  generate(panel, designatorsPayload().designators[0].query);
  await settle();

  assert.ok(panel.elements['#answer'].innerHTML.includes(resolvedPayload().resolution.run));
});

test('a resolution links to the viewer of the system at work on its run', async function () {
  const panel = loadEqlPanel();
  await settle();

  generate(panel, designatorsPayload().designators[0].query);
  await settle();

  assert.ok(panel.elements['#answer'].innerHTML.includes(
    'href="' + resolvedPayload().viewer + '"'));
});

test('a resolution the server gives no viewer for shows no link', async function () {
  const payload = resolvedPayload();
  delete payload.viewer;
  const panel = loadEqlPanel({ resolved: payload });
  await settle();

  generate(panel, designatorsPayload().designators[0].query);
  await settle();

  assert.ok(!panel.elements['#answer'].innerHTML.includes('<a '));
});

test('a resolution refreshes the presets, since the new knowledge changes them', async function () {
  const panel = loadEqlPanel();
  await settle();
  const before = panel.requests.filter(function (r) { return r.url === '/api/kb'; }).length;

  generate(panel, designatorsPayload().designators[0].query);
  await settle();

  const after = panel.requests.filter(function (r) { return r.url === '/api/kb'; }).length;
  assert.strictEqual(after, before + 1);
});

test('a failed resolution reports the error and publishes nothing', async function () {
  const panel = loadEqlPanel({ resolutionFails: true });
  await settle();

  generate(panel, designatorsPayload().designators[0].query);
  await settle();

  assert.strictEqual(emittedEvents(panel, 'hypothesis:generated').length, 0);
  assert.ok(panel.elements['#answer'].innerHTML.includes('NoFittingReferent'));
});

// %% executing the resolved designator in the simulation
test('Execute waits for a resolution', async function () {
  const panel = loadEqlPanel();
  await settle();

  assert.strictEqual(panel.elements['#designator-execute'].disabled, true);
});

test('Execute runs the resolution in the simulation, by its run', async function () {
  const panel = loadEqlPanel();
  await settle();
  generate(panel, designatorsPayload().designators[0].query);
  await settle();

  assert.strictEqual(panel.elements['#designator-execute'].disabled, false);
  panel.elements['#designator-execute'].listeners.click();
  await settle();

  const call = panel.requests.find(function (r) { return r.url === '/api/execute'; });
  assert.deepStrictEqual(JSON.parse(call.init.body), { run: resolvedPayload().resolution.run });
});

test('a started execution names its run and says to watch it live', async function () {
  const panel = loadEqlPanel();
  await settle();
  generate(panel, designatorsPayload().designators[0].query);
  await settle();

  panel.elements['#designator-execute'].listeners.click();
  await settle();

  const answer = panel.elements['#answer'].innerHTML;
  assert.ok(answer.includes(resolvedPayload().resolution.run));
  assert.ok(answer.includes('Live'));
});

test('a failed resolution leaves nothing to execute', async function () {
  const panel = loadEqlPanel({ resolutionFails: true });
  await settle();
  generate(panel, designatorsPayload().designators[0].query);
  await settle();

  assert.strictEqual(panel.elements['#designator-execute'].disabled, true);
});

// %% the fixed demo, kept as an option beside the designators
test('the demo button launches the fixed demo', async function () {
  const panel = loadEqlPanel();
  await settle();

  panel.elements['#demo-execute'].listeners.click();
  await settle();

  const call = panel.requests.find(function (r) { return r.url === '/api/execute'; });
  assert.ok(call);
  assert.strictEqual(call.init.method, 'POST');
  assert.deepStrictEqual(JSON.parse(call.init.body), {});
});

test('a launched demo is reported', async function () {
  const panel = loadEqlPanel();
  await settle();

  panel.elements['#demo-execute'].listeners.click();
  await settle();

  assert.ok(panel.elements['#answer'].innerHTML.includes('demo launched'));
});

test('a failed demo launch reports the error', async function () {
  const panel = loadEqlPanel({ executeFails: true });
  await settle();

  panel.elements['#demo-execute'].listeners.click();
  await settle();

  assert.ok(panel.elements['#answer'].innerHTML.includes('DemoAlreadyRunningError'));
});

test('launching the demo resolves nothing and does not reload the presets', async function () {
  const panel = loadEqlPanel();
  await settle();
  const kbCallsBefore = panel.requests.filter(function (r) { return r.url === '/api/kb'; }).length;

  panel.elements['#demo-execute'].listeners.click();
  await settle();

  assert.ok(!panel.requests.some(function (r) { return r.url === '/api/designator/resolve'; }));
  const kbCallsAfter = panel.requests.filter(function (r) { return r.url === '/api/kb'; }).length;
  assert.strictEqual(kbCallsAfter, kbCallsBefore);
});

// %% opening a run from the address
test('a page opened with a run shows what the run resolved, without resolving again', async function () {
  const panel = loadEqlPanel({ search: '?scene=simple_pouring&run=run-1' });
  await settle();

  assert.ok(panel.requests.some(function (request) { return request.url === '/api/designator/run?name=run-1'; }));
  assert.ok(!panel.requests.some(function (request) { return request.url === '/api/designator/resolve'; }));
  assert.ok(panel.elements['#answer'].innerHTML.includes(resolvedPayload().resolution.actions[0].why));
  assert.ok(panel.elements['#answer'].innerHTML.includes(resolvedPayload().viewer));
});

test('the query of the run opened is put in the search bar', async function () {
  const panel = loadEqlPanel({ search: '?run=run-1' });
  await settle();

  assert.strictEqual(panel.elements['#query-input'].value, resolvedPayload().resolution.query);
});

test('the hypothesis the run consulted is published on the bus', async function () {
  const panel = loadEqlPanel({ search: '?run=run-1' });
  await settle();

  const [published] = emittedEvents(panel, 'hypothesis:generated');
  assert.deepStrictEqual(published.payload, resolvedPayload().hypothesis);
});

test('the run opened is the one Execute performs', async function () {
  const panel = loadEqlPanel({ search: '?run=run-1' });
  await settle();

  assert.strictEqual(panel.elements['#designator-execute'].disabled, false);
  panel.elements['#designator-execute'].listeners.click();
  await settle();
  const launch = panel.requests.find(function (request) { return request.url === '/api/execute'; });
  assert.deepStrictEqual(JSON.parse(launch.init.body), { run: 'run-1' });
});

test('a run that cannot be opened says why and leaves Execute off', async function () {
  const panel = loadEqlPanel({ search: '?run=run-1', openFails: true });
  await settle();

  assert.ok(panel.elements['#answer'].innerHTML.includes('UnknownRun'));
  assert.strictEqual(panel.elements['#designator-execute'].disabled, true);
});

test('a page opened with no run asks for none', async function () {
  const panel = loadEqlPanel();
  await settle();

  assert.ok(!panel.requests.some(function (request) { return request.url.startsWith('/api/designator/run'); }));
});
