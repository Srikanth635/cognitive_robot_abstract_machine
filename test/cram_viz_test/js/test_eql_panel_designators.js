// Tests the predefined designators in the EQL panel: chips for the scene's underspecified
// pouring designators, each resolved by llm_reasoner in the scene on a click, the
// resolution shown in the answer panel and the consulted hypothesis published on the
// bus; and the Execute button beside them.
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const WEB_ROOT = path.join(
  __dirname, '..', '..', '..', 'cram_viz', 'src', 'cram_viz', 'web',
);

function element() {
  return {
    innerHTML: '',
    textContent: '',
    value: '',
    disabled: false,
    style: {},
    className: '',
    title: '',
    children: [],
    listeners: {},
    classList: { add() {}, remove() {}, toggle() {} },
    addEventListener(event, listener) { this.listeners[event] = listener; },
    appendChild(child) { this.children.push(child); },
  };
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
        identifier: 'vessel_into_bowl', label: 'pour from the vessel into the bowl',
        task: '', available: true, missing: [],
        given: ['target_container', 'source_container'], left_open: ['arm', 'tilt_angle'],
      },
      {
        identifier: 'until_half_full', label: 'into the bowl until it is half full',
        task: 'Pour the milk into the bowl until the bowl is half full.', available: false,
        missing: ['PouringCup'], given: [], left_open: [],
      },
    ],
  };
}

function resolvedPayload() {
  return {
    ok: true,
    resolution: {
      variant: 'vessel_into_bowl', label: 'pour from the vessel into the bowl', task: '',
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
    '#designators', '#nl-execute',
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
      return { status: 200, json: async function () { return resolvedPayload(); } };
    }
    if (url === '/api/execute') {
      if (settings.executeFails) {
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
  global.window = { location: { search: '' } };

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
  assert.ok(!panel.root.innerHTML.includes('id="nl-input"'));
});

function chips(panel) {
  return panel.elements['#designators'].children;
}

test('each designator of the scene becomes a chip reading its label', async function () {
  const panel = loadEqlPanel();
  await settle();

  assert.deepStrictEqual(
    chips(panel).map(function (chip) { return chip.textContent; }),
    designatorsPayload().designators.map(function (d) { return d.label; }),
  );
});

test('an unavailable designator says what the scene lacks and resolves nothing', async function () {
  const panel = loadEqlPanel();
  await settle();

  const unavailable = chips(panel)[1];
  assert.ok(unavailable.className.includes('unavailable'));
  assert.ok(unavailable.title.includes('PouringCup'));
  assert.strictEqual(unavailable.listeners.click, undefined);
});

// %% resolving
test('clicking a designator asks the server to resolve it in the scene', async function () {
  const panel = loadEqlPanel();
  await settle();

  chips(panel)[0].listeners.click();
  await settle();

  const call = panel.requests.find(function (r) { return r.url === '/api/designator/resolve'; });
  assert.ok(call);
  assert.deepStrictEqual(JSON.parse(call.init.body), {
    variant: 'vessel_into_bowl', scene: 'simple_pouring',
  });
});

test('a resolution publishes the consulted hypothesis and shows why', async function () {
  const panel = loadEqlPanel();
  await settle();

  chips(panel)[0].listeners.click();
  await settle();

  const published = emittedEvents(panel, 'hypothesis:generated');
  assert.strictEqual(published.length, 1);
  assert.strictEqual(published[0].payload.root, 'g0');
  const answer = panel.elements['#answer'].innerHTML;
  assert.ok(answer.includes(resolvedPayload().resolution.actions[0].why));
  assert.ok(answer.includes('ph_tilt'));
});

test('a resolution refreshes the presets, since the new knowledge changes them', async function () {
  const panel = loadEqlPanel();
  await settle();
  const before = panel.requests.filter(function (r) { return r.url === '/api/kb'; }).length;

  chips(panel)[0].listeners.click();
  await settle();

  const after = panel.requests.filter(function (r) { return r.url === '/api/kb'; }).length;
  assert.strictEqual(after, before + 1);
});

test('a failed resolution reports the error and publishes nothing', async function () {
  const panel = loadEqlPanel({ resolutionFails: true });
  await settle();

  chips(panel)[0].listeners.click();
  await settle();

  assert.strictEqual(emittedEvents(panel, 'hypothesis:generated').length, 0);
  assert.ok(panel.elements['#answer'].innerHTML.includes('NoFittingReferent'));
});

// %% executing -- independent of the designators beside it
test('clicking Execute posts to the execute endpoint', async function () {
  const panel = loadEqlPanel();
  await settle();

  panel.elements['#nl-execute'].listeners.click();
  await settle();

  const call = panel.requests.find(function (r) { return r.url === '/api/execute'; });
  assert.ok(call);
  assert.strictEqual(call.init.method, 'POST');
});

test('a successful execute reports the demo was launched', async function () {
  const panel = loadEqlPanel();
  await settle();

  panel.elements['#nl-execute'].listeners.click();
  await settle();

  assert.ok(panel.elements['#answer'].innerHTML.includes('demo launched'));
});

test('a failed execute reports the error', async function () {
  const panel = loadEqlPanel({ executeFails: true });
  await settle();

  panel.elements['#nl-execute'].listeners.click();
  await settle();

  assert.ok(panel.elements['#answer'].innerHTML.includes('DemoAlreadyRunningError'));
});

test('executing resolves nothing and does not reload the presets', async function () {
  const panel = loadEqlPanel();
  await settle();
  const kbCallsBefore = panel.requests.filter(function (r) { return r.url === '/api/kb'; }).length;

  panel.elements['#nl-execute'].listeners.click();
  await settle();

  assert.ok(!panel.requests.some(function (r) { return r.url === '/api/designator/resolve'; }));
  const kbCallsAfter = panel.requests.filter(function (r) { return r.url === '/api/kb'; }).length;
  assert.strictEqual(kbCallsAfter, kbCallsBefore);
});
