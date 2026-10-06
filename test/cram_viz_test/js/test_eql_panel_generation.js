// Tests the natural-language generation bar in the EQL panel: it posts an
// instruction to the generation endpoint and publishes the refreshed hypothesis
// graph on the bus, without disturbing the EQL query box beside it.
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

function generatedPayload() {
  return {
    ok: true,
    crumb: 'hypothesis graph',
    root: 'g0',
    nodes: [{ id: 'g0', label: 'Pour milk into the mug', kind: 'instruction' }],
    edges: [],
    details: {},
    instruction: 'Pour milk into the mug',
  };
}

function loadEqlPanel(options) {
  const settings = options || {};
  const elements = {};
  const names = [
    '#kb-status', '#answer', '#query-input', '#query-run', '#presets',
    '#nl-input', '#nl-run', '#nl-execute',
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
    if (url === '/api/hypothesis/generate') {
      if (settings.generationFails) {
        return {
          status: 200,
          json: async function () {
            return { ok: false, error: 'RepairBudgetExhausted: gave up' };
          },
        };
      }
      return { status: 200, json: async function () { return generatedPayload(); } };
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
    nameFromSearch() { return null; },
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
test('the panel renders a natural-language bar alongside the EQL query box', async function () {
  const panel = loadEqlPanel();
  await settle();

  assert.ok(panel.root.innerHTML.includes('id="nl-input"'));
  assert.ok(panel.root.innerHTML.includes('id="nl-run"'));
  assert.ok(panel.root.innerHTML.includes('id="nl-execute"'));
  assert.ok(panel.root.innerHTML.includes('id="query-input"'));
});

// %% generating
test('running an instruction posts it to the generation endpoint', async function () {
  const panel = loadEqlPanel();
  await settle();

  panel.elements['#nl-input'].value = 'Pour milk into the mug';
  panel.elements['#nl-run'].listeners.click();
  await settle();

  const call = panel.requests.find(function (r) {
    return r.url === '/api/hypothesis/generate';
  });
  assert.ok(call);
  assert.strictEqual(call.init.method, 'POST');
  assert.deepStrictEqual(JSON.parse(call.init.body), {
    instruction: 'Pour milk into the mug',
  });
});

test('a successful generation publishes the refreshed graph on the bus', async function () {
  const panel = loadEqlPanel();
  await settle();

  panel.elements['#nl-input'].value = 'Pour milk into the mug';
  panel.elements['#nl-run'].listeners.click();
  await settle();

  const generated = emittedEvents(panel, 'hypothesis:generated');
  assert.strictEqual(generated.length, 1);
  assert.strictEqual(generated[0].payload.root, 'g0');
});

test('an empty instruction does not call the endpoint', async function () {
  const panel = loadEqlPanel();
  await settle();

  panel.elements['#nl-input'].value = '   ';
  panel.elements['#nl-run'].listeners.click();
  await settle();

  assert.ok(!panel.requests.some(function (r) {
    return r.url === '/api/hypothesis/generate';
  }));
});

test('a failed generation reports the error and publishes nothing', async function () {
  const panel = loadEqlPanel({ generationFails: true });
  await settle();

  panel.elements['#nl-input'].value = 'Pour milk into the mug';
  panel.elements['#nl-run'].listeners.click();
  await settle();

  assert.strictEqual(emittedEvents(panel, 'hypothesis:generated').length, 0);
  assert.ok(panel.elements['#answer'].innerHTML.includes('RepairBudgetExhausted'));
});

test('generating refreshes the presets, since the new structure changes them', async function () {
  const panel = loadEqlPanel();
  await settle();
  const kbCallsBefore = panel.requests.filter(function (r) {
    return r.url === '/api/kb';
  }).length;

  panel.elements['#nl-input'].value = 'Pour milk into the mug';
  panel.elements['#nl-run'].listeners.click();
  await settle();

  const kbCallsAfter = panel.requests.filter(function (r) {
    return r.url === '/api/kb';
  }).length;
  assert.strictEqual(kbCallsAfter, kbCallsBefore + 1);
});

// %% executing -- independent of the instruction bar beside it
test('clicking Execute posts to the execute endpoint, with no instruction needed', async function () {
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

test('executing does not touch generation or the presets/kb reload', async function () {
  const panel = loadEqlPanel();
  await settle();
  const kbCallsBefore = panel.requests.filter(function (r) { return r.url === '/api/kb'; }).length;

  panel.elements['#nl-execute'].listeners.click();
  await settle();

  assert.ok(!panel.requests.some(function (r) { return r.url === '/api/hypothesis/generate'; }));
  const kbCallsAfter = panel.requests.filter(function (r) { return r.url === '/api/kb'; }).length;
  assert.strictEqual(kbCallsAfter, kbCallsBefore);
});
