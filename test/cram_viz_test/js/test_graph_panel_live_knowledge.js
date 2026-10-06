// Tests that static and live knowledge have independent graph-panel tabs.
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const WEB_ROOT = path.join(
  __dirname,
  '..',
  '..',
  '..',
  'cram_viz',
  'src',
  'cram_viz',
  'web',
);

// %% Panel fixture
function graphPayload(label, key) {
  return {
    ok: true,
    crumb: label,
    nodes: [{ id: key, label: label, group: 'goal' }],
    edges: [],
    details: {},
    key: key,
    sig: key,
    highlight: [],
  };
}

function classList() {
  return {
    values: new Set(),
    toggle(name, enabled) {
      if (enabled) this.values.add(name); else this.values.delete(name);
    },
    add(name) { this.values.add(name); },
    remove(name) { this.values.delete(name); },
  };
}

function element() {
  return {
    style: {},
    classList: classList(),
    textContent: '',
    listeners: {},
    addEventListener(event, listener) { this.listeners[event] = listener; },
  };
}

function loadGraphPanel() {
  const viewNames = [
    'knowledge',
    'live-knowledge',
    'domain',
    'kinematics',
    'plan',
    'chart',
  ];
  const buttons = viewNames.map(function (viewName) {
    const button = element();
    button.dataset = { view: viewName };
    return button;
  });
  const tabs = element();
  tabs.querySelectorAll = function () { return buttons; };
  const elements = {
    '#graph-empty': element(),
    '#graph-nav': element(),
    '#gnav-up': element(),
    '#gnav-home': element(),
    '#gnav-path': element(),
    '#graph-tabs': tabs,
    '#gt-live': element(),
    '#graph': element(),
    '#hypothesis-graph': element(),
    '#hypothesis-toolbar': element(),
    '#hg-expand-all': element(),
    '#hg-collapse-all': element(),
  };
  const root = {
    innerHTML: '',
    querySelector(selector) { return elements[selector]; },
  };
  const handlers = {};
  const bus = {
    on(event, listener) { handlers[event] = listener; },
    emit(event, payload) { if (handlers[event]) handlers[event](payload); },
  };
  const builds = [];
  global.Graph = {
    build(payload) { builds.push(payload); },
    onSelect() {},
    onDoubleSelect() {},
    highlight() {},
    reset() {},
    setStatuses() { return true; },
  };
  const staticPayload = graphPayload('Static Knowledge', 'static-knowledge');
  const livePayload = graphPayload('Live Knowledge', 'live-knowledge');
  const fetchCalls = [];
  global.fetch = async function (url) {
    fetchCalls.push(url);
    if (url === '/api/kb') {
      return { status: 200, json: async function () { return staticPayload; } };
    }
    if (url === 'http://localhost:8765/plan') {
      return { status: 200, json: async function () { return { sig: 'live', nodes: [] }; } };
    }
    if (url === 'http://localhost:8765/info') {
      return { status: 200, json: async function () { return { robot: 'PR2', objects: [] }; } };
    }
    throw new Error('Unexpected fetch: ' + url);
  };
  global.LiveKnowledgeGraph = class {
    build() { return livePayload; }
  };
  global.HypothesisGraph = {
    build() {},
    highlight() {},
    reset() {},
    resize() {},
    showLegend() {},
    onSelect() {},
  };
  global.SceneSelection = { apiUrl(url) { return url; } };
  global.window = { location: { search: '' } };
  global.setInterval = function () { return 1; };
  global.clearInterval = function () {};
  let factory = null;
  global.Panels = {
    define(identifier, panelFactory) {
      assert.strictEqual(identifier, 'graph');
      factory = panelFactory;
    },
  };
  const source = fs.readFileSync(
    path.join(WEB_ROOT, 'panels/graph/panel.js'),
    'utf8',
  );
  new Function(source)();
  factory(root, bus);
  return { root, bus, buttons, builds, fetchCalls };
}

async function settle() {
  await new Promise(function (resolve) { setImmediate(resolve); });
  await new Promise(function (resolve) { setImmediate(resolve); });
}

// %% Independent knowledge tabs
test('live attachment preserves static Knowledge and renders in Live Knowledge', async function () {
  const panel = loadGraphPanel();
  await settle();

  assert.ok(panel.root.innerHTML.includes('data-view="live-knowledge"'));
  assert.strictEqual(panel.builds.at(-1).nodes[0].label, 'Static Knowledge');

  panel.bus.emit('live:changed', { on: true, url: 'http://localhost:8765' });
  await settle();

  assert.strictEqual(panel.builds.at(-1).nodes[0].label, 'Static Knowledge');
  assert.ok(!panel.fetchCalls.includes('http://localhost:8765/plan'));

  const liveButton = panel.buttons.find(function (button) {
    return button.dataset.view === 'live-knowledge';
  });
  liveButton.listeners.click();
  await settle();

  assert.strictEqual(panel.builds.at(-1).nodes[0].label, 'Live Knowledge');
  assert.ok(panel.fetchCalls.includes('http://localhost:8765/plan'));
  assert.ok(panel.fetchCalls.includes('http://localhost:8765/info'));
});
