// Tests that the graph panel wires the Hypothesis Graph tab to its own dedicated
// renderer (HypothesisGraph), overlaying cram_viz's EQL answer panel / highlight loop
// on top, instead of going through graph.js's shared window.Graph.
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

function hypothesisPayload() {
  return {
    ok: true,
    crumb: 'hypothesis graph',
    root: 'h0',
    nodes: [
      { id: 'h0', label: 'Root', kind: 'instruction' },
      { id: 'h1', label: 'Child', kind: 'root' },
    ],
    edges: [{ from: 'h0', to: 'h1', label: 'describes' }],
    details: {
      h0: { label: 'Root', group: 'hypothesis', lines: ['line-root'] },
      h1: { label: 'Child', group: 'hypothesis', lines: ['line-child'] },
    },
    empty: 'No hypothesis knowledge graph available.',
  };
}

function ontologyPayload() {
  const payload = hypothesisPayload();
  payload.crumb = 'ontology graph';
  payload.nodes[1].label = 'Child\npouring:PouringPhaseStructure';
  payload.edges[0].label = 'describes\n(drawn by the viewer)';
  payload.details.h1.lines = ['class: pouring:PouringPhaseStructure'];
  return payload;
}

function loadGraphPanel(options) {
  const settings = options || {};
  const viewNames = [
    'knowledge',
    'live-knowledge',
    'domain',
    'hypothesis',
    'ontology',
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
  const hgExpandAllBtn = element();
  const hgCollapseAllBtn = element();
  const elements = {
    '#graph-empty': element(),
    '#graph-nav': element(),
    '#gnav-up': element(),
    '#gnav-home': element(),
    '#gnav-path': element(),
    '#graph-tabs': tabs,
    '#gt-live': element(),
    '.graph-wrap': element(),
    '#graph': element(),
    '#hypothesis-graph': element(),
    '#hypothesis-toolbar': element(),
    '#hg-expand-all': hgExpandAllBtn,
    '#hg-collapse-all': hgCollapseAllBtn,
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
  global.DesignatorLisp = { build() {}, highlight() {}, onSelect() {} };
  global.Graph = {
    build(payload) { builds.push(payload); },
    onSelect() {},
    onDoubleSelect() {},
    highlight() {},
    reset() {},
    setStatuses() { return true; },
  };
  let hypothesisSelectCallback = null;
  const hypothesisBuilds = [];
  const hypothesisHighlightCalls = [];
  const hypothesisRevealAndHighlightCalls = [];
  let hypothesisResetCalls = 0;
  let hypothesisLegendCalls = 0;
  let hypothesisExpandAllCalls = 0;
  let hypothesisCollapseAllCalls = 0;
  global.HypothesisGraph = {
    build(payload) { hypothesisBuilds.push(payload); },
    highlight(ids) { hypothesisHighlightCalls.push(ids); },
    revealAndHighlight(ids) { hypothesisRevealAndHighlightCalls.push(ids); },
    reset() { hypothesisResetCalls += 1; },
    resize() {},
    showLegend() { hypothesisLegendCalls += 1; },
    expandAll() { hypothesisExpandAllCalls += 1; },
    collapseAll() { hypothesisCollapseAllCalls += 1; },
    onSelect(cb) { hypothesisSelectCallback = cb; },
  };
  const staticPayload = {
    ok: true, crumb: 'Static Knowledge', nodes: [{ id: 'static-knowledge', label: 'Static Knowledge', group: 'goal' }],
    edges: [], details: {}, key: 'static-knowledge', sig: 'static-knowledge', highlight: [],
  };
  const fetchCalls = [];
  global.fetch = async function (url) {
    fetchCalls.push(url);
    if (url === '/api/kb') {
      return { status: 200, json: async function () { return staticPayload; } };
    }
    if (url === '/api/kb/view?name=hypothesis') {
      return {
        status: 200,
        json: async function () { return settings.hypothesis || hypothesisPayload(); },
      };
    }
    if (url === '/api/kb/view?name=ontology') {
      return {
        status: 200,
        json: async function () { return settings.ontology || ontologyPayload(); },
      };
    }
    throw new Error('Unexpected fetch: ' + url);
  };
  global.LiveKnowledgeGraph = class {
    build() {
      return {
        ok: true, crumb: 'live', nodes: [], edges: [], details: {}, key: 'knowledge-live', sig: '', highlight: [],
      };
    }
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
  return {
    root, bus, buttons, builds, fetchCalls, elements,
    hypothesisBuilds, hypothesisHighlightCalls, hypothesisRevealAndHighlightCalls,
    hgExpandAllBtn, hgCollapseAllBtn,
    hypothesisResetCallCount() { return hypothesisResetCalls; },
    hypothesisLegendCallCount() { return hypothesisLegendCalls; },
    hypothesisExpandAllCallCount() { return hypothesisExpandAllCalls; },
    hypothesisCollapseAllCallCount() { return hypothesisCollapseAllCalls; },
    fireHypothesisSelect(id) { hypothesisSelectCallback(id); },
  };
}

async function settle() {
  await new Promise(function (resolve) { setImmediate(resolve); });
  await new Promise(function (resolve) { setImmediate(resolve); });
}

function tabButton(panel, name) {
  return panel.buttons.find(function (button) { return button.dataset.view === name; });
}

// %% Opening the tab
test('opening the Hypothesis Graph tab fetches its data once and builds HypothesisGraph', async function () {
  const panel = loadGraphPanel();
  await settle();

  tabButton(panel, 'hypothesis').listeners.click();
  await settle();

  assert.ok(panel.fetchCalls.includes('/api/kb/view?name=hypothesis'));
  assert.strictEqual(panel.hypothesisBuilds.length, 1);
  assert.strictEqual(panel.hypothesisBuilds[0].root, 'h0');
  assert.strictEqual(panel.hypothesisLegendCallCount(), 1);
});

test('revisiting the Hypothesis Graph tab does not refetch or rebuild', async function () {
  const panel = loadGraphPanel();
  await settle();
  tabButton(panel, 'hypothesis').listeners.click();
  await settle();
  tabButton(panel, 'knowledge').listeners.click();
  await settle();

  tabButton(panel, 'hypothesis').listeners.click();
  await settle();

  assert.strictEqual(
    panel.fetchCalls.filter(function (url) { return url === '/api/kb/view?name=hypothesis'; }).length,
    1,
  );
  assert.strictEqual(panel.hypothesisBuilds.length, 1);
  // the legend is repopulated on every visit even though build() only runs once
  assert.strictEqual(panel.hypothesisLegendCallCount(), 2);
});

// %% container visibility
test('showing the Hypothesis Graph tab hides #graph and shows #hypothesis-graph', async function () {
  const panel = loadGraphPanel();
  await settle();

  tabButton(panel, 'hypothesis').listeners.click();
  await settle();

  assert.strictEqual(panel.elements['#graph'].style.display, 'none');
  assert.notStrictEqual(panel.elements['#hypothesis-graph'].style.display, 'none');

  tabButton(panel, 'knowledge').listeners.click();
  await settle();

  assert.strictEqual(panel.elements['#hypothesis-graph'].style.display, 'none');
  assert.notStrictEqual(panel.elements['#graph'].style.display, 'none');
});

// %% click overlay -> EQL answer panel
test('a HypothesisGraph selection emits entity:select with real detail and relations', async function () {
  const panel = loadGraphPanel();
  await settle();
  tabButton(panel, 'hypothesis').listeners.click();
  await settle();

  let received = null;
  panel.bus.on('entity:select', function (payload) { received = payload; });
  panel.fireHypothesisSelect('h0');

  assert.strictEqual(received.id, 'h0');
  assert.strictEqual(received.detail.label, 'Root');
  assert.deepStrictEqual(received.relations, [{ s: 'Root', p: 'describes', o: 'Child' }]);
});

// a plain node click just spotlights within whatever's already on screen -- the
// clicked node is visible by definition, so there is nothing to reveal. Only EQL
// query results (which can point at a node buried in a collapsed branch) go through
// revealAndHighlight, which resets and rebuilds the expanded set; doing that on every
// click would undo the click's own expand() and any other manual exploration.
test('a HypothesisGraph selection spotlights via highlight, not revealAndHighlight, so manual expansion state is preserved', async function () {
  const panel = loadGraphPanel();
  await settle();
  tabButton(panel, 'hypothesis').listeners.click();
  await settle();

  panel.fireHypothesisSelect('h1');

  assert.deepStrictEqual(panel.hypothesisHighlightCalls.at(-1), ['h1', 'h0']);
  assert.strictEqual(panel.hypothesisRevealAndHighlightCalls.length, 0);
});

// eql/panel.js's describe() -- registered on the same real bus in the running app,
// but not loaded by this file's harness -- always re-emits entity:highlight with a
// focus after any entity:select. This wires that exact cascade explicitly, since
// the isolated fireHypothesisSelect() tests above never trigger it and so cannot
// catch a regression here.
function wireDescribeCascade(panel) {
  panel.bus.on('entity:select', function (p) {
    panel.bus.emit('entity:highlight', { ids: [p.id], focus: p.id });
  });
}

test('a click, once eql/panel.js\'s describe() cascade re-emits entity:highlight, still spotlights via highlight rather than revealAndHighlight', async function () {
  const panel = loadGraphPanel();
  await settle();
  tabButton(panel, 'hypothesis').listeners.click();
  await settle();
  wireDescribeCascade(panel);

  panel.fireHypothesisSelect('h1');

  // the cascade must not fall back to a full reveal-from-root -- that would
  // discard whatever branch the click's own expand() just set up, replacing it
  // with the ancestor paths of h1's *other* neighbours (which, for a node shared
  // by several parents, can be an entirely unrelated branch)
  assert.strictEqual(panel.hypothesisRevealAndHighlightCalls.length, 0);
  assert.deepStrictEqual(panel.hypothesisHighlightCalls.at(-1), ['h1', 'h0']);
});

// %% highlight routing
test('entity:highlight routes to HypothesisGraph.revealAndHighlight while that tab is active, and to Graph otherwise', async function () {
  const panel = loadGraphPanel();
  await settle();
  tabButton(panel, 'hypothesis').listeners.click();
  await settle();

  panel.bus.emit('entity:highlight', { ids: ['h1'] });
  assert.deepStrictEqual(panel.hypothesisRevealAndHighlightCalls.at(-1), ['h1']);

  tabButton(panel, 'knowledge').listeners.click();
  await settle();
  const revealCallsBefore = panel.hypothesisRevealAndHighlightCalls.length;

  panel.bus.emit('entity:highlight', { ids: ['static-knowledge'] });
  assert.strictEqual(panel.hypothesisRevealAndHighlightCalls.length, revealCallsBefore);
});

// %% nothing generated yet
function emptyHypothesisPayload() {
  return {
    ok: true,
    crumb: 'hypothesis graph',
    root: null,
    nodes: [],
    edges: [],
    details: {},
    empty: 'Nothing generated yet — describe an action in the Generate bar.',
  };
}

test('an empty hypothesis view shows its prompt instead of a blank tile', async function () {
  const panel = loadGraphPanel({ hypothesis: emptyHypothesisPayload() });
  await settle();

  tabButton(panel, 'hypothesis').listeners.click();
  await settle();

  assert.notStrictEqual(panel.elements['#graph-empty'].style.display, 'none');
  assert.ok(panel.elements['#graph-empty'].textContent.includes('Nothing generated yet'));
});

test('generating after an empty view hides the prompt again', async function () {
  const panel = loadGraphPanel({ hypothesis: emptyHypothesisPayload() });
  await settle();
  tabButton(panel, 'hypothesis').listeners.click();
  await settle();

  panel.bus.emit('hypothesis:generated', regeneratedPayload());
  await settle();

  assert.strictEqual(panel.elements['#graph-empty'].style.display, 'none');
  assert.strictEqual(panel.hypothesisBuilds.at(-1).root, 'n0');
});

// %% freshly generated knowledge
function regeneratedPayload() {
  return {
    ok: true,
    crumb: 'hypothesis graph',
    root: 'n0',
    nodes: [{ id: 'n0', label: 'Pour milk into the mug', kind: 'instruction' }],
    edges: [],
    details: { n0: { label: 'Pour milk into the mug', group: 'hypothesis', lines: [] } },
    instruction: 'Pour milk into the mug',
  };
}

test('a generated hypothesis rebuilds the graph and opens its tab', async function () {
  const panel = loadGraphPanel();
  await settle();

  panel.bus.emit('hypothesis:generated', regeneratedPayload());
  await settle();

  assert.strictEqual(panel.hypothesisBuilds.length, 1);
  assert.strictEqual(panel.hypothesisBuilds.at(-1).root, 'n0');
  assert.notStrictEqual(panel.elements['#hypothesis-graph'].style.display, 'none');
  assert.strictEqual(panel.elements['#graph'].style.display, 'none');
});

test('a generated hypothesis replaces the cached payload rather than refetching', async function () {
  const panel = loadGraphPanel();
  await settle();
  tabButton(panel, 'hypothesis').listeners.click();
  await settle();
  const fetchesBefore = panel.fetchCalls.filter(function (url) {
    return url === '/api/kb/view?name=hypothesis';
  }).length;

  panel.bus.emit('hypothesis:generated', regeneratedPayload());
  await settle();
  tabButton(panel, 'knowledge').listeners.click();
  await settle();
  tabButton(panel, 'hypothesis').listeners.click();
  await settle();

  assert.strictEqual(
    panel.fetchCalls.filter(function (url) {
      return url === '/api/kb/view?name=hypothesis';
    }).length,
    fetchesBefore,
  );
  assert.strictEqual(panel.hypothesisBuilds.at(-1).root, 'n0');
});

test('a generated hypothesis selection resolves against the new payload', async function () {
  const panel = loadGraphPanel();
  await settle();
  panel.bus.emit('hypothesis:generated', regeneratedPayload());
  await settle();

  let received = null;
  panel.bus.on('entity:select', function (payload) { received = payload; });
  panel.fireHypothesisSelect('n0');

  assert.strictEqual(received.id, 'n0');
  assert.strictEqual(received.detail.label, 'Pour milk into the mug');
});

// %% expand all / collapse all toolbar
test('the expand-all/collapse-all toolbar is only shown on the Hypothesis Graph tab', async function () {
  const panel = loadGraphPanel();
  await settle();

  tabButton(panel, 'hypothesis').listeners.click();
  await settle();
  assert.notStrictEqual(panel.elements['#hypothesis-toolbar'].style.display, 'none');

  tabButton(panel, 'knowledge').listeners.click();
  await settle();
  assert.strictEqual(panel.elements['#hypothesis-toolbar'].style.display, 'none');
});

test('clicking "expand all" calls HypothesisGraph.expandAll', async function () {
  const panel = loadGraphPanel();
  await settle();
  tabButton(panel, 'hypothesis').listeners.click();
  await settle();

  panel.hgExpandAllBtn.listeners.click();

  assert.strictEqual(panel.hypothesisExpandAllCallCount(), 1);
});

test('clicking "collapse all" calls HypothesisGraph.collapseAll', async function () {
  const panel = loadGraphPanel();
  await settle();
  tabButton(panel, 'hypothesis').listeners.click();
  await settle();

  panel.hgCollapseAllBtn.listeners.click();

  assert.strictEqual(panel.hypothesisCollapseAllCallCount(), 1);
});

// %% Ontology Graph tab
function fetchCount(panel, url) {
  return panel.fetchCalls.filter(function (called) { return called === url; }).length;
}

test('opening the Ontology Graph tab builds HypothesisGraph from the ontology view', async function () {
  const panel = loadGraphPanel();
  await settle();

  tabButton(panel, 'ontology').listeners.click();
  await settle();

  assert.strictEqual(fetchCount(panel, '/api/kb/view?name=ontology'), 1);
  assert.strictEqual(panel.hypothesisBuilds.at(-1).crumb, 'ontology graph');
  assert.notStrictEqual(panel.elements['#hypothesis-graph'].style.display, 'none');
  assert.strictEqual(panel.elements['#graph'].style.display, 'none');
  assert.notStrictEqual(panel.elements['#hypothesis-toolbar'].style.display, 'none');
});

test('switching between the two graph tabs rebuilds each from its own cached view', async function () {
  const panel = loadGraphPanel();
  await settle();
  tabButton(panel, 'hypothesis').listeners.click();
  await settle();
  tabButton(panel, 'ontology').listeners.click();
  await settle();

  tabButton(panel, 'hypothesis').listeners.click();
  await settle();

  assert.strictEqual(panel.hypothesisBuilds.at(-1).crumb, 'hypothesis graph');
  assert.strictEqual(fetchCount(panel, '/api/kb/view?name=hypothesis'), 1);
  assert.strictEqual(fetchCount(panel, '/api/kb/view?name=ontology'), 1);
});

test('a selection on the Ontology Graph tab carries the ontology view\'s detail and labels', async function () {
  const panel = loadGraphPanel();
  await settle();
  tabButton(panel, 'ontology').listeners.click();
  await settle();

  let received = null;
  panel.bus.on('entity:select', function (payload) { received = payload; });
  panel.fireHypothesisSelect('h1');

  assert.deepStrictEqual(received.detail, ontologyPayload().details.h1);
  assert.deepStrictEqual(received.relations, [
    { s: 'Root', p: ontologyPayload().edges[0].label, o: 'Child' },
  ]);
});

test('a generated hypothesis refetches the Ontology Graph while that tab is open', async function () {
  const panel = loadGraphPanel();
  await settle();
  tabButton(panel, 'ontology').listeners.click();
  await settle();

  panel.bus.emit('hypothesis:generated', regeneratedPayload());
  await settle();

  assert.strictEqual(fetchCount(panel, '/api/kb/view?name=ontology'), 2);
  assert.strictEqual(panel.hypothesisBuilds.at(-1).crumb, 'ontology graph');
});
