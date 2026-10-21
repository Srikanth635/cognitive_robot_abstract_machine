// Tests the knowledge panel: the three views fetched from the add-on's view route and drawn
// by their own modules once per payload; the click and highlight overlay on the bus; and a
// freshly generated hypothesis replacing what is shown.
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const WEB_ADDON = path.join(
  __dirname, '..', '..', '..', 'cramera_resolution', 'src', 'cramera_resolution', 'web', 'addon',
);

const PANEL_ID = 'resolution-knowledge';
const VIEW_URL = '/api/resolution/view?name=';

// %% a page to mount the panel into
function element() {
  const created = {
    innerHTML: '', textContent: '', style: {}, children: [], listeners: {}, classes: new Set(),
    dataset: {},
    classList: {
      toggle(name, force) {
        const on = force === undefined ? !this.owner.classes.has(name) : force;
        if (on) this.owner.classes.add(name); else this.owner.classes.delete(name);
      },
    },
    addEventListener(event, listener) { this.listeners[event] = listener; },
  };
  created.classList.owner = created;
  return created;
}

function hypothesisPayload(label) {
  return {
    ok: true, crumb: 'hypothesis graph', root: 'root',
    nodes: [
      { id: 'root', label: label || 'Root', kind: 'instruction' },
      { id: 'a', label: 'A', kind: 'root' },
      { id: 'b', label: 'B', kind: 'root' },
    ],
    edges: [
      { from: 'root', to: 'a', label: 'has' },
      { from: 'root', to: 'b', label: 'has' },
    ],
    details: {
      root: { label: label || 'Root', lines: ['line-root'] },
      a: { label: 'A', lines: ['line-a'] },
      b: { label: 'B', lines: ['line-b'] },
    },
    empty: 'No hypothesis knowledge graph available.',
  };
}

function emptyHypothesisPayload() {
  return {
    ok: true, crumb: 'hypothesis graph', root: null, nodes: [], edges: [], details: {},
    empty: 'Nothing generated yet — pick a designator and click Resolve.',
  };
}

function ontologyPayload() {
  return { ok: true, crumb: 'ontology graph', nodes: [{ id: 'c', label: 'C' }], edges: [], details: {} };
}

function designatorPayload() {
  return { ok: true, crumb: 'PouringAction designator', designator: { node: 'description' } };
}

function loadPanel(options) {
  const settings = options || {};
  const buttons = ['hypothesis', 'ontology', 'designator'].map(function (view) {
    const button = element();
    button.dataset.view = view;
    return button;
  });
  const elements = {};
  [
    '#resolution-knowledge-empty', '#resolution-hypothesis-toolbar', '#resolution-legend',
    '#resolution-hypothesis-graph', '#resolution-ontology-graph', '#resolution-designator-view',
    '#resolution-expand-all', '#resolution-collapse-all',
  ].forEach(function (name) { elements[name] = element(); });
  elements['#resolution-knowledge-tabs'] = Object.assign(element(), {
    querySelectorAll() { return buttons; },
  });

  const root = { innerHTML: '', querySelector(selector) { return elements[selector]; } };
  const handlers = {};
  const emitted = [];
  const bus = {
    on(event, listener) { (handlers[event] = handlers[event] || []).push(listener); },
    emit(event, payload) { emitted.push({ event: event, payload: payload }); },
  };
  function publish(event, payload) { (handlers[event] || []).forEach(function (l) { l(payload); }); }

  const requests = [];
  function answer(payload) { return { status: 200, json: async function () { return payload; } }; }
  global.fetch = async function (url) {
    requests.push(url);
    if (settings.unreachable) throw new Error('connection refused');
    if (url === VIEW_URL + 'hypothesis&scene=simple_pouring') {
      return answer(settings.hypothesis || hypothesisPayload());
    }
    if (url === VIEW_URL + 'ontology&scene=simple_pouring') {
      return answer(settings.ontology || ontologyPayload());
    }
    if (url === VIEW_URL + 'designator&scene=simple_pouring') {
      return answer(settings.designator || designatorPayload());
    }
    throw new Error('Unexpected fetch: ' + url);
  };

  const calls = [];
  let selectCallback = null;
  global.HypothesisGraph = {
    build(container, payload) { calls.push({ call: 'build', container: container, payload: payload }); },
    showLegend(legend) { calls.push({ call: 'showLegend', legend: legend }); },
    highlight(ids) { calls.push({ call: 'highlight', ids: ids }); },
    revealAndHighlight(ids) { calls.push({ call: 'revealAndHighlight', ids: ids }); },
    reset() { calls.push({ call: 'reset' }); },
    expandAll() { calls.push({ call: 'expandAll' }); },
    collapseAll() { calls.push({ call: 'collapseAll' }); },
    onSelect(callback) { selectCallback = callback; },
  };
  global.OntologyGraph = { build(container, payload) { calls.push({ call: 'ontology', container: container, payload: payload }); } };
  global.DesignatorView = { build(container, payload) { calls.push({ call: 'designator', container: container, payload: payload }); } };
  global.SceneContext = {
    withScene(url) { return url + (url.indexOf('?') >= 0 ? '&' : '?') + 'scene=simple_pouring'; },
  };
  const layout = settings.layout || { left: ['robot-scene'], right: ['eql', 'graph'] };
  global.window = { CRAMERA_CONFIG: { layout: layout } };

  const defined = {};
  global.Panels = { define(identifier, factory) { defined[identifier] = factory; } };
  new Function(fs.readFileSync(path.join(WEB_ADDON, 'knowledge_panel.js'), 'utf8'))();
  return {
    root, bus, elements, buttons, requests, emitted, calls, defined, layout, publish,
    select(id) { selectCallback(id); },
    mount() { return defined[PANEL_ID](root, bus); },
  };
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

function callsNamed(panel, name) {
  return panel.calls.filter(function (entry) { return entry.call === name; });
}

async function openTab(panel, view) {
  panel.buttons.filter(function (b) { return b.dataset.view === view; })[0].listeners.click();
  await settle();
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
  Object.keys(panel.elements).forEach(function (selector) {
    assert.ok(panel.root.innerHTML.includes('id="' + selector.slice(1) + '"'), selector);
  });
  ['graph', 'legend', 'graph-tabs', 'graph-empty'].forEach(function (shipped) {
    assert.ok(!panel.root.innerHTML.includes('id="' + shipped + '"'), shipped);
  });
});

// %% opening the hypothesis graph
test('the hypothesis graph is fetched once for the scene and drawn into its container', async function () {
  const panel = await mounted();
  assert.deepStrictEqual(panel.requests, [VIEW_URL + 'hypothesis&scene=simple_pouring']);
  const [built] = callsNamed(panel, 'build');
  assert.strictEqual(built.container, panel.elements['#resolution-hypothesis-graph']);
  assert.deepStrictEqual(built.payload, hypothesisPayload());
});

test('the legend is drawn into the panel\'s own legend element', async function () {
  const panel = await mounted();
  assert.strictEqual(callsNamed(panel, 'showLegend')[0].legend, panel.elements['#resolution-legend']);
});

test('revisiting the hypothesis graph neither fetches nor builds again', async function () {
  const panel = await mounted();
  await openTab(panel, 'ontology');
  await openTab(panel, 'hypothesis');
  assert.strictEqual(panel.requests.filter(function (u) { return u.includes('hypothesis'); }).length, 1);
  assert.strictEqual(callsNamed(panel, 'build').length, 1);
});

test('only the container of the open view is shown', async function () {
  const panel = await mounted();
  await openTab(panel, 'ontology');
  assert.strictEqual(panel.elements['#resolution-hypothesis-graph'].style.display, 'none');
  assert.strictEqual(panel.elements['#resolution-ontology-graph'].style.display, '');
  assert.strictEqual(panel.elements['#resolution-designator-view'].style.display, 'none');
});

test('the toolbar and legend belong to the hypothesis graph alone', async function () {
  const panel = await mounted();
  assert.strictEqual(panel.elements['#resolution-hypothesis-toolbar'].style.display, '');
  await openTab(panel, 'designator');
  assert.strictEqual(panel.elements['#resolution-hypothesis-toolbar'].style.display, 'none');
  assert.strictEqual(panel.elements['#resolution-legend'].style.display, 'none');
});

test('the tab of the open view is the active one', async function () {
  const panel = await mounted();
  await openTab(panel, 'designator');
  assert.deepStrictEqual(
    panel.buttons.map(function (b) { return b.classes.has('active'); }),
    [false, false, true],
  );
});

// %% a node clicked
test('a selected node emits entity:select with its detail and relations', async function () {
  const panel = await mounted();
  panel.select('a');
  const [selected] = panel.emitted.filter(function (e) { return e.event === 'entity:select'; });
  assert.deepStrictEqual(selected.payload, {
    id: 'a',
    detail: hypothesisPayload().details.a,
    relations: [{ s: 'Root', p: 'has', o: 'A' }],
  });
});

test('a selected node is spotlit in place with its neighbours, not revealed from the root', async function () {
  const panel = await mounted();
  panel.select('root');
  assert.deepStrictEqual(callsNamed(panel, 'highlight')[0].ids, ['root', 'a', 'b']);
  assert.deepStrictEqual(callsNamed(panel, 'revealAndHighlight'), []);
});

test('a node the payload has no detail of selects nothing', async function () {
  const panel = await mounted();
  panel.select('unknown');
  assert.deepStrictEqual(panel.emitted, []);
});

// %% highlights from elsewhere
test('a query result reveals its nodes from the root while the hypothesis graph is open', async function () {
  const panel = await mounted();
  panel.publish('entity:highlight', { ids: ['b'] });
  assert.deepStrictEqual(callsNamed(panel, 'revealAndHighlight')[0].ids, ['b']);
});

test('a node described elsewhere is spotlit in place with its neighbours', async function () {
  const panel = await mounted();
  panel.publish('entity:highlight', { ids: ['a'], focus: 'a' });
  assert.deepStrictEqual(callsNamed(panel, 'highlight')[0].ids, ['a', 'root']);
  assert.deepStrictEqual(callsNamed(panel, 'revealAndHighlight'), []);
});

test('a highlight naming nothing resets the spotlight', async function () {
  const panel = await mounted();
  panel.publish('entity:highlight', { ids: [] });
  assert.strictEqual(callsNamed(panel, 'reset').length, 1);
});

test('a highlight leaves the hypothesis graph alone while another view is open', async function () {
  const panel = await mounted();
  await openTab(panel, 'ontology');
  panel.publish('entity:highlight', { ids: ['b'] });
  assert.deepStrictEqual(callsNamed(panel, 'revealAndHighlight'), []);
});

// %% nothing generated yet
test('an empty hypothesis view shows its prompt instead of a blank canvas', async function () {
  const panel = await mounted({ hypothesis: emptyHypothesisPayload() });
  assert.strictEqual(panel.elements['#resolution-knowledge-empty'].style.display, '');
  assert.strictEqual(panel.elements['#resolution-knowledge-empty'].textContent, emptyHypothesisPayload().empty);
});

test('generating after an empty view hides the prompt again', async function () {
  const panel = await mounted({ hypothesis: emptyHypothesisPayload() });
  panel.publish('hypothesis:generated', hypothesisPayload());
  await settle();
  assert.strictEqual(panel.elements['#resolution-knowledge-empty'].style.display, 'none');
});

// %% a freshly generated hypothesis
test('a generated hypothesis rebuilds the graph from the payload it brings', async function () {
  const panel = await mounted();
  panel.publish('hypothesis:generated', hypothesisPayload('New'));
  await settle();
  const builds = callsNamed(panel, 'build');
  assert.strictEqual(builds.length, 2);
  assert.strictEqual(builds[1].payload.nodes[0].label, 'New');
});

test('a generated hypothesis is not fetched again', async function () {
  const panel = await mounted();
  panel.publish('hypothesis:generated', hypothesisPayload('New'));
  await settle();
  assert.strictEqual(panel.requests.length, 1);
});

test('a node selected after a generation is read from the new payload', async function () {
  const panel = await mounted();
  const generated = hypothesisPayload('New');
  generated.details.a = { label: 'Changed A', lines: [] };
  panel.publish('hypothesis:generated', generated);
  await settle();
  panel.select('a');
  const [selected] = panel.emitted.filter(function (e) { return e.event === 'entity:select'; });
  assert.strictEqual(selected.payload.detail.label, 'Changed A');
});

test('a payload that is not ok changes nothing', async function () {
  const panel = await mounted();
  panel.publish('hypothesis:generated', { ok: false, error: 'x' });
  await settle();
  assert.strictEqual(callsNamed(panel, 'build').length, 1);
});

// %% the toolbar
test('expand all and collapse all go to the hypothesis graph', async function () {
  const panel = await mounted();
  panel.elements['#resolution-expand-all'].listeners.click();
  panel.elements['#resolution-collapse-all'].listeners.click();
  assert.strictEqual(callsNamed(panel, 'expandAll').length, 1);
  assert.strictEqual(callsNamed(panel, 'collapseAll').length, 1);
});

// %% the ontology graph and the designator
test('the ontology graph is fetched and drawn into its own container', async function () {
  const panel = await mounted();
  await openTab(panel, 'ontology');
  const [drawn] = callsNamed(panel, 'ontology');
  assert.strictEqual(drawn.container, panel.elements['#resolution-ontology-graph']);
  assert.deepStrictEqual(drawn.payload, ontologyPayload());
});

test('switching between the views fetches and draws each once', async function () {
  const panel = await mounted();
  await openTab(panel, 'ontology');
  await openTab(panel, 'hypothesis');
  await openTab(panel, 'ontology');
  assert.strictEqual(panel.requests.filter(function (u) { return u.includes('ontology'); }).length, 1);
  assert.strictEqual(callsNamed(panel, 'ontology').length, 2);
});

test('the designator is fetched and drawn into its own container', async function () {
  const panel = await mounted();
  await openTab(panel, 'designator');
  const [drawn] = callsNamed(panel, 'designator');
  assert.strictEqual(drawn.container, panel.elements['#resolution-designator-view']);
  assert.deepStrictEqual(drawn.payload, designatorPayload());
});

test('the designator, which has no nodes, is not reported as empty', async function () {
  const panel = await mounted();
  await openTab(panel, 'designator');
  assert.strictEqual(panel.elements['#resolution-knowledge-empty'].style.display, 'none');
});

test('a generated hypothesis fetches the ontology again while its view is open', async function () {
  const panel = await mounted();
  await openTab(panel, 'ontology');
  panel.publish('hypothesis:generated', hypothesisPayload('New'));
  await settle();
  assert.strictEqual(panel.requests.filter(function (u) { return u.includes('ontology'); }).length, 2);
});

// %% a view that cannot be shown
test('a view the server refuses says why', async function () {
  const panel = await mounted({ hypothesis: { ok: false, error: 'NoSuchScene: no scene bundle' } });
  assert.strictEqual(panel.elements['#resolution-knowledge-empty'].style.display, '');
  assert.ok(panel.elements['#resolution-knowledge-empty'].textContent.includes('NoSuchScene'));
  assert.deepStrictEqual(callsNamed(panel, 'build'), []);
});

test('a server that cannot be reached says so', async function () {
  const panel = await mounted({ unreachable: true });
  assert.ok(panel.elements['#resolution-knowledge-empty'].textContent.includes('Could not load this view'));
});
