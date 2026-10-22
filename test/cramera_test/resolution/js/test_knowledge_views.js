// Tests the knowledge views added to the graph panel as tabs: the three views fetched from the
// add-on's view route and drawn by their own modules once per payload; a tab of ours taking
// the canvas and cramera's own tabs giving it back; the click and highlight overlay on the
// bus; and a freshly generated hypothesis replacing what is shown.
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const GRAPH_PANEL = path.join(
  __dirname, '..', '..', '..', '..', 'cramera', 'src', 'cramera', 'web', 'panels', 'graph',
);

const VIEW_URL = '/api/resolution/view?name=';
const OPEN_ATTRIBUTE = 'data-resolution-view';

// %% a graph panel to attach the views to
function element() {
  const created = {
    innerHTML: '', textContent: '', style: {}, listeners: {}, classes: new Set(), dataset: {},
    attributes: {}, inserted: [],
    classList: {
      toggle(name, force) {
        const on = force === undefined ? !this.owner.classes.has(name) : force;
        if (on) this.owner.classes.add(name); else this.owner.classes.delete(name);
      },
      add(name) { this.owner.classes.add(name); },
      remove(name) { this.owner.classes.delete(name); },
      contains(name) { return this.owner.classes.has(name); },
    },
    addEventListener(event, listener) { this.listeners[event] = listener; },
    insertAdjacentHTML(where, html) { this.inserted.push({ where: where, html: html }); },
    setAttribute(name, value) { this.attributes[name] = value; },
    removeAttribute(name) { delete this.attributes[name]; },
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

function loadViews(options) {
  const settings = options || {};
  const ownTabs = ['hypothesis', 'ontology', 'designator'].map(function (view) {
    const button = element();
    button.dataset.resolutionTab = view;
    return button;
  });
  const cramerasTabs = ['knowledge', 'plan'].map(function (view) {
    const button = element();
    button.dataset.view = view;
    return button;
  });
  const wrapElements = {};
  [
    '#resolution-knowledge-empty', '#resolution-hypothesis-toolbar', '#resolution-legend',
    '#resolution-hypothesis-graph', '#resolution-ontology-graph', '#resolution-designator-view',
    '#resolution-expand-all', '#resolution-collapse-all',
  ].forEach(function (name) { wrapElements[name] = element(); });

  const liveBadge = element();
  const tabsEl = Object.assign(element(), {
    querySelector(selector) { return selector === '#gt-live' ? liveBadge : null; },
    querySelectorAll(selector) {
      return selector === '[data-resolution-tab]' ? ownTabs : cramerasTabs;
    },
  });
  const wrapEl = Object.assign(element(), {
    querySelector(selector) { return wrapElements[selector]; },
  });
  const graphRoot = {
    querySelector(selector) { return selector === '#graph-tabs' ? tabsEl : wrapEl; },
  };

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
  const channels = [];
  global.BroadcastChannel = settings.noChannel ? undefined : class {
    constructor(channelName) { this.channelName = channelName; this.posted = []; channels.push(this); }
    postMessage(message) { this.posted.push(message); }
  };
  global.window = {};
  new Function(fs.readFileSync(path.join(GRAPH_PANEL, 'knowledge_views.js'), 'utf8'))();
  global.window.KnowledgeViews.attach(graphRoot, bus);
  return {
    bus, ownTabs, cramerasTabs, wrapEl, tabsEl, liveBadge, wrapElements, requests, emitted,
    calls, publish, graphRoot, channels,
    select(id) { selectCallback(id); },
  };
}

async function settle() {
  for (let i = 0; i < 8; i += 1) {
    await new Promise(function (resolve) { setImmediate(resolve); });
  }
}

function callsNamed(page, name) {
  return page.calls.filter(function (entry) { return entry.call === name; });
}

async function openTab(page, view) {
  page.ownTabs.filter(function (b) { return b.dataset.resolutionTab === view; })[0].listeners.click();
  await settle();
}

async function opened(options) {
  const page = loadViews(options);
  await openTab(page, 'hypothesis');
  return page;
}

// %% attaching to the graph panel
test('the tabs are inserted into the tab bar before the live badge', function () {
  const page = loadViews();
  const [inserted] = page.liveBadge.inserted;
  assert.strictEqual(inserted.where, 'beforebegin');
  ['Hypothesis Graph', 'Ontology Graph', 'Designator'].forEach(function (label) {
    assert.ok(inserted.html.includes('>' + label + '<'), label);
  });
});

test('every view has a tab named for it', function () {
  const page = loadViews();
  const [inserted] = page.liveBadge.inserted;
  ['hypothesis', 'ontology', 'designator'].forEach(function (view) {
    assert.ok(inserted.html.includes('data-resolution-tab="' + view + '"'), view);
  });
});

test('the views are added to the area of the graph panel', function () {
  const page = loadViews();
  const [inserted] = page.wrapEl.inserted;
  assert.strictEqual(inserted.where, 'beforeend');
  Object.keys(page.wrapElements).forEach(function (selector) {
    assert.ok(inserted.html.includes('id="' + selector.slice(1) + '"'), selector);
  });
});

test('no id of the views is one cramera\'s graph panel uses', function () {
  const page = loadViews();
  ['graph', 'legend', 'graph-tabs', 'graph-empty', 'graph-nav'].forEach(function (shipped) {
    assert.ok(!page.wrapEl.inserted[0].html.includes('id="' + shipped + '"'), shipped);
  });
});

// %% opening the hypothesis graph
test('a tab of ours takes the canvas', async function () {
  const page = await opened();
  assert.strictEqual(page.wrapEl.attributes[OPEN_ATTRIBUTE], 'hypothesis');
});

test('the hypothesis graph is fetched once for the scene and drawn into its container', async function () {
  const page = await opened();
  assert.deepStrictEqual(page.requests, [VIEW_URL + 'hypothesis&scene=simple_pouring']);
  const [built] = callsNamed(page, 'build');
  assert.strictEqual(built.container, page.wrapElements['#resolution-hypothesis-graph']);
  assert.deepStrictEqual(built.payload, hypothesisPayload());
});

test('nothing is fetched before a tab is opened', function () {
  assert.deepStrictEqual(loadViews().requests, []);
});

test('the legend is drawn into the views\' own legend element', async function () {
  const page = await opened();
  assert.strictEqual(callsNamed(page, 'showLegend')[0].legend, page.wrapElements['#resolution-legend']);
});

test('revisiting the hypothesis graph neither fetches nor builds again', async function () {
  const page = await opened();
  await openTab(page, 'ontology');
  await openTab(page, 'hypothesis');
  assert.strictEqual(page.requests.filter(function (u) { return u.includes('hypothesis'); }).length, 1);
  assert.strictEqual(callsNamed(page, 'build').length, 1);
});

test('only the container of the open view is shown', async function () {
  const page = await opened();
  await openTab(page, 'ontology');
  assert.strictEqual(page.wrapElements['#resolution-hypothesis-graph'].style.display, 'none');
  assert.strictEqual(page.wrapElements['#resolution-ontology-graph'].style.display, '');
  assert.strictEqual(page.wrapElements['#resolution-designator-view'].style.display, 'none');
});

test('the toolbar and legend belong to the hypothesis graph alone', async function () {
  const page = await opened();
  assert.strictEqual(page.wrapElements['#resolution-hypothesis-toolbar'].style.display, '');
  await openTab(page, 'designator');
  assert.strictEqual(page.wrapElements['#resolution-hypothesis-toolbar'].style.display, 'none');
  assert.strictEqual(page.wrapElements['#resolution-legend'].style.display, 'none');
});

test('the tab of the open view is the lit one', async function () {
  const page = await opened();
  await openTab(page, 'designator');
  assert.deepStrictEqual(
    page.ownTabs.map(function (b) { return b.classes.has('resolution-active'); }),
    [false, false, true],
  );
});

// %% cramera's own tabs
test('a tab of cramera\'s gives the canvas back', async function () {
  const page = await opened();
  page.cramerasTabs[1].listeners.click();
  assert.ok(!(OPEN_ATTRIBUTE in page.wrapEl.attributes));
});

test('no tab of ours is lit once cramera\'s tab is opened', async function () {
  const page = await opened();
  page.cramerasTabs[0].listeners.click();
  assert.deepStrictEqual(
    page.ownTabs.map(function (b) { return b.classes.has('resolution-active'); }),
    [false, false, false],
  );
});

test('a tab of ours takes the canvas again after cramera\'s', async function () {
  const page = await opened();
  page.cramerasTabs[0].listeners.click();
  await openTab(page, 'ontology');
  assert.strictEqual(page.wrapEl.attributes[OPEN_ATTRIBUTE], 'ontology');
});

// %% a node clicked
test('a selected node emits entity:select with its detail and relations', async function () {
  const page = await opened();
  page.select('a');
  const [selected] = page.emitted.filter(function (e) { return e.event === 'entity:select'; });
  assert.deepStrictEqual(selected.payload, {
    id: 'a',
    detail: hypothesisPayload().details.a,
    relations: [{ s: 'Root', p: 'has', o: 'A' }],
  });
});

test('a selected node is spotlit in place with its neighbours, not revealed from the root', async function () {
  const page = await opened();
  page.select('root');
  assert.deepStrictEqual(callsNamed(page, 'highlight')[0].ids, ['root', 'a', 'b']);
  assert.deepStrictEqual(callsNamed(page, 'revealAndHighlight'), []);
});

test('a node the payload has no detail of selects nothing', async function () {
  const page = await opened();
  page.select('unknown');
  assert.deepStrictEqual(page.emitted, []);
});

// %% highlights from elsewhere
test('a query result reveals its nodes from the root while the hypothesis graph is open', async function () {
  const page = await opened();
  page.publish('entity:highlight', { ids: ['b'] });
  assert.deepStrictEqual(callsNamed(page, 'revealAndHighlight')[0].ids, ['b']);
});

test('a node described elsewhere is spotlit in place with its neighbours', async function () {
  const page = await opened();
  page.publish('entity:highlight', { ids: ['a'], focus: 'a' });
  assert.deepStrictEqual(callsNamed(page, 'highlight')[0].ids, ['a', 'root']);
  assert.deepStrictEqual(callsNamed(page, 'revealAndHighlight'), []);
});

test('a highlight naming nothing resets the spotlight', async function () {
  const page = await opened();
  page.publish('entity:highlight', { ids: [] });
  assert.strictEqual(callsNamed(page, 'reset').length, 1);
});

test('a highlight leaves the hypothesis graph alone while another view is open', async function () {
  const page = await opened();
  await openTab(page, 'ontology');
  page.publish('entity:highlight', { ids: ['b'] });
  assert.deepStrictEqual(callsNamed(page, 'revealAndHighlight'), []);
});

test('a highlight leaves the hypothesis graph alone while a tab of cramera\'s is open', async function () {
  const page = await opened();
  page.cramerasTabs[0].listeners.click();
  page.publish('entity:highlight', { ids: ['b'] });
  assert.deepStrictEqual(callsNamed(page, 'revealAndHighlight'), []);
});

// %% nothing generated yet
test('an empty hypothesis view shows its prompt instead of a blank canvas', async function () {
  const page = await opened({ hypothesis: emptyHypothesisPayload() });
  const prompt = page.wrapElements['#resolution-knowledge-empty'];
  assert.strictEqual(prompt.style.display, '');
  assert.strictEqual(prompt.textContent, emptyHypothesisPayload().empty);
});

test('generating after an empty view hides the prompt again', async function () {
  const page = await opened({ hypothesis: emptyHypothesisPayload() });
  page.publish('hypothesis:generated', hypothesisPayload());
  await settle();
  assert.strictEqual(page.wrapElements['#resolution-knowledge-empty'].style.display, 'none');
});

// %% a freshly generated hypothesis
test('a generated hypothesis rebuilds the open graph from the payload it brings', async function () {
  const page = await opened();
  page.publish('hypothesis:generated', hypothesisPayload('New'));
  await settle();
  const builds = callsNamed(page, 'build');
  assert.strictEqual(builds.length, 2);
  assert.strictEqual(builds[1].payload.nodes[0].label, 'New');
});

test('a generated hypothesis is not fetched again', async function () {
  const page = await opened();
  page.publish('hypothesis:generated', hypothesisPayload('New'));
  await settle();
  assert.strictEqual(page.requests.length, 1);
});

test('a generated hypothesis is not drawn while no tab of ours is open', async function () {
  const page = loadViews();
  page.publish('hypothesis:generated', hypothesisPayload('New'));
  await settle();
  assert.deepStrictEqual(callsNamed(page, 'build'), []);
});

test('a generated hypothesis marks the Hypothesis Graph tab while no tab of ours is open', async function () {
  const page = loadViews();
  page.publish('hypothesis:generated', hypothesisPayload('New'));
  assert.strictEqual(page.ownTabs[0].classes.has('fresh'), true);
});

test('opening the Hypothesis Graph tab draws the generated hypothesis without fetching it', async function () {
  const page = loadViews();
  page.publish('hypothesis:generated', hypothesisPayload('New'));
  await openTab(page, 'hypothesis');
  assert.strictEqual(callsNamed(page, 'build')[0].payload.nodes[0].label, 'New');
  assert.deepStrictEqual(page.requests, []);
});

test('opening the Hypothesis Graph tab takes its mark off', async function () {
  const page = loadViews();
  page.publish('hypothesis:generated', hypothesisPayload('New'));
  await openTab(page, 'hypothesis');
  assert.strictEqual(page.ownTabs[0].classes.has('fresh'), false);
});

test('a generated hypothesis while another of our views is open leaves that view open', async function () {
  const page = await opened();
  await openTab(page, 'ontology');
  page.publish('hypothesis:generated', hypothesisPayload('New'));
  await settle();
  assert.strictEqual(page.wrapEl.attributes[OPEN_ATTRIBUTE], 'ontology');
});

test('a node selected after a generation is read from the new payload', async function () {
  const page = await opened();
  const generated = hypothesisPayload('New');
  generated.details.a = { label: 'Changed A', lines: [] };
  page.publish('hypothesis:generated', generated);
  await settle();
  page.select('a');
  const [selected] = page.emitted.filter(function (e) { return e.event === 'entity:select'; });
  assert.strictEqual(selected.payload.detail.label, 'Changed A');
});

test('a payload that is not ok changes nothing', async function () {
  const page = await opened();
  page.publish('hypothesis:generated', { ok: false, error: 'x' });
  await settle();
  assert.strictEqual(callsNamed(page, 'build').length, 1);
});

// %% the toolbar
test('expand all and collapse all go to the hypothesis graph', async function () {
  const page = await opened();
  page.wrapElements['#resolution-expand-all'].listeners.click();
  page.wrapElements['#resolution-collapse-all'].listeners.click();
  assert.strictEqual(callsNamed(page, 'expandAll').length, 1);
  assert.strictEqual(callsNamed(page, 'collapseAll').length, 1);
});

// %% the ontology graph and the designator
test('the ontology graph is fetched and drawn into its own container', async function () {
  const page = loadViews();
  await openTab(page, 'ontology');
  const [drawn] = callsNamed(page, 'ontology');
  assert.strictEqual(drawn.container, page.wrapElements['#resolution-ontology-graph']);
  assert.deepStrictEqual(drawn.payload, ontologyPayload());
});

test('switching between the views fetches each once and draws the ontology graph each time', async function () {
  const page = await opened();
  await openTab(page, 'ontology');
  await openTab(page, 'hypothesis');
  await openTab(page, 'ontology');
  assert.strictEqual(page.requests.filter(function (u) { return u.includes('ontology'); }).length, 1);
  assert.strictEqual(callsNamed(page, 'ontology').length, 2);
});

test('the designator is fetched and drawn into its own container', async function () {
  const page = loadViews();
  await openTab(page, 'designator');
  const [drawn] = callsNamed(page, 'designator');
  assert.strictEqual(drawn.container, page.wrapElements['#resolution-designator-view']);
  assert.deepStrictEqual(drawn.payload, designatorPayload());
});

test('the designator, which has no nodes, is not reported as empty', async function () {
  const page = loadViews();
  await openTab(page, 'designator');
  assert.strictEqual(page.wrapElements['#resolution-knowledge-empty'].style.display, 'none');
});

test('a generated hypothesis fetches the ontology again while its view is open', async function () {
  const page = loadViews();
  await openTab(page, 'ontology');
  page.publish('hypothesis:generated', hypothesisPayload('New'));
  await settle();
  assert.strictEqual(page.requests.filter(function (u) { return u.includes('ontology'); }).length, 2);
});

// %% a view that cannot be shown
test('a view the server refuses says why', async function () {
  const page = await opened({ hypothesis: { ok: false, error: 'NoSuchScene: no scene bundle' } });
  const prompt = page.wrapElements['#resolution-knowledge-empty'];
  assert.strictEqual(prompt.style.display, '');
  assert.ok(prompt.textContent.includes('NoSuchScene'));
  assert.deepStrictEqual(callsNamed(page, 'build'), []);
});

test('a server that cannot be reached says so', async function () {
  const page = await opened({ unreachable: true });
  assert.ok(page.wrapElements['#resolution-knowledge-empty'].textContent.includes('Could not load this view'));
});

// %% hypotheses across windows
// The graph panel pops out into a window of its own, which is not the one Resolve is pressed
// in, so a generated hypothesis is announced to the other windows of the viewer.
test('a generated hypothesis is announced to the other windows on a channel of its own', async function () {
  const page = await opened();
  page.publish('hypothesis:generated', hypothesisPayload('New'));
  const [channel] = page.channels;
  assert.deepStrictEqual(channel.posted, [hypothesisPayload('New')]);
});

test('a payload that is not ok is not announced', async function () {
  const page = await opened();
  page.publish('hypothesis:generated', { ok: false, error: 'x' });
  assert.deepStrictEqual(page.channels[0].posted, []);
});

test('a hypothesis another window announced replaces what is shown, as one generated here does', async function () {
  const page = await opened();
  page.channels[0].onmessage({ data: hypothesisPayload('Elsewhere') });
  await settle();
  const builds = callsNamed(page, 'build');
  assert.strictEqual(builds[1].payload.nodes[0].label, 'Elsewhere');
  assert.strictEqual(page.requests.length, 1);
});

test('a hypothesis another window announced is not announced again', async function () {
  const page = await opened();
  page.channels[0].onmessage({ data: hypothesisPayload('Elsewhere') });
  assert.deepStrictEqual(page.channels[0].posted, []);
});

test('a message that is not an ok payload changes nothing', async function () {
  const page = await opened();
  page.channels[0].onmessage({ data: { ok: false } });
  await settle();
  assert.strictEqual(callsNamed(page, 'build').length, 1);
});

test('a browser without broadcast channels shows what is generated here all the same', async function () {
  const page = await opened({ noChannel: true });
  page.publish('hypothesis:generated', hypothesisPayload('New'));
  await settle();
  assert.strictEqual(callsNamed(page, 'build')[1].payload.nodes[0].label, 'New');
});
