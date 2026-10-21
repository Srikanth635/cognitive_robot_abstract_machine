// Unit tests for addon/hypothesis_graph.js: the Hypothesis Graph view's own vis-network
// instance (repulsion physics kept permanently on), its expand/collapse interaction, and
// its highlight/reset overlay for query results.
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const WEB = path.join(
  __dirname, '..', '..', '..', 'cramera_resolution', 'src', 'cramera_resolution', 'web', 'addon',
);

// matches the source page's own debounce: a click is applied 300ms later so a
// following double-click can still cancel it (see hypothesis_graph.js for why this
// also matters to vis-network's own double-click detection, not just UX feel)
const CLICK_DEBOUNCE_MILLISECONDS = 300;

function wait(milliseconds) {
  return new Promise(function (resolve) { setTimeout(resolve, milliseconds); });
}

// ---- stubs for the vis-network classes and the page --------------------------------
class DataSet {
  constructor(items) {
    this.items = {};
    (items || []).forEach((item) => { this.items[item.id] = item; });
  }
  clear() { this.items = {}; }
  add(list) { (Array.isArray(list) ? list : [list]).forEach((item) => { this.items[item.id] = item; }); }
  update(list) {
    (Array.isArray(list) ? list : [list]).forEach((patch) => {
      this.items[patch.id] = Object.assign({}, this.items[patch.id], patch);
    });
  }
  get(id) { return this.items[id]; }
  getIds() { return Object.keys(this.items); }
}

let lastOptions = null;
let lastData = null;
let lastNetwork = null;

class Network {
  constructor(container, data, options) {
    lastData = data;
    lastOptions = options;
    lastNetwork = this;
    this.data = data;
    this.handlers = {};
  }
  on(event, cb) { (this.handlers[event] = this.handlers[event] || []).push(cb); }
  fire(event, payload) { (this.handlers[event] || []).forEach((cb) => cb(payload)); }
  fit() {}
  redraw() {}
  // mirrors the real API: whatever x/y a node currently carries in the DataSet is
  // its "settled" position, same as vis-network tracking it internally
  getPositions(ids) {
    const result = {};
    (ids || this.data.nodes.getIds()).forEach((id) => {
      const item = this.data.nodes.get(id);
      result[id] = { x: (item && item.x) || 0, y: (item && item.y) || 0 };
    });
    return result;
  }
}

function element() {
  return { className: '', innerHTML: '', children: [], appendChild(child) { this.children.push(child); } };
}

function loadHypothesisGraph() {
  const graphEl = element();
  const legendEl = element();
  global.document = { createElement() { return element(); } };
  global.window = {};
  global.vis = { DataSet, Network };
  new Function(fs.readFileSync(path.join(WEB, 'hypothesis_graph.js'), 'utf8'))();
  return { HypothesisGraph: global.window.HypothesisGraph, legendEl, graphEl };
}

// root -> a, root -> b, a -> shared, a -> a1, b -> shared, shared -> leaf
// ('shared' has two parents -- exercises the same DAG collapse semantics validated
// for the client-side algorithm this ports)
function fixturePayload() {
  return {
    ok: true,
    crumb: 'hypothesis graph',
    root: 'root',
    nodes: [
      { id: 'root', label: 'Root', kind: 'instruction' },
      { id: 'a', label: 'A', kind: 'root' },
      { id: 'b', label: 'B', kind: 'root' },
      { id: 'shared', label: 'Shared', kind: 'group' },
      { id: 'a1', label: 'A1', kind: 'fragment' },
      { id: 'leaf', label: 'Leaf', kind: 'fragment' },
    ],
    edges: [
      { from: 'root', to: 'a', label: 'has' },
      { from: 'root', to: 'b', label: 'has' },
      { from: 'a', to: 'shared', label: 'has' },
      { from: 'a', to: 'a1', label: 'has' },
      { from: 'b', to: 'shared', label: 'has' },
      { from: 'shared', to: 'leaf', label: 'has' },
    ],
    details: {
      root: { label: 'Root', lines: ['line-root'] },
      a: { label: 'A', lines: ['line-a'] },
      b: { label: 'B', lines: ['line-b'] },
      shared: { label: 'Shared', lines: ['line-shared'] },
      a1: { label: 'A1', lines: ['line-a1'] },
      leaf: { label: 'Leaf', lines: ['line-leaf'] },
    },
  };
}

function visibleIds() {
  return Object.keys(lastData.nodes.items).sort();
}

async function clickAndSettle(id) {
  lastNetwork.fire('click', { nodes: [id] });
  await wait(CLICK_DEBOUNCE_MILLISECONDS + 50);
}

// %% network configuration
test('the network keeps repulsion physics permanently on, damped well above the vis-network default', function () {
  const { HypothesisGraph, graphEl } = loadHypothesisGraph();
  HypothesisGraph.build(graphEl, fixturePayload());

  assert.strictEqual(lastOptions.physics.enabled, true);
  assert.strictEqual(lastOptions.physics.solver, 'repulsion');
  assert.strictEqual(lastOptions.physics.repulsion.nodeDistance, 220);
  // raised from vis-network's springy 0.09 default -- otherwise revealing a couple
  // of new nodes flings the whole already-visible graph into an oscillating "blast"
  assert.ok(lastOptions.physics.repulsion.damping > 0.09);
  assert.strictEqual(lastOptions.nodes.shape, 'dot');
});

// %% initial state
test('build shows only the root node initially', function () {
  const { HypothesisGraph, graphEl } = loadHypothesisGraph();
  HypothesisGraph.build(graphEl, fixturePayload());

  assert.deepStrictEqual(visibleIds(), ['root']);
});

// %% click is debounced -- this is what lets a following double-click cancel it
// (and, in the real browser, is what stops vis-network's own double-click pairing
// from breaking: mutating the DataSet between the two physical clicks of a double-
// click gesture makes vis-network treat them as two unrelated single clicks instead)
test('a click does not expand immediately -- it applies after a short debounce', async function () {
  const { HypothesisGraph, graphEl } = loadHypothesisGraph();
  HypothesisGraph.build(graphEl, fixturePayload());

  lastNetwork.fire('click', { nodes: ['root'] });
  assert.deepStrictEqual(visibleIds(), ['root']);

  await wait(CLICK_DEBOUNCE_MILLISECONDS + 50);
  assert.deepStrictEqual(visibleIds(), ['a', 'b', 'root']);
});

test('a double-click cancels its own leading click -- the expand never applies', async function () {
  const { HypothesisGraph, graphEl } = loadHypothesisGraph();
  HypothesisGraph.build(graphEl, fixturePayload());

  // a real double-click gesture fires click then doubleClick back to back
  lastNetwork.fire('click', { nodes: ['root'] });
  lastNetwork.fire('doubleClick', { nodes: ['root'] });
  await wait(CLICK_DEBOUNCE_MILLISECONDS + 50);

  assert.deepStrictEqual(visibleIds(), ['root']);
});

// %% expand on click
test('clicking a node expands its direct children and fires the select callback', async function () {
  const { HypothesisGraph, graphEl } = loadHypothesisGraph();
  HypothesisGraph.build(graphEl, fixturePayload());
  let selected = null;
  HypothesisGraph.onSelect(function (id) { selected = id; });

  await clickAndSettle('root');

  assert.deepStrictEqual(visibleIds(), ['a', 'b', 'root']);
  assert.strictEqual(selected, 'root');
});

test('clicking an already-expanded node does not duplicate its children', async function () {
  const { HypothesisGraph, graphEl } = loadHypothesisGraph();
  HypothesisGraph.build(graphEl, fixturePayload());

  await clickAndSettle('root');
  await clickAndSettle('root');

  assert.deepStrictEqual(visibleIds(), ['a', 'b', 'root']);
});

// %% node positioning -- new nodes grow from their parent, not the canvas center
test('a newly-revealed node spawns near the position of the node that revealed it', async function () {
  const { HypothesisGraph, graphEl } = loadHypothesisGraph();
  HypothesisGraph.build(graphEl, fixturePayload());
  // simulate physics having settled 'root' at a spot away from the origin
  lastData.nodes.update([{ id: 'root', x: 500, y: -300 }]);

  await clickAndSettle('root');

  const a = lastData.nodes.get('a');
  const b = lastData.nodes.get('b');
  // spawned within the jitter radius of root's position, not at (0, 0) / the
  // network's default spawn point
  assert.ok(Math.abs(a.x - 500) <= 10 && Math.abs(a.y - -300) <= 10);
  assert.ok(Math.abs(b.x - 500) <= 10 && Math.abs(b.y - -300) <= 10);
});

test('a still-visible node keeps its exact settled position across a re-render', async function () {
  const { HypothesisGraph, graphEl } = loadHypothesisGraph();
  HypothesisGraph.build(graphEl, fixturePayload());
  await clickAndSettle('root');
  // simulate physics having moved 'a' since it was first revealed
  lastData.nodes.update([{ id: 'a', x: 42, y: 99 }]);

  // re-clicking 'root' re-renders the same visible set -- 'a' must not be
  // reset to a fresh spawn position, only genuinely new nodes should move
  await clickAndSettle('root');

  const a = lastData.nodes.get('a');
  assert.strictEqual(a.x, 42);
  assert.strictEqual(a.y, 99);
});

// %% single focused branch -- expanding a node collapses whatever else was open
test('clicking a sibling collapses the previously-expanded branch', async function () {
  const { HypothesisGraph, graphEl } = loadHypothesisGraph();
  HypothesisGraph.build(graphEl, fixturePayload());

  await clickAndSettle('root');
  await clickAndSettle('a');
  assert.deepStrictEqual(visibleIds(), ['a', 'a1', 'b', 'root', 'shared']);

  await clickAndSettle('b');

  // 'a' and 'b' are still both visible as root's direct children, but only 'b' is
  // itself expanded now -- 'a1' (only reachable through 'a's own expansion)
  // disappears, 'shared' survives because 'b' also points at it
  assert.deepStrictEqual(visibleIds(), ['a', 'b', 'root', 'shared']);
});

test('clicking a descendant keeps its whole ancestor chain expanded', async function () {
  const { HypothesisGraph, graphEl } = loadHypothesisGraph();
  HypothesisGraph.build(graphEl, fixturePayload());

  await clickAndSettle('root');
  await clickAndSettle('a');
  await clickAndSettle('a1');

  // root -> a -> a1 all stay open since a1 is a descendant of a, not a sibling
  assert.deepStrictEqual(visibleIds(), ['a', 'a1', 'b', 'root', 'shared']);
});

test('clicking a node with several parents stays on the branch it was reached through', async function () {
  const { HypothesisGraph, graphEl } = loadHypothesisGraph();
  HypothesisGraph.build(graphEl, fixturePayload());

  await clickAndSettle('root');
  await clickAndSettle('b');
  await clickAndSettle('shared');

  // 'shared' is also reachable through 'a' (edges list 'a'->'shared' before
  // 'b'->'shared'). Picking an ancestor by raw edge order rather than by which
  // branch is actually open would silently switch to 'a' here and reveal 'a1'
  // (a's own, unrelated child) even though the click came through 'b'
  assert.deepStrictEqual(visibleIds(), ['a', 'b', 'leaf', 'root', 'shared'].sort());
});

// %% collapse on double-click
test('double-clicking an already-expanded node collapses back to its parent\'s view', async function () {
  const { HypothesisGraph, graphEl } = loadHypothesisGraph();
  HypothesisGraph.build(graphEl, fixturePayload());

  await clickAndSettle('root');
  await clickAndSettle('a');
  // the double-click's own leading click on 'a' is a no-op (already expanded);
  // doubleClick itself is not debounced, so this collapses immediately
  lastNetwork.fire('click', { nodes: ['a'] });
  lastNetwork.fire('doubleClick', { nodes: ['a'] });

  // 'a' and 'b' remain visible as root's direct children, but 'a' is no longer
  // itself expanded, so 'a1' and 'shared' (only reachable through 'a') disappear
  assert.deepStrictEqual(visibleIds(), ['a', 'b', 'root']);
});

// %% expand all / collapse all
test('expandAll reveals every node in the payload', function () {
  const { HypothesisGraph, graphEl } = loadHypothesisGraph();
  HypothesisGraph.build(graphEl, fixturePayload());

  HypothesisGraph.expandAll();

  assert.deepStrictEqual(
    visibleIds(),
    ['a', 'a1', 'b', 'leaf', 'root', 'shared'].sort(),
  );
});

test('collapseAll returns to the initial root-only view', function () {
  const { HypothesisGraph, graphEl } = loadHypothesisGraph();
  HypothesisGraph.build(graphEl, fixturePayload());
  HypothesisGraph.expandAll();

  HypothesisGraph.collapseAll();

  assert.deepStrictEqual(visibleIds(), ['root']);
  assert.deepStrictEqual(lastData.edges.getIds(), []);
});

// %% highlight overlay (driven by EQL query results)
test('highlight dims every node except the given ids', async function () {
  const { HypothesisGraph, graphEl } = loadHypothesisGraph();
  HypothesisGraph.build(graphEl, fixturePayload());
  await clickAndSettle('root');

  HypothesisGraph.highlight(['a']);

  assert.strictEqual(lastData.nodes.get('a').opacity, 1);
  assert.ok(lastData.nodes.get('root').opacity < 1);
  assert.ok(lastData.nodes.get('b').opacity < 1);
});

test('reset restores full opacity to every visible node', async function () {
  const { HypothesisGraph, graphEl } = loadHypothesisGraph();
  HypothesisGraph.build(graphEl, fixturePayload());
  await clickAndSettle('root');
  HypothesisGraph.highlight(['a']);

  HypothesisGraph.reset();

  assert.strictEqual(lastData.nodes.get('root').opacity, 1);
  assert.strictEqual(lastData.nodes.get('a').opacity, 1);
  assert.strictEqual(lastData.nodes.get('b').opacity, 1);
});

// %% reveal path to EQL answers
// A query's answers can be scattered deep in the tree; expandAll (which shows every
// node at once) becomes unreadable at real scale. revealAndHighlight instead expands
// only each answer's ancestors -- the node itself becomes visible as a child of its
// now-expanded parent, but its own subtree stays collapsed.
test('revealAndHighlight expands the path from root to a target node, leaving its own children collapsed', function () {
  const { HypothesisGraph, graphEl } = loadHypothesisGraph();
  HypothesisGraph.build(graphEl, fixturePayload());

  HypothesisGraph.revealAndHighlight(['shared']);

  // root -> a is the (first-discovered) path to 'shared'; expanding 'a' also
  // surfaces its other direct child 'a1' -- expanding always reveals every direct
  // child, not just the one on the target's path. 'leaf' (shared's own child)
  // stays hidden, which is the point of the feature.
  assert.deepStrictEqual(visibleIds(), ['a', 'a1', 'b', 'root', 'shared'].sort());
});

test('revealAndHighlight on the root node does not expand anything beyond the root', function () {
  const { HypothesisGraph, graphEl } = loadHypothesisGraph();
  HypothesisGraph.build(graphEl, fixturePayload());

  HypothesisGraph.revealAndHighlight(['root']);

  assert.deepStrictEqual(visibleIds(), ['root']);
});

test('revealAndHighlight unions the paths for multiple targets across different branches', function () {
  const { HypothesisGraph, graphEl } = loadHypothesisGraph();
  HypothesisGraph.build(graphEl, fixturePayload());

  HypothesisGraph.revealAndHighlight(['a1', 'leaf']);

  assert.deepStrictEqual(visibleIds(), ['a', 'a1', 'b', 'leaf', 'root', 'shared'].sort());
});

test('revealAndHighlight replaces previously-expanded branches unrelated to the new targets', async function () {
  const { HypothesisGraph, graphEl } = loadHypothesisGraph();
  HypothesisGraph.build(graphEl, fixturePayload());
  await clickAndSettle('root');
  await clickAndSettle('b');
  await clickAndSettle('shared');
  assert.ok(visibleIds().includes('leaf'));

  HypothesisGraph.revealAndHighlight(['a1']);

  assert.deepStrictEqual(visibleIds(), ['a', 'a1', 'b', 'root', 'shared'].sort());
});

test('revealAndHighlight spotlights the given ids like highlight does', function () {
  const { HypothesisGraph, graphEl } = loadHypothesisGraph();
  HypothesisGraph.build(graphEl, fixturePayload());

  HypothesisGraph.revealAndHighlight(['a1']);

  assert.strictEqual(lastData.nodes.get('a1').opacity, 1);
  assert.ok(lastData.nodes.get('root').opacity < 1);
});

test('revealAndHighlight ignores ids that are not real nodes when computing paths', function () {
  const { HypothesisGraph, graphEl } = loadHypothesisGraph();
  HypothesisGraph.build(graphEl, fixturePayload());

  HypothesisGraph.revealAndHighlight(['does-not-exist']);

  assert.deepStrictEqual(visibleIds(), ['root']);
});

// %% legend
test('showLegend populates the legend element with the four node kinds', function () {
  const { HypothesisGraph, legendEl, graphEl } = loadHypothesisGraph();
  HypothesisGraph.build(graphEl, fixturePayload());

  HypothesisGraph.showLegend(legendEl);

  assert.strictEqual(legendEl.children.length, 4);
});
