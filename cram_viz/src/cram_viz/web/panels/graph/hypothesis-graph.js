/* ============================================================================
 * panels/graph/hypothesis-graph.js — the Hypothesis Graph tab.
 *
 * llm_reasoner's cereal-pouring knowledge graph, rendered with its own dedicated
 * vis-network instance instead of graph.js's shared window.Graph wrapper. graph.js
 * deliberately freezes physics once a view's layout settles (so re-opening a static
 * tab is instant); this tab instead keeps physics running forever (repulsion solver)
 * so the layout stays alive as you expand it — but with damping raised well above
 * vis-network's springy 0.09 default, so injecting a couple of new nodes settles them
 * into place instead of flinging the whole already-visible graph into an oscillating
 * "blast". This module owns its own vis.Network rather than fighting graph.js's
 * freeze-on-settle design to get that continuous effect at all.
 *
 * panel.js overlays cram_viz's own behaviour on top through the small API this module
 * exposes: build() once per tab load, onSelect() to hook the EQL answer panel into
 * node clicks, highlight()/reset() to spotlight nodes already on screen,
 * revealAndHighlight() to additionally reveal the root-to-node path for each id (used
 * for EQL query results, where a hit can be buried deep in a collapsed branch),
 * resize() for the splitter. Expand-on-click and collapse-branch-on-double-click are
 * handled entirely inside this module. Unlike the standalone exported page this tab
 * otherwise reproduces, a click here expands only the clicked node's own path from
 * the root plus its immediate children, collapsing any other branch — the source
 * page instead leaves every node ever clicked expanded forever, which on a graph
 * this size (many phases, each with several categories of their own) grows into a
 * large, cluttered mass that then visibly reflows on every subsequent click.
 * ==========================================================================*/
(function (root) {
  'use strict';

  const NODE_COLOR = {
    instruction: { background: '#e76f51', label: 'Pouring instruction' },
    root: { background: '#e9c46a', label: 'Knowledge root' },
    group: { background: '#457b9d', label: 'Knowledge group' },
    fragment: { background: '#2a9d8f', label: 'Knowledge detail' },
    unresolved: { background: '#e63946', label: 'Not stated in instruction' },
  };
  const NODE_BORDER_COLOR = '#888888';
  const EDGE_COLOR = '#9aa0a6';

  // A click is applied after this delay instead of immediately, so a following
  // double-click can still cancel it — ported from the source page. This isn't just
  // UX polish: mutating the DataSet (via expand -> renderGraph) synchronously between
  // the two physical clicks of a double-click gesture stops vis-network from pairing
  // them into a 'doubleClick' event at all, so without the debounce double-click
  // stops working entirely, not just "feels laggy".
  const CLICK_DEBOUNCE_MILLISECONDS = 300;

  const NETWORK_OPTIONS = {
    nodes: {
      shape: 'dot',
      size: 13,
      font: { color: '#202124', face: 'Arial', size: 12, strokeWidth: 0 },
    },
    edges: {
      color: EDGE_COLOR,
      arrows: 'to',
      font: { background: '#ffffff', color: '#5f6368', face: 'Arial', size: 10, strokeWidth: 0 },
    },
    physics: {
      enabled: true,
      solver: 'repulsion',
      repulsion: { nodeDistance: 220, damping: 0.5 },
    },
  };

  let network = null;
  let visibleNodes = null;
  let visibleEdges = null;
  let fullPayload = { root: null, nodes: [], edges: [], details: {} };
  let outgoing = {};
  let incoming = {};
  let expandedIdentifiers = new Set();
  let allNodeIds = [];
  let selectCb = function () {};
  let pendingClickTimeout = null;

  function visNode(node, position) {
    const color = node.unresolved ? NODE_COLOR.unresolved : (NODE_COLOR[node.kind] || NODE_COLOR.fragment);
    const rendered = {
      id: node.id,
      label: node.label,
      title: node.label,
      color: { background: color.background, border: NODE_BORDER_COLOR },
    };
    if (position) {
      rendered.x = position.x;
      rendered.y = position.y;
    }
    return rendered;
  }

  /**
   * Where a newly-revealed node should start, so it grows outward from the node
   * that revealed it instead of flying in from vis-network's default spawn point
   * (the canvas center) — which is what makes expanding a node on one side of an
   * already-large graph look like a "blast" radiating from the middle of the
   * screen rather than a local change next to where you actually clicked.
   */
  function spawnPositionFor(identifier, knownPositions) {
    const parents = incoming[identifier] || [];
    const revealingParent = parents.find(
      (parent) => expandedIdentifiers.has(parent) && knownPositions[parent],
    );
    if (!revealingParent) return null;
    const parentPosition = knownPositions[revealingParent];
    // a small jitter so several siblings revealed at once don't spawn stacked on
    // the exact same point, which the physics solver treats as a zero-distance
    // (singular) case
    return {
      x: parentPosition.x + (Math.random() - 0.5) * 20,
      y: parentPosition.y + (Math.random() - 0.5) * 20,
    };
  }

  function descendantsOf(identifier) {
    const descendants = new Set();
    const frontier = (outgoing[identifier] || []).slice();
    while (frontier.length > 0) {
      const current = frontier.pop();
      if (descendants.has(current)) continue;
      descendants.add(current);
      frontier.push(...(outgoing[current] || []));
    }
    return descendants;
  }

  /**
   * Ids currently on screen: the root, every expanded node, and their direct
   * children — a node stays visible as long as ANY still-expanded node points at
   * it, even one whose own subtree was independently collapsed elsewhere.
   */
  function visibleIdentifiers() {
    const identifiers = new Set([fullPayload.root]);
    expandedIdentifiers.forEach((identifier) => {
      identifiers.add(identifier);
      (outgoing[identifier] || []).forEach((child) => identifiers.add(child));
    });
    return identifiers;
  }

  function renderGraph() {
    const identifiers = visibleIdentifiers();
    // visibleNodes.clear() below drops the network's own memory of every node's
    // settled position, so without re-supplying x/y here, EVERY click -- not just
    // ones that reveal something new -- would re-scatter the entire already-visible
    // graph from vis-network's default spawn point and let physics re-settle it
    // from scratch. Carrying still-visible nodes' exact positions forward keeps
    // them motionless; only genuinely new nodes need a starting point, and that
    // point is their revealing parent, not the canvas center.
    const previouslyVisible = new Set(allNodeIds);
    const knownPositions = previouslyVisible.size
      ? network.getPositions([...previouslyVisible])
      : {};
    const nodes = fullPayload.nodes
      .filter((n) => identifiers.has(n.id))
      .map((n) => visNode(n, knownPositions[n.id] || spawnPositionFor(n.id, knownPositions)));
    const edges = fullPayload.edges
      .filter((e) => expandedIdentifiers.has(e.from) && identifiers.has(e.to))
      .map((e) => ({ from: e.from, to: e.to, title: e.label }));
    allNodeIds = nodes.map((n) => n.id);
    visibleNodes.clear();
    visibleNodes.add(nodes);
    visibleEdges.clear();
    visibleEdges.add(edges);
  }

  /**
   * Expand exactly one node at a time: the clicked node plus the root-to-it path,
   * discarding whatever branch was previously open. Without this, every node ever
   * clicked stays expanded forever (nothing auto-collapses), so a large exploration
   * session leaves many unrelated branches open at once — then physics has to
   * reflow all of them on every subsequent click, which reads as random,
   * unconnected nodes "expanding" on their own.
   */
  function expand(identifier) {
    const path = new Set(ancestorsOf(identifier));
    path.add(identifier);
    expandedIdentifiers = path;
    renderGraph();
  }

  function collapseBranch(identifier) {
    expandedIdentifiers.delete(identifier);
    descendantsOf(identifier).forEach((descendant) => {
      expandedIdentifiers.delete(descendant);
    });
    renderGraph();
  }

  /** Reveal every node in the payload at once. */
  function expandAll() {
    expandedIdentifiers = new Set(fullPayload.nodes.map((n) => n.id));
    renderGraph();
  }

  /** Collapse back to showing only the root node. */
  function collapseAll() {
    expandedIdentifiers = new Set();
    renderGraph();
  }

  /** One path of ancestor ids from the root down to (excluding) the given node. */
  /**
   * One path of ancestor ids from the root down to (excluding) the given node.
   *
   * A node with several parents (a role, event or condition shared by more than
   * one phase) picks whichever parent is currently expanded, so clicking a node
   * you reached through phase A keeps you inside phase A's branch. Only when
   * none of its parents are currently expanded (nothing is open yet, or this id
   * is reached from an EQL result rather than a click) does it fall back to
   * whichever parent this node happened to be reached through first while the
   * payload's edges were built — without this, the choice would depend on
   * *global* edge order (which phase's group referenced it first when the graph
   * was built), not on which branch the user actually navigated through, so a
   * shared node could jump the view to a phase that was never clicked at all.
   */
  function ancestorsOf(identifier) {
    const chain = [];
    const seen = new Set();
    let current = identifier;
    while (true) {
      const parents = incoming[current];
      if (!parents || !parents.length) break;
      const parent = parents.find((candidate) => expandedIdentifiers.has(candidate)) || parents[0];
      if (seen.has(parent)) break;
      seen.add(parent);
      chain.push(parent);
      current = parent;
    }
    return chain;
  }

  /**
   * Collapse everything, reveal only the root-to-node path for each given id
   * (each target's own children stay collapsed), then spotlight them — used for EQL
   * query results, where expandAll's "show every descendant" makes a real answer set
   * unreadable.
   */
  function revealAndHighlight(ids) {
    expandedIdentifiers = new Set();
    ids.forEach((identifier) => {
      ancestorsOf(identifier).forEach((ancestor) => expandedIdentifiers.add(ancestor));
    });
    renderGraph();
    highlight(ids);
  }

  /** (Re)build the graph from a full node/edge/details payload — called once per tab load. */
  function build(payload) {
    fullPayload = payload;
    outgoing = {};
    incoming = {};
    (payload.edges || []).forEach((edge) => {
      if (!outgoing[edge.from]) outgoing[edge.from] = [];
      outgoing[edge.from].push(edge.to);
      if (!incoming[edge.to]) incoming[edge.to] = [];
      incoming[edge.to].push(edge.from);
    });
    expandedIdentifiers = new Set();

    if (!network) {
      visibleNodes = new vis.DataSet([]);
      visibleEdges = new vis.DataSet([]);
      network = new vis.Network(
        document.getElementById('hypothesis-graph'),
        { nodes: visibleNodes, edges: visibleEdges },
        NETWORK_OPTIONS,
      );
      network.on('click', function (params) {
        if (!params.nodes.length) return;
        clearTimeout(pendingClickTimeout);
        const identifier = params.nodes[0];
        pendingClickTimeout = setTimeout(function () {
          expand(identifier);
          selectCb(identifier);
        }, CLICK_DEBOUNCE_MILLISECONDS);
      });
      network.on('doubleClick', function (params) {
        clearTimeout(pendingClickTimeout);
        if (params.nodes.length) collapseBranch(params.nodes[0]);
      });
    }
    renderGraph();
  }

  /** Dim everything, then spotlight the given ids (an EQL query's highlight result). */
  function highlight(ids) {
    if (!visibleNodes) return;
    const set = {};
    ids.forEach((id) => { set[id] = 1; });
    visibleNodes.update(
      allNodeIds.map((id) => ({ id: id, opacity: set[id] ? 1 : 0.16 })),
    );
  }

  function reset() {
    if (!visibleNodes) return;
    visibleNodes.update(allNodeIds.map((id) => ({ id: id, opacity: 1 })));
  }

  function resize() {
    if (!network) return;
    network.redraw();
    network.fit({ animation: false });
  }

  /** Populate the shared #legend element with this tab's fixed node-kind palette. */
  function showLegend() {
    const legendEl = document.getElementById('legend');
    if (!legendEl) return;
    legendEl.innerHTML = '';
    Object.keys(NODE_COLOR).forEach((kind) => {
      const style = NODE_COLOR[kind];
      const row = document.createElement('div');
      row.className = 'li';
      row.innerHTML =
        '<span class="dot" style="background:' + style.background + '"></span>' + style.label;
      legendEl.appendChild(row);
    });
  }

  root.HypothesisGraph = {
    build: build,
    highlight: highlight,
    revealAndHighlight: revealAndHighlight,
    reset: reset,
    resize: resize,
    showLegend: showLegend,
    expandAll: expandAll,
    collapseAll: collapseAll,
    onSelect: function (cb) { selectCb = cb; },
  };
})(window);
