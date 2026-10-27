/* ============================================================================
 * panels/graph/knowledge_views.js — the knowledge a resolution consulted, as tabs of the
 * graph panel, beside Knowledge, Plan, Statechart, Kinematics and Transforms.
 *
 *   Hypothesis Graph   the active hypothesis projected phase first: click a node to expand
 *                      it, double-click to collapse its branch
 *   Ontology Graph     the classes behind it, up to CRAM-AICOR and DUL
 *   Designator         the action designator, as Lisp or JSON
 *
 * The graph panel hands over its root once it is built; the tabs are added to its tab bar,
 * and a tab of ours takes the area of the graph's canvas while it is open. Each view is
 * fetched from /api/resolution/view and drawn by its own module (hypothesis_graph.js,
 * ontology_graph.js, designator_view.js) into a container of its own, once per payload.
 * A new hypothesis replaces what is shown, and marks the Hypothesis Graph tab until it is
 * opened.
 *
 * Bus events:
 *   listens  hypothesis:generated {payload}     the hypothesis a resolution consulted
 *   listens  entity:highlight {ids, focus?}     spotlight the matching nodes
 *   emits    entity:select {id, detail, relations}   a node was clicked
 * ==========================================================================*/
(function () {
  'use strict';

  const VIEW_ROUTE = '/api/resolution/view';
  //: the channel the windows of one viewer announce a generated hypothesis on
  const HYPOTHESIS_CHANNEL = 'cramera-hypothesis';

  const VIEW = Object.freeze({
    hypothesis: 'hypothesis',
    ontology: 'ontology',
    designator: 'designator',
  });

  const TABS_HTML =
    '<button data-resolution-tab="hypothesis" title="the hypothesis the resolution consulted, phase first — click a node to expand it, double-click to collapse its branch">Hypothesis Graph</button>' +
    '<button data-resolution-tab="ontology" title="the ontology behind the hypothesis: the classes it instantiates, what they specialise up to CRAM-AICOR and DUL, and how the hypothesis links them">Ontology Graph</button>' +
    '<button data-resolution-tab="designator" title="the action designator: coraplex\'s action slots followed by the generated hypothesis, as Lisp or JSON">Designator</button>';

  const VIEWS_HTML =
    '<div class="resolution-views" id="resolution-views">' +
    '  <div id="resolution-hypothesis-graph"></div>' +
    '  <div id="resolution-ontology-graph" style="display:none"></div>' +
    '  <div id="resolution-designator-view" style="display:none"></div>' +
    '  <div id="resolution-knowledge-empty" class="knowledge-empty" style="display:none"></div>' +
    '  <div id="resolution-hypothesis-toolbar" class="hypothesis-toolbar">' +
    '    <button id="resolution-expand-all" title="Expand every node in the graph">Expand all</button>' +
    '    <button id="resolution-collapse-all" title="Collapse back to the root node">Collapse all</button>' +
    '  </div>' +
    '  <div class="legend" id="resolution-legend"></div>' +
    '</div>';

  /* Add the views to a graph panel: its root, and the bus it shares with the others. */
  function attach(graphRoot, bus) {
    const tabsEl = graphRoot.querySelector('#graph-tabs');
    const wrapEl = graphRoot.querySelector('.graph-wrap');
    if (!tabsEl || !wrapEl) return;
    tabsEl.querySelector('#gt-live').insertAdjacentHTML('beforebegin', TABS_HTML);
    wrapEl.insertAdjacentHTML('beforeend', VIEWS_HTML);

    const emptyEl = wrapEl.querySelector('#resolution-knowledge-empty');
    const toolbarEl = wrapEl.querySelector('#resolution-hypothesis-toolbar');
    const legendEl = wrapEl.querySelector('#resolution-legend');
    const containers = {
      hypothesis: wrapEl.querySelector('#resolution-hypothesis-graph'),
      ontology: wrapEl.querySelector('#resolution-ontology-graph'),
      designator: wrapEl.querySelector('#resolution-designator-view'),
    };
    const ownTabs = Array.prototype.slice.call(tabsEl.querySelectorAll('[data-resolution-tab]'));
    const cramerasTabs = Array.prototype.slice.call(tabsEl.querySelectorAll('button:not([data-resolution-tab])'));
    wrapEl.querySelector('#resolution-expand-all').addEventListener('click', function () { HypothesisGraph.expandAll(); });
    wrapEl.querySelector('#resolution-collapse-all').addEventListener('click', function () { HypothesisGraph.collapseAll(); });

    let open = null;            // the view being shown, or null while cramera's own tab is
    const base = {};            // view -> the payload as loaded from the server
    let hypothesisDrawn = false;
    const tabOf = {};
    ownTabs.forEach(function (button) { tabOf[button.dataset.resolutionTab] = button; });

    // each view is drawn by its own module, once per loaded payload
    const draw = {
      hypothesis: function (payload) {
        if (hypothesisDrawn) return;
        hypothesisDrawn = true;
        HypothesisGraph.build(containers.hypothesis, payload);
      },
      ontology: function (payload) { OntologyGraph.build(containers.ontology, payload); },
      designator: function (payload) { DesignatorView.build(containers.designator, payload); },
    };

    // ---- showing a view ---------------------------------------------------------------
    function showEmptyState(payload) {
      const empty = !(payload.nodes || []).length && open !== VIEW.designator;
      emptyEl.style.display = empty ? '' : 'none';
      emptyEl.textContent = empty ? (payload.empty || 'Nothing to show in this view.') : '';
    }

    function showError(message) {
      emptyEl.style.display = '';
      emptyEl.textContent = message;
    }

    async function load(name) {
      const response = await fetch(SceneContext.withScene(VIEW_ROUTE + '?name=' + name));
      return response.json();
    }

    async function showView(name) {
      open = name;
      wrapEl.setAttribute('data-resolution-view', name);
      Object.keys(tabOf).forEach(function (view) {
        tabOf[view].classList.toggle('resolution-active', view === name);
      });
      if (name === VIEW.hypothesis) tabOf[name].classList.remove('fresh');
      Object.keys(containers).forEach(function (view) {
        containers[view].style.display = view === name ? '' : 'none';
      });
      toolbarEl.style.display = name === VIEW.hypothesis ? '' : 'none';
      legendEl.style.display = name === VIEW.hypothesis ? '' : 'none';
      emptyEl.style.display = 'none';
      try {
        if (!base[name]) {
          const payload = await load(name);
          // the reader can be on another tab by the time a view arrives
          if (open !== name) return;
          if (!payload.ok) {
            showError(payload.error || 'view unavailable');
            return;
          }
          base[name] = payload;
        }
        showEmptyState(base[name]);
        draw[name](base[name]);
      } catch (error) {
        showError('Could not load this view: ' + ((error && error.message) || error));
        return;
      }
      if (name === VIEW.hypothesis) HypothesisGraph.showLegend(legendEl);
    }

    // a tab of cramera's own gives the canvas back
    function giveBackCanvas() {
      open = null;
      wrapEl.removeAttribute('data-resolution-view');
      Object.keys(tabOf).forEach(function (view) { tabOf[view].classList.remove('resolution-active'); });
    }

    ownTabs.forEach(function (button) {
      button.addEventListener('click', function () { showView(button.dataset.resolutionTab); });
    });
    cramerasTabs.forEach(function (button) { button.addEventListener('click', giveBackCanvas); });

    // ---- a node of the hypothesis graph was clicked -------------------------------------
    function labelOf(payload, id) {
      return (payload.details[id] && payload.details[id].label) || id;
    }

    function neighboursOf(payload, id) {
      return (payload.edges || [])
        .filter(function (edge) { return edge.from === id || edge.to === id; })
        .map(function (edge) { return edge.from === id ? edge.to : edge.from; });
    }

    function select(id) {
      const payload = base[VIEW.hypothesis];
      if (!payload || !payload.details || !payload.details[id]) return;
      const relations = (payload.edges || [])
        .filter(function (edge) { return edge.from === id || edge.to === id; })
        .map(function (edge) {
          return { s: labelOf(payload, edge.from), p: edge.label, o: labelOf(payload, edge.to) };
        });
      bus.emit('entity:select', { id: id, detail: payload.details[id], relations: relations });
      // a clicked node is already on screen: spotlight it and its neighbours in place
      HypothesisGraph.highlight([id].concat(neighboursOf(payload, id)));
    }
    HypothesisGraph.onSelect(select);

    // ---- highlights from elsewhere (an EQL result, a node described in another panel) ----
    bus.on('entity:highlight', function (highlighted) {
      // the other views draw classes and slots, not the fragments an answer names
      if (open !== VIEW.hypothesis) return;
      const payload = base[VIEW.hypothesis];
      const ids = ((highlighted && highlighted.ids) || []).slice();
      if (payload && highlighted && highlighted.focus) {
        neighboursOf(payload, highlighted.focus).forEach(function (id) { ids.push(id); });
      }
      if (!ids.length) { HypothesisGraph.reset(); return; }
      // a click spotlights in place; a query result may be buried in a collapsed branch
      if (highlighted && highlighted.focus) HypothesisGraph.highlight(ids);
      else HypothesisGraph.revealAndHighlight(ids);
    });

    // ---- a freshly generated hypothesis replaces what is shown --------------------------
    function replaceHypothesis(payload) {
      Object.keys(base).forEach(function (view) { delete base[view]; });
      base[VIEW.hypothesis] = payload;
      hypothesisDrawn = false;
      if (open) showView(open);
      else tabOf[VIEW.hypothesis].classList.add('fresh');
    }

    // the window the graph panel was popped out into is not the one Resolve was pressed in
    const channel = typeof BroadcastChannel === 'function'
      ? new BroadcastChannel(HYPOTHESIS_CHANNEL) : null;
    bus.on('hypothesis:generated', function (payload) {
      if (!payload || !payload.ok) return;
      replaceHypothesis(payload);
      if (channel) channel.postMessage(payload);
    });
    if (channel) {
      channel.onmessage = function (message) {
        if (message.data && message.data.ok) replaceHypothesis(message.data);
      };
    }
  }

  window.KnowledgeViews = { attach: attach };
})();
