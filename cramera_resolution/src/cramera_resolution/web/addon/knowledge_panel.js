/* ============================================================================
 * addon/knowledge_panel.js — the knowledge a resolution consulted, in three views.
 *
 *   Hypothesis Graph   the active hypothesis projected phase first: click a node to expand
 *                      it, double-click to collapse its branch
 *   Ontology Graph     the classes behind it, up to CRAM-AICOR and DUL
 *   Designator         the action designator, as Lisp or JSON
 *
 * Each view is fetched from /api/resolution/view and drawn by its own module
 * (hypothesis_graph.js, ontology_graph.js, designator_view.js) into a container of this
 * panel, once per payload. A new hypothesis replaces what is shown.
 *
 * Bus events:
 *   listens  hypothesis:generated {payload}     the hypothesis a resolution consulted
 *   listens  entity:highlight {ids, focus?}     spotlight the matching nodes
 *   emits    entity:select {id, detail, relations}   a node was clicked
 * ==========================================================================*/
(function () {
  'use strict';

  const PANEL_ID = 'resolution-knowledge';
  const VIEW_ROUTE = '/api/resolution/view';

  const VIEW = Object.freeze({
    hypothesis: 'hypothesis',
    ontology: 'ontology',
    designator: 'designator',
  });

  Panels.define(PANEL_ID, function (root, bus) {
    root.innerHTML =
      '<div class="panel-head">' +
      '  <h2>Knowledge · what a resolution consulted</h2>' +
      '</div>' +
      '<div class="resolution-knowledge-wrap">' +
      '  <div class="knowledge-tabs" id="resolution-knowledge-tabs">' +
      '    <button data-view="hypothesis" class="active" title="the hypothesis the resolution consulted, phase first — click a node to expand it, double-click to collapse its branch">Hypothesis Graph</button>' +
      '    <button data-view="ontology" title="the ontology behind the hypothesis: the classes it instantiates, what they specialise up to CRAM-AICOR and DUL, and how the hypothesis links them">Ontology Graph</button>' +
      '    <button data-view="designator" title="the action designator: coraplex\'s action slots followed by the generated hypothesis, as Lisp or JSON">Designator</button>' +
      '  </div>' +
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

    const tabsEl = root.querySelector('#resolution-knowledge-tabs');
    const emptyEl = root.querySelector('#resolution-knowledge-empty');
    const toolbarEl = root.querySelector('#resolution-hypothesis-toolbar');
    const legendEl = root.querySelector('#resolution-legend');
    const containers = {
      hypothesis: root.querySelector('#resolution-hypothesis-graph'),
      ontology: root.querySelector('#resolution-ontology-graph'),
      designator: root.querySelector('#resolution-designator-view'),
    };
    root.querySelector('#resolution-expand-all').addEventListener('click', function () { HypothesisGraph.expandAll(); });
    root.querySelector('#resolution-collapse-all').addEventListener('click', function () { HypothesisGraph.collapseAll(); });

    let tab = VIEW.hypothesis;
    const base = {};            // view -> the payload as loaded from the server
    let hypothesisDrawn = false;

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
      const empty = !(payload.nodes || []).length && tab !== VIEW.designator;
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

    async function showTab(name) {
      tab = name;
      Array.prototype.forEach.call(tabsEl.querySelectorAll('button'), function (button) {
        button.classList.toggle('active', button.dataset.view === name);
      });
      Object.keys(containers).forEach(function (view) {
        containers[view].style.display = view === name ? '' : 'none';
      });
      toolbarEl.style.display = name === VIEW.hypothesis ? '' : 'none';
      legendEl.style.display = name === VIEW.hypothesis ? '' : 'none';
      emptyEl.style.display = 'none';
      try {
        if (!base[name]) {
          const payload = await load(name);
          if (!payload.ok) {
            showError(payload.error || 'view unavailable');
            return;
          }
          base[name] = payload;
          showEmptyState(payload);
          draw[name](payload);
        } else {
          showEmptyState(base[name]);
          draw[name](base[name]);
        }
      } catch (error) {
        showError('Could not load this view: ' + ((error && error.message) || error));
        return;
      }
      if (name === VIEW.hypothesis) HypothesisGraph.showLegend(legendEl);
    }

    Array.prototype.forEach.call(tabsEl.querySelectorAll('button'), function (button) {
      button.addEventListener('click', function () { showTab(button.dataset.view); });
    });

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
      if (tab !== VIEW.hypothesis) return;
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
    bus.on('hypothesis:generated', function (payload) {
      if (!payload || !payload.ok) return;
      Object.keys(base).forEach(function (view) { delete base[view]; });
      base[VIEW.hypothesis] = payload;
      hypothesisDrawn = false;
      showTab(tab);
    });

    showTab(VIEW.hypothesis);
    return {};
  });

  // mounted below the panels cramera ships, before the page boots them
  const layout = (window.CRAMERA_CONFIG || {}).layout || {};
  if (layout.right) layout.right.push(PANEL_ID);
})();
