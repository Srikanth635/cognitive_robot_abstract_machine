/* ============================================================================
 * panels/graph/panel.js — the graph view with eight tabs.
 *
 *   Knowledge        the scene-backed entity graph + CRAM architecture
 *   Live Knowledge   semantic entities projected from the attached live execution
 *   Hypothesis Graph llm_reasoner's cereal-pouring knowledge graph, rendered by its
 *                    own dedicated engine (see hypothesis-graph.js) instead of
 *                    graph.js's shared window.Graph
 *   Ontology Graph   the same graph with the ontology class of every node and the
 *                    ontology property of every link beneath their labels
 *   Designator       a real coraplex action designator's own fields, reflected and
 *                    classified value vs referent -- static and illustrative, no
 *                    live World, no resolver call (see cram_viz.designator_knowledge)
 *   Kinematics       the robot's URDF tree (links as nodes, joints as edges)
 *   Plan             the executed plan tree, node border = execution status
 *   Statechart       the giskardpy motion statechart of the running motion group
 *
 * Live Knowledge, Plan and Statechart take data from the cram-viz-live bridge
 * while it is attached. The scene-backed Knowledge graph remains independent.
 *
 * Bus events:
 *   emits    entity:select {id, detail, relations}   node clicked
 *   listens  entity:highlight {ids, focus?}          spotlight matching nodes -- on
 *                                                     the Hypothesis Graph tab this
 *                                                     also reveals the root-to-node
 *                                                     path for each id, since an EQL
 *                                                     answer can be buried in a
 *                                                     collapsed branch
 *   listens  scene:step {step}                       highlight the running episode
 *   listens  live:changed {on, url}                  start/stop the status poll
 *
 * Rendering is delegated to graph.js (window.Graph, vis-network wrapper).
 * ==========================================================================*/
Panels.define('graph', function (root, bus) {
  root.innerHTML =
    '<div class="graph-wrap">' +
    '  <div class="graph-tabs" id="graph-tabs">' +
    '    <button data-view="knowledge" class="active" title="the entity graph (EQL / knowledge base)">Knowledge</button>' +
    '    <button data-view="live-knowledge" title="semantic entities and actions projected from the attached live demo">Live Knowledge</button>' +
    '    <button data-view="hypothesis" title="llm_reasoner\'s cereal-pouring knowledge graph — click a node to expand it, double-click to collapse its branch">Hypothesis Graph</button>' +
    '    <button data-view="ontology" title="the Hypothesis Graph with the ontology class of every node and the ontology property of every link beneath their labels — click a node for the terms backing its values">Ontology Graph</button>' +
    '    <button data-view="designator" title="a real coraplex action designator\'s own fields, reflected and classified value vs referent — illustrative only, no live World">Designator</button>' +
    '    <button data-view="kinematics" title="the robot\'s kinematic structure — URDF links &amp; joints">Kinematics</button>' +
    '    <button data-view="plan" title="the plan tree, with the execution status of every node">Plan</button>' +
    '    <button data-view="chart" title="the giskardpy motion statechart of the running motion group">Statechart</button>' +
    '    <span class="gt-live" id="gt-live" title="node status is streaming from the running demo">◉ live status</span>' +
    '  </div>' +
    '  <div id="designator-lisp" class="designator-lisp"></div>' +
    '  <div id="graph"></div>' +
    '  <div id="hypothesis-graph" style="display:none"></div>' +
    '  <div id="graph-empty" class="graph-empty" style="display:none"></div>' +
    '  <div id="graph-nav" class="graph-nav" style="display:none">' +
    '    <button id="gnav-home" title="back to the overview">⌂</button>' +
    '    <button id="gnav-up" title="one level up">↑ back</button>' +
    '    <span id="gnav-path"></span>' +
    '  </div>' +
    '  <div id="hypothesis-toolbar" class="hypothesis-toolbar" style="display:none">' +
    '    <button id="hg-expand-all" title="Expand every node in the graph">Expand all</button>' +
    '    <button id="hg-collapse-all" title="Collapse back to the root node">Collapse all</button>' +
    '  </div>' +
    '  <div class="legend" id="legend"></div>' +
    '</div>';

  const emptyEl = root.querySelector('#graph-empty');
  const navEl = root.querySelector('#graph-nav');
  const navUp = root.querySelector('#gnav-up');
  const navHome = root.querySelector('#gnav-home');
  const navPath = root.querySelector('#gnav-path');
  const tabsEl = root.querySelector('#graph-tabs');
  const liveBadge = root.querySelector('#gt-live');
  const graphWrapEl = root.querySelector('.graph-wrap');
  const graphEl = root.querySelector('#graph');
  const hypothesisGraphEl = root.querySelector('#hypothesis-graph');
  const hypothesisToolbarEl = root.querySelector('#hypothesis-toolbar');
  const hgExpandAllBtn = root.querySelector('#hg-expand-all');
  const hgCollapseAllBtn = root.querySelector('#hg-collapse-all');
  hgExpandAllBtn.addEventListener('click', function () { HypothesisGraph.expandAll(); });
  hgCollapseAllBtn.addEventListener('click', function () { HypothesisGraph.collapseAll(); });

  // ---- tabs -------------------------------------------------------------------
  const TABS = {
    knowledge:  { url: '/api/kb' },
    'live-knowledge': { live: 'knowledge' },
    hypothesis: { url: '/api/kb/view?name=hypothesis' },
    ontology:   { url: '/api/kb/view?name=ontology' },
    designator: { url: '/api/kb/view?name=designator' },
    kinematics: { url: '/api/kb/view?name=kinematics' },
    plan:       { url: '/api/kb/view?name=plan' },
    chart:      { url: '/api/kb/view?name=chart' },
  };
  let tab = 'knowledge';
  let view = null;            // the currently rendered payload
  const base = {};            // tab -> payload as loaded from the server
  const shown = {};           // tab -> payload currently rendered (drill-downs)
  const stacks = {};          // tab -> parent payloads for the back button
  Object.keys(TABS).forEach(function (t) { stacks[t] = []; });
  let inGraphSet = {};

  // The Hypothesis and Ontology Graph tabs draw the same projection with different
  // labels, so they share one HypothesisGraph renderer and container; it is rebuilt
  // whenever the other of the two was drawn last.
  const PROJECTION_TABS = ['hypothesis', 'ontology'];
  function isProjectionTab(name) { return PROJECTION_TABS.includes(name); }
  let drawnProjectionTab = null;
  function drawProjection(name) {
    if (drawnProjectionTab === name) return;
    drawnProjectionTab = name;
    HypothesisGraph.build(base[name]);
  }

  function setView(payload) {
    view = payload;
    shown[tab] = payload;
    inGraphSet = {};
    payload.nodes.forEach(function (n) { inGraphSet[n.id] = 1; });
    if (emptyEl) {
      const empty = !payload.nodes.length;
      emptyEl.style.display = empty ? '' : 'none';
      emptyEl.textContent = empty ? (payload.empty || 'Nothing to show in this view.') : '';
    }
    Graph.build({
      nodes: payload.nodes, edges: payload.edges, legend: payload.legend,
      layout: payload.layout, arrows: !!payload.arrows, statusLegend: !!payload.statusLegend,
      spacing: payload.spacing,
      key: (payload.key || tab) + '#' + stacks[tab].length,
    });
    updateNav();
  }
  function updateNav() {
    const inside = stacks[tab].length > 0;
    navEl.style.display = inside ? '' : 'none';
    if (inside) {
      const path = stacks[tab].slice(1).map(function (v) { return v.crumb; }).concat([view.crumb]);
      navPath.textContent = path.join(' / ');
    }
  }
  async function drill(id) {
    if (tab === 'live-knowledge') return;
    if (!view.details[id]) return;
    try {
      const r = await fetch(SceneSelection.apiUrl(
        '/api/kb/expand?node=' + encodeURIComponent(id),
        window.location.search,
      ));
      const p = await r.json();
      if (!p.ok) return;                       // node has no inside view
      stacks[tab].push(view);
      setView(p);
      select(id);
    } catch (err) { /* server unreachable — stay where we are */ }
  }
  function goBack() { if (stacks[tab].length) setView(stacks[tab].pop()); }
  function goHome() {
    if (!stacks[tab].length) return;
    stacks[tab] = [];
    setView(base[tab]);
  }
  navUp.addEventListener('click', goBack);
  navHome.addEventListener('click', goHome);

  function liveKnowledgePlaceholder() {
    const payload = new LiveKnowledgeGraph(null, null).build();
    payload.live = 'knowledge';
    payload.empty = 'Attach a running demo, then open Live Knowledge to follow it.';
    return payload;
  }

  async function showTab(name) {
    if (!TABS[name]) return;
    tab = name;
    tabsEl.querySelectorAll('button').forEach(function (b) {
      b.classList.toggle('active', b.dataset.view === name);
    });
    graphEl.style.display = isProjectionTab(name) ? 'none' : '';
    hypothesisGraphEl.style.display = isProjectionTab(name) ? '' : 'none';
    hypothesisToolbarEl.style.display = isProjectionTab(name) ? '' : 'none';
    graphWrapEl.classList.toggle('split', name === 'designator');
    if (isProjectionTab(name)) {
      navEl.style.display = 'none';
      emptyEl.style.display = 'none';
      if (base[name]) {
        showHypothesisEmptyState(base[name]);
        drawProjection(name);
      }
      if (!base[name]) {
        try {
          const r = await fetch(SceneSelection.apiUrl(
            TABS[name].url,
            window.location.search,
          ));
          const p = await r.json();
          if (!p.ok) {
            emptyEl.style.display = '';
            emptyEl.textContent = p.error || 'view unavailable';
            return;
          }
          base[name] = p;
          showHypothesisEmptyState(p);
          drawnProjectionTab = name;
          HypothesisGraph.build(p);
        } catch (err) {
          emptyEl.style.display = '';
          emptyEl.textContent = 'Could not load this view: ' + ((err && err.message) || err);
          return;
        }
      }
      HypothesisGraph.showLegend();
      return;
    }
    if (!base[name] && TABS[name].live) {
      base[name] = liveKnowledgePlaceholder();
    }
    if (!base[name]) {
      emptyEl.style.display = '';
      emptyEl.textContent = 'loading…';
      try {
        const r = await fetch(SceneSelection.apiUrl(
          TABS[name].url,
          window.location.search,
        ));
        if (r.status === 404) throw new Error('this build needs the /api/kb/view route — restart the server');
        const p = await r.json();
        if (!p.ok) {
          emptyEl.textContent = p.error || 'view unavailable';
          return;
        }
        p.key = name;
        base[name] = p;
      } catch (err) {
        emptyEl.textContent = 'Could not load this view: ' + ((err && err.message) || err);
        return;
      }
    }
    setView(shown[name] || base[name]);
    if (name === 'designator') DesignatorLisp.build(shown[name] || base[name]);
    liveRefresh(true);            // a live tab picks the bridge status up at once
  }
  tabsEl.querySelectorAll('button').forEach(function (b) {
    b.addEventListener('click', function () { showTab(b.dataset.view); });
  });

  // ---- node click → describe in whatever panel listens -------------------------
  function select(id) {
    const d = view && view.details && view.details[id];
    if (!d) return;
    const relations = (view.edges || [])
      .filter(function (e) { return e.from === id || e.to === id; })
      .map(function (e) {
        return { s: labelOf(e.from), p: e.label || e.kind, o: labelOf(e.to) };
      });
    bus.emit('entity:select', { id: id, detail: d, relations: relations });
    if (tab === 'designator') DesignatorLisp.highlight(id);
    spotlight({ ids: [id], focus: id });
  }
  function labelOf(id) { return (view.details[id] && view.details[id].label) || id; }
  Graph.onSelect(select);
  Graph.onDoubleSelect(drill);
  DesignatorLisp.onSelect(select);

  // ---- Hypothesis Graph tab: same entity:select overlay, its own click source ----
  // Nothing is generated at startup, so the tab explains how to fill it rather than
  // rendering an empty canvas.
  function showHypothesisEmptyState(payload) {
    const empty = !(payload.nodes || []).length;
    emptyEl.style.display = empty ? '' : 'none';
    emptyEl.textContent = empty ? (payload.empty || 'Nothing to show in this view.') : '';
  }

  function hypothesisLabelOf(payload, id) {
    return (payload.details[id] && payload.details[id].label) || id;
  }
  function hypothesisNeighbourIds(payload, focusId) {
    return (payload.edges || [])
      .filter(function (e) { return e.from === focusId || e.to === focusId; })
      .map(function (e) { return e.from === focusId ? e.to : e.from; });
  }
  function hypothesisSelect(id) {
    const payload = base[drawnProjectionTab];
    if (!payload || !payload.details || !payload.details[id]) return;
    const relations = (payload.edges || [])
      .filter(function (e) { return e.from === id || e.to === id; })
      .map(function (e) {
        return {
          s: hypothesisLabelOf(payload, e.from),
          p: e.label,
          o: hypothesisLabelOf(payload, e.to),
        };
      });
    bus.emit('entity:select', { id: id, detail: payload.details[id], relations: relations });
    // a clicked node is already on screen, so there's nothing to reveal -- just
    // spotlight it and its neighbours in place, the same as before this node
    // triggered its own click.
    HypothesisGraph.highlight([id].concat(hypothesisNeighbourIds(payload, id)));
  }
  HypothesisGraph.onSelect(hypothesisSelect);

  // ---- highlights (from EQL results or our own selection) ----------------------
  function spotlight(p) {
    const ids = (p && p.ids) || [];
    if (isProjectionTab(tab)) {
      const payload = base[tab];
      let hypothesisHi = ids.slice();
      if (payload && p && p.focus) {
        hypothesisHi = hypothesisHi.concat(hypothesisNeighbourIds(payload, p.focus));
      }
      if (!hypothesisHi.length) { HypothesisGraph.reset(); return; }
      // entity:highlight always carries a single p.focus when it originates from a
      // click (hypothesisSelect -> entity:select -> eql/panel.js's describe(),
      // which re-emits entity:highlight with that same id as focus) -- a clicked
      // node is already on screen, so spotlighting it and its neighbours in place
      // is enough. Only a genuine EQL query result (no single focus; a hit can be
      // buried in a currently-collapsed branch) needs revealAndHighlight's full
      // reveal-from-root. Calling that on every click instead would undo the
      // click's own single-branch expand() the moment the clicked node has a
      // *different* parent elsewhere (e.g. a role or event shared by several
      // phases) -- ancestorsOf() would reveal that unrelated parent's branch too.
      if (p && p.focus) {
        HypothesisGraph.highlight(hypothesisHi);
      } else {
        HypothesisGraph.revealAndHighlight(hypothesisHi);
      }
      return;
    }
    let hi = ids.filter(function (id) { return inGraphSet[id]; });
    if (p && p.focus && inGraphSet[p.focus]) {
      const neighbours = (view.edges || [])
        .filter(function (e) { return e.from === p.focus || e.to === p.focus; })
        .map(function (e) { return e.from === p.focus ? e.to : e.from; });
      hi = hi.concat(neighbours.filter(function (id) { return inGraphSet[id]; }));
    }
    if (hi.length) Graph.highlight(hi); else Graph.reset();
  }
  bus.on('entity:highlight', spotlight);

  // ---- freshly generated hypothesis knowledge ----------------------------------
  // The EQL panel posts an instruction to the generator and publishes the refreshed
  // graph here. It arrives complete, so it replaces the cached payload outright
  // instead of invalidating it and refetching.
  bus.on('hypothesis:generated', function (payload) {
    if (!payload || !payload.ok) return;
    base['hypothesis'] = payload;
    delete base['ontology'];
    drawnProjectionTab = null;
    showTab(tab === 'ontology' ? 'ontology' : 'hypothesis');
  });

  bus.on('scene:step', function (p) {
    if (p.step === '__done__') { Graph.reset(); return; }
    if (tab === 'knowledge' && !stacks[tab].length && inGraphSet[p.step]) select(p.step);
  });

  // ---- live status overlay (Plan / Statechart tabs) -----------------------------
  // The bridge publishes the plan tree and the executing motion statechart with
  // per-node status. Structure changes (the plan grows as actions expand, a new
  // statechart is compiled per motion group) rebuild the graph; a pure status
  // change only re-colours the rings, so the layout never jumps.
  const PLAN_GROUP = {
    ActionNode: 'event', MotionNode: 'robot', ConditionNode: 'goal',
    AttachmentNode: 'object', DetachmentNode: 'object',
  };
  const PLAN_LEGEND = [
    { group: 'event', label: 'Action' }, { group: 'robot', label: 'Motion' },
    { group: 'goal', label: 'Condition' }, { group: 'object', label: 'Attach / detach' },
    { group: 'ind', label: 'Other plan node' },
  ];
  const CHART_LEGEND = [
    { group: 'robot', label: 'Task (motion constraint)' },
    { group: 'concept', label: 'Monitor / observation' },
    { group: 'klass', label: 'Goal (contains nodes)' },
    { group: 'event', label: 'End / cancel motion' },
  ];
  const LIVE_STATUS_INTERVAL_MILLISECONDS = 100;
  const liveSig = { knowledge: '', plan: '', chart: '' };
  let liveHighlightSignature = '';
  let liveTimer = null;
  let liveState = { on: false, url: '' };

  function liveSource() {
    if (TABS[tab].live) return TABS[tab].live;
    const payload = shown[tab] || base[tab];
    return (payload && payload.live) || null;
  }

  function applyLiveKnowledgeHighlight(payload, force) {
    const identifiers = payload.highlight || [];
    const signature = identifiers.join('|');
    if (!force && signature === liveHighlightSignature) return;
    liveHighlightSignature = signature;
    if (identifiers.length) Graph.highlight(identifiers); else Graph.reset();
  }

  function planPayload(live) {
    const nodes = [], edges = [], details = {};
    (live.nodes || []).forEach(function (n) {
      const label = (n.label || '?').replace('Action', '');
      const lines = ['a ' + n.kind,
                     'status: ' + n.status + (n.derived ? ' (derived from the motion statechart)' : '')];
      if (n.arm) lines.push('arm: ' + n.arm);
      if (n.target) lines.push('target: ' + n.target);
      nodes.push({ id: n.id, label: label, group: PLAN_GROUP[n.kind] || 'ind',
                   title: [label].concat(lines).join('\n'), status: n.status });
      details[n.id] = { label: label, group: PLAN_GROUP[n.kind] || 'ind', lines: lines };
      if (n.parent) edges.push({ from: n.parent, to: n.id, kind: 'prop', label: 'has step' });
    });
    return { ok: true, crumb: 'live plan', nodes: nodes, edges: edges, details: details,
             legend: PLAN_LEGEND, layout: 'hier', arrows: true, statusLegend: true,
             live: 'plan', key: 'plan-live',
             empty: 'The bridge is attached but the demo has not started its plan yet.' };
  }

  function chartPayload(live) {
    const nodes = [], edges = [], details = {}, isParent = {};
    (live.nodes || []).forEach(function (n) { if (n.parent) isParent[n.parent] = 1; });
    (live.nodes || []).forEach(function (n) {
      const group = isParent[n.id] ? 'klass'
        : /EndMotion|CancelMotion/.test(n.cls) ? 'event'
        : /Monitor|Reached|Observation|Condition/.test(n.cls + n.name) ? 'concept' : 'robot';
      const lines = ['a ' + n.cls, 'life cycle: ' + n.life, 'observation: ' + n.obs];
      nodes.push({ id: n.id, label: n.name, group: group,
                   title: [n.name].concat(lines).join('\n'), status: n.life });
      details[n.id] = { label: n.name, group: group, lines: lines };
      if (n.parent) edges.push({ from: n.parent, to: n.id, kind: 'type', label: 'contains' });
    });
    (live.edges || []).forEach(function (e) {
      edges.push({ from: e.from, to: e.to, kind: e.kind, label: (e.kind || '').toLowerCase() + ' transition' });
    });
    return { ok: true, crumb: 'statechart' + (live.title ? ' · ' + live.title : ''),
             nodes: nodes, edges: edges, details: details, legend: CHART_LEGEND,
             layout: 'hier', arrows: true, statusLegend: true, live: 'chart', key: 'chart-live',
             empty: 'Attached, but no motion statechart is executing right now.' };
  }

  async function liveRefresh(force) {
    const source = liveSource();
    const active = !!source && liveState.on;
    liveBadge.classList.toggle('on', active);
    if (!active) return;
    if (stacks[tab].length) return;
    let payload;
    let signature;
    try {
      if (source === 'knowledge') {
        const responses = await Promise.all([
          fetch(liveState.url + '/plan'),
          fetch(liveState.url + '/info'),
        ]);
        const plan = await responses[0].json();
        const information = await responses[1].json();
        payload = new LiveKnowledgeGraph(plan, information).build();
        if (!payload.nodes.length) return;
        signature = payload.sig;
      } else {
        const live = await fetch(
          liveState.url + (source === 'plan' ? '/plan' : '/chart'),
        ).then(function (response) { return response.json(); });
        if (!live || !live.nodes) return;
        payload = source === 'plan' ? planPayload(live) : chartPayload(live);
        signature = live.sig;
      }
    } catch (error) { return; }
    if (force || signature !== liveSig[source]) {
      liveSig[source] = signature;
      if (source !== 'knowledge') base[tab] = payload;
      setView(payload);
      if (source === 'knowledge') applyLiveKnowledgeHighlight(payload, true);
      return;
    }
    const statusByIdentifier = {};
    payload.nodes.forEach(function (node) {
      if (node.status) statusByIdentifier[node.id] = node.status;
    });
    if (!Graph.setStatuses(statusByIdentifier)) {
      if (source !== 'knowledge') base[tab] = payload;
      setView(payload);
      if (source === 'knowledge') applyLiveKnowledgeHighlight(payload, true);
      return;
    }
    if (source !== 'knowledge') base[tab] = payload;
    if (view && view.details) view.details = payload.details;
    if (source === 'knowledge') applyLiveKnowledgeHighlight(payload);
  }

  bus.on('live:changed', function (p) {
    liveState = { on: !!p.on, url: p.url || '' };
    if (liveTimer) { clearInterval(liveTimer); liveTimer = null; }
    if (liveState.on) {
      liveTimer = setInterval(
        function () { liveRefresh(false); },
        LIVE_STATUS_INTERVAL_MILLISECONDS,
      );
      liveRefresh(true);
    } else {
      liveBadge.classList.remove('on');
      liveSig.knowledge = liveSig.plan = liveSig.chart = '';
      liveHighlightSignature = '';
      // drop live payloads so every tab falls back to the recorded bundle
      ['plan', 'chart'].forEach(function (viewName) {
        delete base[viewName];
        delete shown[viewName];
        stacks[viewName] = [];
      });
      delete base['live-knowledge'];
      delete shown['live-knowledge'];
      stacks['live-knowledge'] = [];
      if (tab === 'live-knowledge' || tab === 'plan' || tab === 'chart') showTab(tab);
    }
  });

  // ---- boot ---------------------------------------------------------------------
  showTab('knowledge');
});
