/* ============================================================================
 * panels/eql/panel.js — the EQL (Entity Query Language) console.
 *
 * Query box + predefined designators + presets + answer panel. Queries are executed
 * server-side (POST /api/eql) against the episode knowledge base; the KB overview
 * (GET /api/kb) provides presets and per-entity details. The scene's predefined,
 * underspecified designators (GET /api/designators) are resolved by llm_reasoner in the
 * scene on a click (POST /api/designator/resolve).
 *
 * Bus events:
 *   emits    kb:ready {payload}          the /api/kb overview, once loaded
 *   emits    entity:highlight {ids, focus?}   results / described entity
 *   emits    hypothesis:generated {payload}   the hypothesis a resolution consulted
 *   listens  scene:part-clicked {id}     describe the clicked part
 *   listens  scene:step {step}           describe the running episode
 *   listens  entity:select {id, detail, relations}   node clicked in a graph
 * ==========================================================================*/
Panels.define('eql', function (root, bus) {
  root.innerHTML =
    '<div class="panel-head">' +
    '  <h2>EQL · entity query language</h2>' +
    '  <span id="kb-status" class="kb-status">loading knowledge base…</span>' +
    '</div>' +
    '<div class="query-box">' +
    '  <div class="query-row">' +
    '    <textarea id="query-input" rows="2" spellcheck="false" placeholder="the(entity(obj).where(obj.name == \'milk\'))   —  vars: obj, ep, arm, j, rob"></textarea>' +
    '    <button id="query-run">Run</button>' +
    '  </div>' +
    '  <div class="designator-row">' +
    '    <div id="designators" class="designators" title="Predefined, underspecified pouring designators as krrood match queries: click one to have llm_reasoner resolve it in this scene"></div>' +
    '    <button id="nl-execute" title="Launch the demo under the live bridge — hardcoded for now, independent of the designators">Execute</button>' +
    '  </div>' +
    '  <div id="presets" class="presets"></div>' +
    '</div>' +
    '<div id="answer" class="answer"><div class="answer-empty">Loading the episode knowledge base…</div></div>';

  const kbStatus = root.querySelector('#kb-status');
  const answerEl = root.querySelector('#answer');
  const input = root.querySelector('#query-input');
  const runBtn = root.querySelector('#query-run');
  const presetsEl = root.querySelector('#presets');
  const designatorsEl = root.querySelector('#designators');
  const nlExecuteBtn = root.querySelector('#nl-execute');

  let kb = null;   // /api/kb overview (presets + entity details)

  // ---- boot -----------------------------------------------------------------
  fetch(SceneSelection.apiUrl('/api/kb', window.location.search)).then(function (r) {
    return r.json();
  }).then(boot).catch(function (err) {
    kbStatus.textContent = 'KB error';
    answerEl.innerHTML = '<div class="qerr">Failed to reach the EQL server:\n' + esc(String(err)) + '</div>';
  });

  function boot(payload) {
    if (!payload.ok) {
      kbStatus.textContent = 'EQL unavailable';
      answerEl.innerHTML = '<div class="qerr">' + esc(payload.error || 'unknown error') + '</div>';
      return;
    }
    kb = payload;
    kbStatus.textContent = payload.status;
    kbStatus.classList.add('ready');
    buildPresets(payload.presets || []);
    welcome();
    bus.emit('kb:ready', { payload: payload });
    loadDesignators();
  }

  function welcome() {
    answerEl.innerHTML =
      '<div class="goal">EQL · knowledge &amp; reasoning</div>' +
      '<p class="headline"><b>Correctness, concepts and specifications</b> are captured as ' +
      '<b>rules</b> and <b>description-logic axioms / predicates</b>, and made explorable as a ' +
      '<b>graph</b> — queried with <b>EQL</b>, krrood’s pythonic entity query language from the ' +
      'CRAM architecture.</p>' +
      '<p class="hint-txt">Ready-made variables: <code>obj</code> (bench objects), <code>ep</code> ' +
      '(action episodes), <code>arm</code>, <code>j</code> (joint motion), <code>rob</code>, ' +
      '<code>pkg</code> / <code>sub</code> / <code>cls</code> (CRAM packages, subpackages, classes). ' +
      'Build queries like <code>the(entity(obj).where(obj.name == \'milk\'))</code> — ' +
      'or click a preset below, or a node in the graph.</p>';
  }

  // ---- describe an entity in the answer panel --------------------------------
  // Two sources: our own kb.details (scene clicks) and graph panels, which send
  // the full detail payload of the node the user clicked (entity:select).
  function describe(id, detail, relations) {
    const d = detail || (kb && kb.details && kb.details[id]);
    if (!d) return;
    let html = '<div class="goal">entity · ' + esc(id) + '</div>';
    html += '<div class="ansrow"><span class="tag" style="background:' + groupColor(d.group) + '">' +
      esc(d.group) + '</span><div class="body"><span class="name">' + esc(d.label) + '</span></div></div>';
    (d.lines || []).forEach(function (l) {
      html += '<div class="ansrow"><div class="body"><span class="name">' + esc(l) + '</span></div></div>';
    });
    if (relations && relations.length) {
      html += '<div class="ansub">Relations</div>';
      relations.slice(0, 40).forEach(function (r) {
        html += '<div class="ansrow"><div class="body"><span class="name">' + esc(r.s) +
          ' <span class="rel">' + esc(r.p) + '</span> ' + esc(r.o) + '</span></div></div>';
      });
      if (relations.length > 40) {
        html += '<div class="ansrow"><div class="body"><span class="sub">… ' + (relations.length - 40) + ' more</span></div></div>';
      }
    }
    answerEl.innerHTML = html;
    bus.emit('entity:highlight', { ids: [id], focus: id });
  }

  bus.on('scene:part-clicked', function (p) { describe(p.id); });
  bus.on('entity:select', function (p) { describe(p.id, p.detail, p.relations); });
  bus.on('scene:step', function (p) {
    if (p.step !== '__done__' && kb && kb.details && kb.details[p.step]) describe(p.step);
  });

  // ---- presets ----------------------------------------------------------------
  function buildPresets(presets) {
    presetsEl.innerHTML = '';
    presets.forEach(function (p) {
      const b = document.createElement('div');
      b.className = 'preset'; b.textContent = p.text;
      b.title = p.code;
      b.addEventListener('click', function () {
        input.value = p.code;
        runQuery(p.code);
      });
      presetsEl.appendChild(b);
    });
  }

  // ---- predefined designators ----------------------------------------------------
  // Underspecified pouring designators, each a krrood match query the resolver completes
  // in the scene shown: it reads the scene's world, consults the knowledge component, and
  // the answer panel shows
  // what it chose, why and what that rests on. The consulted hypothesis replaces what the
  // Hypothesis Graph, Ontology Graph and Designator tabs show, and every EQL variable over
  // it, so the presets are reloaded alongside it.
  async function loadDesignators() {
    try {
      const response = await fetch(SceneSelection.apiUrl('/api/designators', window.location.search));
      buildDesignators(await response.json());
    } catch (err) {
      designatorsEl.textContent = 'designators unavailable: ' + String(err);
    }
  }

  function buildDesignators(payload) {
    designatorsEl.innerHTML = '';
    if (!payload.ok) {
      designatorsEl.textContent = 'designators unavailable: ' + (payload.error || 'unknown error');
      return;
    }
    payload.designators.forEach(function (designator) {
      const chip = document.createElement('pre');
      chip.className = 'designator-query' + (designator.available ? '' : ' unavailable');
      chip.textContent = designator.query;
      chip.title = designatorHint(designator);
      if (designator.available) {
        chip.addEventListener('click', function () { resolve(designator); });
      }
      designatorsEl.appendChild(chip);
    });
  }

  function designatorHint(designator) {
    if (designator.available) {
      return designator.identifier + ': click to resolve it in this scene';
    }
    const lacks = designator.unbound.map(function (name) {
      return 'an object named ' + name;
    }).concat(designator.without_candidates.map(function (slot) {
      return 'a candidate for ' + slot;
    }));
    return designator.identifier + ' needs ' + lacks.join(', ') + ', which this scene lacks';
  }

  let resolving = false;
  async function resolve(designator) {
    if (resolving) return;
    resolving = true;
    answerEl.innerHTML =
      '<div class="goal">resolving · ' + esc(designator.identifier) + '</div>' +
      queryHtml(designator.query) +
      '<p class="hint-txt">The resolver reads the scene and consults the knowledge ' +
      'component about the action. Knowledge for a situation met before is served from ' +
      'the store; a new situation is generated once, which takes a few minutes.</p>';
    try {
      const response = await fetch('/api/designator/resolve', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          designator: designator.identifier,
          scene: SceneSelection.nameFromSearch(window.location.search),
        }),
      });
      const payload = await response.json();
      if (!payload.ok) {
        answerEl.innerHTML =
          '<div class="goal">resolution failed · ' + esc(designator.identifier) + '</div>' +
          '<div class="qerr">' + esc(payload.error || 'resolution failed') + '</div>';
      } else {
        answerEl.innerHTML = resolutionHtml(payload.resolution);
        bus.emit('hypothesis:generated', payload.hypothesis);
        await reloadKnowledgeBase();
      }
    } catch (err) {
      answerEl.innerHTML =
        '<div class="goal">resolution failed</div>' +
        '<div class="qerr">' + esc(String(err)) + '</div>';
    }
    resolving = false;
  }

  function queryHtml(query) {
    return '<pre class="resolved-query">' + esc(query) + '</pre>';
  }

  function resolutionHtml(resolution) {
    let html = '<div class="goal">resolved · ' + esc(resolution.identifier) + '</div>' +
      queryHtml(resolution.query);
    resolution.actions.forEach(function (action) {
      html += '<div class="ansub">' + esc(resolution.designator) + '</div>';
      action.slots.forEach(function (slot) {
        html += '<div class="ansrow"><span class="tag">' + (slot.given ? 'given' : 'chosen') +
          '</span><div class="body"><span class="name">' + esc(slot.name) + ' = ' +
          esc(slot.value) + '</span></div></div>';
      });
      html += '<div class="ansrow"><div class="body"><span class="sub">why: ' +
        esc(action.why) + '</span></div></div>';
      action.grounds.forEach(function (ground) {
        html += '<div class="ansrow"><div class="body"><span class="sub">rests on ' +
          esc(ground.source) + ': ' + esc(ground.name) + '</span></div></div>';
      });
    });
    const situation = Object.keys(resolution.situation).map(function (feature) {
      return feature + ' = ' + resolution.situation[feature];
    });
    html += '<p class="hint-txt">knowledge consulted for ' +
      esc(situation.length ? situation.join(', ') : 'an unknown situation') +
      (resolution.record_id ? ' (hypothesis ' + esc(resolution.record_id) + ')' : '') +
      ' · resolver cost $' + resolution.cost.toFixed(3) +
      ' · shown in the Hypothesis Graph, Ontology Graph and Designator tabs</p>';
    return html;
  }

  async function reloadKnowledgeBase() {
    const response = await fetch(SceneSelection.apiUrl('/api/kb', window.location.search));
    const payload = await response.json();
    if (!payload.ok) return;
    kb = payload;
    buildPresets(payload.presets || []);
    bus.emit('kb:ready', { payload: payload });
  }

  // ---- execute the hardcoded demo under the live bridge -------------------------
  // Independent of the designators and the hypothesis above -- for now this
  // always launches the same fixed demo script. Nothing further is wired here to
  // show it running: the 3D scene panel already polls the live bridge's /info
  // endpoint every few seconds and shows its own "Live" button the moment one
  // answers, exactly as if the demo had been started from a terminal.
  let executing = false;
  async function execute() {
    if (executing) return;
    executing = true;
    nlExecuteBtn.textContent = '…';
    nlExecuteBtn.disabled = true;
    try {
      const response = await fetch('/api/execute', { method: 'POST' });
      const payload = await response.json();
      if (!payload.ok) {
        answerEl.innerHTML =
          '<div class="goal">execution failed</div>' +
          '<div class="qerr">' + esc(payload.error || 'execution failed') + '</div>';
      } else {
        answerEl.innerHTML =
          '<div class="goal">demo launched</div>' +
          '<p class="hint-txt">Starting under the live bridge. Once it comes up, ' +
          'the 3D scene\'s own Live button will appear — click it to attach.</p>';
      }
    } catch (err) {
      answerEl.innerHTML =
        '<div class="goal">execution failed</div>' +
        '<div class="qerr">' + esc(String(err)) + '</div>';
    }
    executing = false;
    nlExecuteBtn.textContent = 'Execute';
    nlExecuteBtn.disabled = false;
  }

  nlExecuteBtn.addEventListener('click', execute);

  // ---- run an EQL query --------------------------------------------------------
  let running = false;
  async function runQuery(code) {
    code = (code || '').trim(); if (!code || running) return;
    running = true;
    runBtn.textContent = '…';
    try {
      const r = await fetch('/api/eql', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          code: code,
          scene: SceneSelection.nameFromSearch(window.location.search),
        }),
      });
      render(code, await r.json());
    } catch (err) {
      render(code, { ok: false, error: String(err) });
    }
    running = false;
    runBtn.textContent = 'Run';
  }

  function render(code, res) {
    let html = '<div class="goal">&gt;&gt;&gt; ' + esc(code) + '</div>';
    if (!res.ok) {
      answerEl.innerHTML = html + '<div class="qerr">' + esc(res.error || 'query failed') + '</div>';
      bus.emit('entity:highlight', { ids: [] });
      return;
    }
    if (!res.count) {
      answerEl.innerHTML = html + '<div class="nores">No solutions — the query returned nothing.</div>';
      bus.emit('entity:highlight', { ids: [] });
      return;
    }
    html += '<p class="headline"><b>' + res.count + '</b> result' + (res.count === 1 ? '' : 's') +
      (res.more ? ' (truncated)' : '') + '.</p>';
    res.rows.forEach(function (row) {
      if (row.__entity__ !== undefined) html += entityRow(row);
      else html += valueRow(row);
    });
    answerEl.innerHTML = html;
    bus.emit('entity:highlight', { ids: res.highlight || [] });
  }

  function entityRow(row) {
    const g = groupOfType(row.__type__);
    const sub = [];
    for (const k in row) {
      if (k.indexOf('__') === 0 || row[k] === null || row[k] === undefined) continue;
      sub.push(k + ': ' + row[k]);
    }
    return '<div class="ansrow"><span class="tag" style="background:' + groupColor(g) + '">' +
      esc(row.__type__) + '</span><div class="body"><span class="name">' + esc(row.__entity__) +
      '</span><span class="sub">' + esc(sub.join('  ·  ')) + '</span></div></div>';
  }
  function valueRow(row) {
    const parts = Object.keys(row).map(function (k) {
      return '<code>' + esc(k) + ' = ' + esc(String(row[k])) + '</code>';
    }).join(' ');
    return '<div class="ansrow"><div class="body">' + parts + '</div></div>';
  }

  runBtn.addEventListener('click', function () { runQuery(input.value); });
  input.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); runQuery(input.value); }
  });

  // ---- helpers ----------------------------------------------------------------
  const TYPE_GROUP = {
    BenchObject: 'object', ActionEpisode: 'event', Arm: 'robot', Gripper: 'robot',
    Robot: 'robot', JointMotion: 'robot', Position: 'concept',
    Package: 'concept', SubPackage: 'klass', PythonClass: 'klass',
  };
  function groupOfType(t) { return TYPE_GROUP[t] || 'ind'; }
  const GROUP_COLOR = {
    root: '#e8eefb', klass: '#5b8cff', upper: '#8c9bbd',
    robot: '#ff7a9c', object: '#39d5c8', event: '#b98cff', goal: '#ffb648', concept: '#4bd38a', ind: '#7f8db0',
  };
  function groupColor(g) { return GROUP_COLOR[g] || '#5b8cff'; }
  function esc(s) { return String(s).replace(/[&<>]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]; }); }
});
