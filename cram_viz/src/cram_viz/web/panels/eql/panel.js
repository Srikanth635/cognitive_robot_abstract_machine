/* ============================================================================
 * panels/eql/panel.js — the EQL (Entity Query Language) console.
 *
 * Query box + presets + answer panel. Queries are executed server-side
 * (POST /api/eql) against the episode knowledge base; the KB overview
 * (GET /api/kb) provides presets and per-entity details.
 *
 * Bus events:
 *   emits    kb:ready {payload}          the /api/kb overview, once loaded
 *   emits    entity:highlight {ids, focus?}   results / described entity
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
    '  <div class="nl-row">' +
    '    <input id="nl-input" spellcheck="false" placeholder="Enter the instruction">' +
    '    <button id="nl-run" title="Generate a hypothesis for this instruction and show it in the Hypothesis Graph tab">Generate</button>' +
    '    <button id="nl-execute" title="Launch the demo under the live bridge — hardcoded for now, independent of the instruction above">Execute</button>' +
    '  </div>' +
    '  <div class="query-row">' +
    '    <textarea id="query-input" rows="2" spellcheck="false" placeholder="the(entity(obj).where(obj.name == \'milk\'))   —  vars: obj, ep, arm, j, rob"></textarea>' +
    '    <button id="query-run">Run</button>' +
    '  </div>' +
    '  <div id="presets" class="presets"></div>' +
    '</div>' +
    '<div id="answer" class="answer"><div class="answer-empty">Loading the episode knowledge base…</div></div>';

  const kbStatus = root.querySelector('#kb-status');
  const answerEl = root.querySelector('#answer');
  const input = root.querySelector('#query-input');
  const runBtn = root.querySelector('#query-run');
  const presetsEl = root.querySelector('#presets');
  const nlInput = root.querySelector('#nl-input');
  const nlRunBtn = root.querySelector('#nl-run');
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

  // ---- generate knowledge from a natural-language instruction ------------------
  // The instruction goes to llm_reasoner, which produces a Flanagan structure; the
  // server projects it with the action-frame and image-schema representations and
  // returns the refreshed graph. Generation replaces what every EQL variable ranges
  // over, so the presets are reloaded alongside it.
  let generating = false;
  async function generate(instruction) {
    instruction = (instruction || '').trim();
    if (!instruction || generating) return;
    generating = true;
    nlRunBtn.textContent = '…';
    nlRunBtn.disabled = true;
    answerEl.innerHTML =
      '<div class="goal">generating · ' + esc(instruction) + '</div>' +
      '<p class="hint-txt">Asking the model for the stereotypical structure of this ' +
      'action. A repeated instruction is served from the store instead.</p>';
    try {
      const response = await fetch('/api/hypothesis/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ instruction: instruction }),
      });
      const payload = await response.json();
      if (!payload.ok) {
        answerEl.innerHTML =
          '<div class="goal">generation failed</div>' +
          '<div class="qerr">' + esc(payload.error || 'generation failed') + '</div>';
      } else {
        bus.emit('hypothesis:generated', payload);
        answerEl.innerHTML =
          '<div class="goal">generated · ' + esc(instruction) + '</div>' +
          '<p class="headline"><b>' + (payload.nodes || []).length + '</b> nodes across ' +
          'the phase structure, action frame and image-schema profile.</p>' +
          '<p class="hint-txt">Shown in the Hypothesis Graph tab. Every pouring_* EQL ' +
          'variable now ranges over this structure.</p>';
        await reloadKnowledgeBase();
      }
    } catch (err) {
      answerEl.innerHTML =
        '<div class="goal">generation failed</div>' +
        '<div class="qerr">' + esc(String(err)) + '</div>';
    }
    generating = false;
    nlRunBtn.textContent = 'Generate';
    nlRunBtn.disabled = false;
  }

  async function reloadKnowledgeBase() {
    const response = await fetch(SceneSelection.apiUrl('/api/kb', window.location.search));
    const payload = await response.json();
    if (!payload.ok) return;
    kb = payload;
    buildPresets(payload.presets || []);
    bus.emit('kb:ready', { payload: payload });
  }

  nlRunBtn.addEventListener('click', function () { generate(nlInput.value); });
  nlInput.addEventListener('keydown', function (event) {
    if (event.key === 'Enter') { event.preventDefault(); generate(nlInput.value); }
  });

  // ---- execute the hardcoded demo under the live bridge -------------------------
  // Independent of the instruction/generated hypothesis above -- for now this
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
