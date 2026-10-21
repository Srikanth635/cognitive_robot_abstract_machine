/* ============================================================================
 * addon/designator_panel.js — resolving underspecified actions with llm_reasoner.
 *
 * The scene's predefined designators are krrood match queries; a click copies one into
 * the query box, where it can be edited before Resolve resolves it in the scene. The box
 * suggests the action designators and, inside one, its parameters, through cramera's own
 * suggestion menu, as the EQL box suggests names (POST
 * /api/resolution/resolve). What the resolver chose, why, what that rests on and the
 * pours it tried in the simulation are shown below. Execute performs the resolution's
 * run under the live viewer. Runs kept earlier are listed and shown again on a click, or
 * on a page opened with ?run=<name>, as the llm_reasoner viewer links to it.
 *
 * Bus events:
 *   emits  hypothesis:generated {payload}   the hypothesis a resolution consulted
 * ==========================================================================*/
(function () {
  'use strict';

  const PANEL_ID = 'resolution-designators';

  const ROUTES = Object.freeze({
    designators: '/api/resolution/designators',
    runs: '/api/resolution/runs',
    run: '/api/resolution/run',
    resolve: '/api/resolution/resolve',
    execute: '/api/resolution/execute',
  });

  Panels.define(PANEL_ID, function (root, bus) {
    const view = window.ResolutionView;
    root.innerHTML =
      '<div class="panel-head">' +
      '  <h2>Underspecified Action Designators</h2>' +
      '</div>' +
      '<div class="query-box">' +
      '  <textarea id="resolution-query" rows="3" spellcheck="false" placeholder="the(PouringAction)(…)   —  type for suggestions or pick a designator below, then Resolve"></textarea>' +
      '  <div class="query-actions">' +
      '    <button id="resolution-resolve" class="primary" title="Resolve the query in the box with llm_reasoner in this scene">Resolve</button>' +
      '    <button id="resolution-execute" title="Perform the resolved designator in the simulation, under the live viewer" disabled>Execute</button>' +
      '  </div>' +
      '  <div class="knowledge-choices">' +
      '    <div id="resolution-phase-structure" class="perspectives" title="The phase structure Resolve consults; untick it to leave it out"></div>' +
      '    <div id="resolution-perspectives" class="perspectives" title="Perspectives on the action core Resolve also consults; untick one to leave it out"></div>' +
      '    <div id="resolution-adjustment" class="perspectives" title="Whether a pour that misses may open the pour geometry the query leaves out; untick it to keep the query as it is"></div>' +
      '  </div>' +
      '  <div class="query-sources">' +
      '    <div class="source-tabs">' +
      '      <button id="resolution-tab-designators" title="Underspecified designators as krrood match queries: click one to copy it into the query box, then Resolve">Designators</button>' +
      '      <button id="resolution-tab-runs" title="Resolutions kept earlier: click one to show it again">Runs</button>' +
      '    </div>' +
      '    <div id="resolution-designators" class="designators"></div>' +
      '    <div id="resolution-runs" class="designators"></div>' +
      '  </div>' +
      '</div>' +
      '<div id="resolution-answer" class="answer"><div class="answer-empty">Pick a designator, then Resolve.</div></div>';

    const answerEl = root.querySelector('#resolution-answer');
    const queryEl = root.querySelector('#resolution-query');
    const resolveBtn = root.querySelector('#resolution-resolve');
    const executeBtn = root.querySelector('#resolution-execute');
    const designatorsEl = root.querySelector('#resolution-designators');
    const runsEl = root.querySelector('#resolution-runs');
    const phaseStructureEl = root.querySelector('#resolution-phase-structure');
    const perspectivesEl = root.querySelector('#resolution-perspectives');
    const adjustmentEl = root.querySelector('#resolution-adjustment');
    executeBtn.disabled = true;  // nothing to execute before a run is shown

    // ---- the box suggests the action designators and their parameters -------------------
    let vocabulary = [];  // what the server offers to complete
    const suggestions = EqlSuggestions.of({
      input: queryEl,
      anchor: root.querySelector('.query-box'),
      entries: function () {
        return DesignatorCompletion.entriesAt(vocabulary, queryEl.value, queryEl.selectionStart);
      },
      fetchMembers: function () { return Promise.resolve([]); },
    });
    queryEl.addEventListener('keydown', function (event) {
      // the menu owns the arrows, Tab, Escape and Enter while it is open
      if (suggestions.handledKey(event)) event.preventDefault();
    });

    // ---- the two lists below the query box, one shown at a time ------------------------
    const sources = [
      { tab: root.querySelector('#resolution-tab-designators'), list: designatorsEl },
      { tab: root.querySelector('#resolution-tab-runs'), list: runsEl },
    ];
    function showSource(shown) {
      sources.forEach(function (source) {
        source.tab.classList.toggle('on', source === shown);
        source.list.classList.toggle('hidden', source !== shown);
      });
    }
    sources.forEach(function (source) {
      source.tab.addEventListener('click', function () { showSource(source); });
    });
    showSource(sources[0]);

    // ---- the designators the scene offers --------------------------------------------
    async function loadDesignators() {
      try {
        const response = await fetch(SceneContext.withScene(ROUTES.designators));
        buildDesignators(await response.json());
      } catch (error) {
        designatorsEl.textContent = 'designators unavailable: ' + String(error);
      }
    }

    function buildDesignators(payload) {
      designatorsEl.innerHTML = '';
      if (!payload.ok) {
        designatorsEl.textContent = 'designators unavailable: ' + (payload.error || 'unknown error');
        return;
      }
      vocabulary = payload.action_vocabulary || [];
      buildChoice(phaseStructureEl, payload.phase_structure, '');
      buildChoice(adjustmentEl, payload.geometry_adjustment, '');
      perspectivesEl.innerHTML = '';
      (payload.perspectives || []).forEach(function (perspective) {
        perspectivesEl.appendChild(knowledgeChoice(perspective.label, perspective.summary, perspective.name));
      });
      payload.designators.forEach(function (designator) {
        const chip = document.createElement('pre');
        chip.className = 'designator-query' + (designator.available ? '' : ' unavailable');
        chip.textContent = designator.query_on_one_line;
        chip.title = designator.query + '\n' + view.designatorHint(designator);
        chip.addEventListener('click', function () { queryEl.value = designator.query; });
        designatorsEl.appendChild(chip);
      });
    }

    // ---- what Resolve consults: each choice a box, ticked by default -------------------
    function buildChoice(container, offered, value) {
      container.innerHTML = '';
      if (!offered) return;
      container.appendChild(knowledgeChoice(offered.label, offered.summary, value));
    }

    function knowledgeChoice(label, summary, value) {
      const choice = document.createElement('label');
      choice.className = 'perspective-choice';
      choice.title = label + ': ' + summary;
      const box = document.createElement('input');
      box.type = 'checkbox';
      box.value = value;
      box.checked = true;
      const text = document.createElement('span');
      text.textContent = label;
      choice.appendChild(box);
      choice.appendChild(text);
      return choice;
    }

    function leftOut(container) {
      const choice = container.children[0];
      return Boolean(choice) && !choice.children[0].checked;
    }

    function chosenPerspectives() {
      return Array.prototype.map.call(perspectivesEl.children, function (choice) { return choice.children[0]; })
        .filter(function (box) { return box.checked; })
        .map(function (box) { return box.value; });
    }

    function resolveRequest(query) {
      const request = { query: query, scene: SceneContext.name() };
      const perspectives = chosenPerspectives();
      if (perspectives.length) request.perspectives = perspectives;
      if (leftOut(phaseStructureEl)) request.phase_structure = false;
      if (leftOut(adjustmentEl)) request.adjust_geometry = false;
      return request;
    }

    // ---- resolving the query in the box ------------------------------------------------
    let resolving = false;
    let shownRun = null;

    async function postJson(route, body) {
      const response = await fetch(route, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      return response.json();
    }

    function failure(heading, query, message) {
      answerEl.innerHTML = '<div class="goal">' + view.esc(heading) + '</div>' +
        (query ? view.queryHtml(query) : '') +
        '<div class="qerr">' + view.esc(message) + '</div>';
    }

    async function resolve(query) {
      if (!(query || '').trim() || resolving) return;
      resolving = true;
      shownRun = null;
      executeBtn.disabled = true;
      resolveBtn.textContent = '…';
      answerEl.innerHTML =
        '<div class="goal">resolving</div>' + view.queryHtml(query) +
        '<p class="hint-txt">The resolver reads the scene and consults the knowledge ' +
        'component about the action. Each resolution is a run of its own from the same ' +
        'initial environment, generating the knowledge it needs, which takes a few minutes.</p>';
      try {
        const payload = await postJson(ROUTES.resolve, resolveRequest(query));
        if (payload.ok) showResolution(payload);
        else failure('resolution failed', query, payload.error || 'resolution failed');
      } catch (error) {
        failure('resolution failed', '', String(error));
      }
      answerEl.scrollTop = 0;  // a new answer is read from its top
      resolveBtn.textContent = 'Resolve';
      resolving = false;
      loadRuns();
    }

    resolveBtn.addEventListener('click', function () { resolve(queryEl.value); });

    // The answer a resolution gave, whether just made or opened from a run kept earlier:
    // the answer area, the hypothesis it consulted, and Execute for its run.
    function showResolution(payload) {
      answerEl.innerHTML = view.resolutionHtml(payload.resolution, payload.viewer);
      shownRun = payload.resolution.run;
      executeBtn.disabled = false;
      bus.emit('hypothesis:generated', payload.hypothesis);
    }

    // ---- runs kept earlier -------------------------------------------------------------
    async function loadRuns() {
      try {
        const response = await fetch(ROUTES.runs);
        buildRuns(await response.json());
      } catch (error) {
        runsEl.textContent = 'runs unavailable: ' + String(error);
      }
    }

    function buildRuns(payload) {
      runsEl.innerHTML = '';
      if (!payload.ok) {
        runsEl.textContent = 'runs unavailable: ' + (payload.error || 'unknown error');
        return;
      }
      if (!payload.runs.length) {
        runsEl.textContent = 'no run kept yet';
        return;
      }
      payload.runs.forEach(function (run) {
        const entry = document.createElement('pre');
        entry.className = 'designator-query';
        entry.textContent = view.runText(run);
        entry.title = view.runTitle(run);
        entry.addEventListener('click', function () { openRun(run.name); });
        runsEl.appendChild(entry);
      });
    }

    async function openRun(name) {
      answerEl.innerHTML = '<div class="goal">opening run ' + view.esc(name) + '</div>';
      try {
        const response = await fetch(
          SceneContext.withScene(ROUTES.run + '?name=' + encodeURIComponent(name)));
        const payload = await response.json();
        if (!payload.ok) {
          failure('run could not be opened', '', payload.error || 'unknown error');
          return;
        }
        queryEl.value = payload.resolution.query;
        showResolution(payload);
      } catch (error) {
        failure('run could not be opened', '', String(error));
      }
    }

    // ---- performing the run shown ------------------------------------------------------
    // Execute performs the shown run in the simulation; the 3D scene panel already polls the
    // live viewer's bridge and shows its own Live button the moment it answers.
    let executing = false;
    async function execute() {
      if (!shownRun || executing) return;
      executing = true;
      executeBtn.textContent = '…';
      executeBtn.disabled = true;
      try {
        const payload = await postJson(ROUTES.execute, { run: shownRun });
        if (payload.ok) {
          answerEl.innerHTML = '<div class="goal">executing run ' + view.esc(shownRun) + '</div>' +
            '<p class="hint-txt">Starting the simulation from the run\'s initial environment ' +
            'under the live viewer. Once it comes up, the 3D scene\'s own Live button will ' +
            'appear — click it to watch.</p>';
        } else {
          failure('execution failed', '', payload.error || 'execution failed');
        }
      } catch (error) {
        failure('execution failed', '', String(error));
      }
      executing = false;
      executeBtn.textContent = 'Execute';
      executeBtn.disabled = false;
    }

    executeBtn.addEventListener('click', execute);

    loadDesignators();
    loadRuns();
    const named = new URLSearchParams(window.location.search).get('run');
    if (named) openRun(named);

    return {};
  });

  // mounted right below the EQL panel, before the page boots them
  const layout = (window.CRAMERA_CONFIG || {}).layout || {};
  if (layout.right) layout.right.splice(layout.right.indexOf('eql') + 1, 0, PANEL_ID);
})();
