/* ============================================================================
 * addon/panel_tabs.js — the EQL box and the underspecified action designators, as two tabs
 * of one frame, one shown at a time.
 *
 * Right after the page has mounted its panels, and before it makes the column's panels
 * resizable, the two panels move into a frame of their own, with a tab bar above them in
 * cramera's stylesheet's tab classes. Each panel keeps its own chrome and its own state.
 * A page opened on a resolution (?run=) starts on the designators. A graph node or a part
 * of the scene clicked is described by the EQL panel, so the EQL tab comes forward.
 *
 * Bus events:
 *   listens  entity:select {id, detail, relations}   a graph node was clicked
 *   listens  scene:part-clicked {id}                 a part of the scene was clicked
 * ==========================================================================*/
(function () {
  'use strict';

  const FRAME_ID = 'resolution-tabs';

  const TABS = Object.freeze([
    { panel: 'eql', label: 'EQL Query', title: 'ask the knowledge base a question in EQL' },
    {
      panel: 'resolution-designators', label: 'Underspecified Action Designators',
      title: 'resolve an underspecified action with llm_reasoner',
    },
  ]);

  function group(bus) {
    const sections = TABS.map(function (tab) {
      return document.querySelector('[data-panel="' + tab.panel + '"]');
    });
    // a layout that mounts only one of them has nothing to share a frame
    if (sections.some(function (section) { return !section; })) return;
    const slot = sections[0].parentNode;
    if (sections.some(function (section) { return section.parentNode !== slot; })) return;

    const frame = document.createElement('section');
    frame.className = 'panel-tabs';
    frame.dataset.panel = FRAME_ID;
    const bar = document.createElement('div');
    bar.className = 'tab-bar';
    const body = document.createElement('div');
    body.className = 'tab-body';
    frame.appendChild(bar);
    frame.appendChild(body);
    slot.insertBefore(frame, sections[0]);

    const buttons = TABS.map(function (tab, position) {
      const button = document.createElement('button');
      button.textContent = tab.label;
      button.title = tab.title;
      button.addEventListener('click', function () { show(position); });
      bar.appendChild(button);
      body.appendChild(sections[position]);
      return button;
    });

    // a class of our own: the page's visibility menu toggles .hidden on these sections
    function show(position) {
      sections.forEach(function (section, other) {
        section.classList.toggle('tab-inactive', other !== position);
        buttons[other].classList.toggle('active', other === position);
      });
    }

    show(new URLSearchParams(window.location.search).has('run') ? 1 : 0);
    ['entity:select', 'scene:part-clicked'].forEach(function (event) {
      bus.on(event, function () { show(0); });
    });
  }

  // the page boots its panels in one call, and sizes the column's panels right after it
  const boot = Panels.boot;
  Panels.boot = function () {
    boot.apply(Panels, arguments);
    group(window.Bus);
  };
  window.ResolutionPanelTabs = { group: group };
})();
