/* ============================================================================
 * core/panel-tabs.js — panels sharing one frame, one shown at a time.
 *
 * panel-groups.js names the groups: each group lists its panels, in the order of their tabs,
 * with the tab's label and when the panel comes forward besides a click on its tab:
 * opened by a query parameter of the page's address, or brought forward by a bus event.
 * install() runs right after the page has mounted its panels, and before the column's
 * panels are made resizable: the panels of a group move into a frame of their own, with
 * a tab bar above them in the stylesheet's tab classes. Each panel keeps its own chrome
 * and its own state. A group with a panel that is not mounted, or mounted in another
 * column than the rest, is left as it is.
 *
 * A panel is stepped aside with a class of its own, as the visibility menu toggles
 * .hidden on the same sections.
 * ==========================================================================*/
(function (global) {
  'use strict';

  const INACTIVE_CLASS = 'tab-inactive';

  /* The tab of the group that opens first: the first one named by the address, else the
     first of the group. */
  function startingTab(tabs, search) {
    const parameters = new URLSearchParams(search);
    const named = tabs.findIndex(function (tab) {
      return tab.openedByParameter && parameters.has(tab.openedByParameter);
    });
    return named < 0 ? 0 : named;
  }

  function frameOf(tabs) {
    const frame = document.createElement('section');
    frame.className = 'panel-tabs';
    frame.dataset.panel = 'tabs:' + tabs.map(function (tab) { return tab.panel; }).join('+');
    const bar = document.createElement('div');
    bar.className = 'tab-bar';
    const body = document.createElement('div');
    body.className = 'tab-body';
    frame.appendChild(bar);
    frame.appendChild(body);
    return { frame: frame, bar: bar, body: body };
  }

  function group(tabs, bus, search) {
    const sections = tabs.map(function (tab) {
      return document.querySelector('[data-panel="' + tab.panel + '"]');
    });
    if (sections.some(function (section) { return !section; })) return;
    const slot = sections[0].parentNode;
    if (sections.some(function (section) { return section.parentNode !== slot; })) return;

    const built = frameOf(tabs);
    slot.insertBefore(built.frame, sections[0]);
    const buttons = tabs.map(function (tab, position) {
      const button = document.createElement('button');
      button.textContent = tab.label;
      button.title = tab.title || '';
      button.addEventListener('click', function () { show(position); });
      built.bar.appendChild(button);
      built.body.appendChild(sections[position]);
      return button;
    });

    function show(position) {
      sections.forEach(function (section, other) {
        section.classList.toggle(INACTIVE_CLASS, other !== position);
        buttons[other].classList.toggle('active', other === position);
      });
    }

    show(startingTab(tabs, search));
    tabs.forEach(function (tab, position) {
      (tab.comesForwardOn || []).forEach(function (event) {
        bus.on(event, function () { show(position); });
      });
    });
  }

  /* Group the configured panels. Arguments default to the page's own. */
  function install(groups, bus, search) {
    const configured = groups || global.CRAMERA_TABS || [];
    configured.forEach(function (tabs) {
      group(tabs, bus || global.Bus, search === undefined ? global.location.search : search);
    });
  }

  global.PanelTabs = { INACTIVE_CLASS: INACTIVE_CLASS, startingTab: startingTab, install: install };
})(typeof window !== 'undefined' ? window : this);
