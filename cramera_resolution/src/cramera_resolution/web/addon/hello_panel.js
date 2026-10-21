/* ============================================================================
 * addon/hello_panel.js — a panel the add-on mounts into cramera's page, to show that
 * a panel can be added without editing cramera: it is defined through Panels.define
 * and listed in the layout before the page boots its panels.
 * ==========================================================================*/
(function () {
  'use strict';

  const PANEL_ID = 'resolution-hello';

  Panels.define(PANEL_ID, function (root) {
    root.innerHTML =
      '<div class="panel-title">Resolution add-on</div>' +
      '<div id="resolution-hello-status">asking the add-on server…</div>';
    const status = root.querySelector('#resolution-hello-status');
    fetch('/api/resolution/ping')
      .then(function (response) { return response.json(); })
      .then(function (answer) { status.textContent = 'add-on server: ' + answer.addon; })
      .catch(function (error) { status.textContent = 'add-on server unavailable: ' + error; });
    return {};
  });

  const layout = (window.CRAMERA_CONFIG || {}).layout || {};
  if (layout.right) layout.right.push(PANEL_ID);
})();
