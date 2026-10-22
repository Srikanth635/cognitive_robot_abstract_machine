const BrowserSource = require('./browser_source');
// Unit tests for web/core/scene.js (node:test).
// SceneContext is how every panel learns which onboarded scene the URL names, so its
// API requests target the scene actually shown in the viewer instead of the server's
// fallback default.
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const WEB = path.join(__dirname, '..', '..', '..', 'cramera', 'src', 'cramera', 'web');

function loadWithSearch(search) {
  global.window = { location: { search: search } };
  new Function(BrowserSource.read(path.join(WEB, 'core/scene.js')))();
}

test('name() is null without a ?scene= param', function () {
  loadWithSearch('');
  assert.strictEqual(window.SceneContext.name(), null);
});

test('name() reads the ?scene= param', function () {
  loadWithSearch('?scene=g1-warehouse');
  assert.strictEqual(window.SceneContext.name(), 'g1-warehouse');
});

test('name() reads ?scene= alongside other params', function () {
  loadWithSearch('?foo=bar&scene=g1-warehouse&baz=qux');
  assert.strictEqual(window.SceneContext.name(), 'g1-warehouse');
});

test('withScene() leaves a url unchanged without an active scene', function () {
  loadWithSearch('');
  assert.strictEqual(window.SceneContext.withScene('/api/kb'), '/api/kb');
});

test('withScene() appends ?scene= to a bare url', function () {
  loadWithSearch('?scene=g1-warehouse');
  assert.strictEqual(window.SceneContext.withScene('/api/kb'), '/api/kb?scene=g1-warehouse');
});

test('withScene() appends &scene= to a url that already has a query string', function () {
  loadWithSearch('?scene=g1-warehouse');
  assert.strictEqual(
    window.SceneContext.withScene('/api/kb/view?name=plan'),
    '/api/kb/view?name=plan&scene=g1-warehouse'
  );
});

// %% showing the 3D scene alone
// The scene stands alone in two situations: another page embeds it in an iframe (the
// Plan Builder's ?scene view, the replay popup), or the page was opened in a window of
// its own with ?layout=scene — the pop-out a user drags onto a second screen.
test('a top-level page with ?scene= keeps the full layout', function () {
  loadWithSearch('?scene=g1-warehouse');
  assert.strictEqual(window.SceneContext.sceneOnly(false), false);
});

test('a framed page with ?scene= shows the scene alone', function () {
  loadWithSearch('?scene=g1-warehouse');
  assert.strictEqual(window.SceneContext.sceneOnly(true), true);
});

test('a framed replay popup shows the scene alone', function () {
  loadWithSearch('?replay=12');
  assert.strictEqual(window.SceneContext.sceneOnly(true), true);
});

test('a framed page without a scene or replay keeps the full layout', function () {
  loadWithSearch('');
  assert.strictEqual(window.SceneContext.sceneOnly(true), false);
});

test('?layout=scene shows the scene alone even at top level', function () {
  loadWithSearch('?scene=g1-warehouse&layout=scene');
  assert.strictEqual(window.SceneContext.sceneOnly(false), true);
});

test('the class the stylesheet styles for the scene alone is the one SceneContext names', function () {
  loadWithSearch('');
  const stylesheet = BrowserSource.read(path.join(WEB, 'app.css'));
  assert.ok(stylesheet.indexOf(':root.' + window.SceneContext.SCENE_ONLY_CLASS + ' ') >= 0);
});

test('the scene-only layout is named by the constant the pop-out url uses', function () {
  loadWithSearch('?layout=' + window.SceneContext.LAYOUT_SCENE);
  assert.strictEqual(window.SceneContext.sceneOnly(false), true);
});

// %% the popped-out window itself
// An embedded scene (Plan Builder, Sandbox) still offers the pop-out; only the window
// that already is the pop-out has nothing to pop out.
test('a framed embed with ?scene is not the popped-out window', function () {
  loadWithSearch('?scene');
  assert.strictEqual(window.SceneContext.poppedOut(), false);
});

test('?layout=scene is the popped-out window', function () {
  loadWithSearch('?scene=g1-warehouse&layout=scene');
  assert.strictEqual(window.SceneContext.poppedOut(), true);
});

test('the popped-out window is also scene-only', function () {
  loadWithSearch('?layout=scene');
  assert.strictEqual(window.SceneContext.sceneOnly(false), true);
  assert.strictEqual(window.SceneContext.poppedOut(), true);
});

test('the class the stylesheet hides the pop-out button under is the one SceneContext names', function () {
  loadWithSearch('');
  const stylesheet = BrowserSource.read(path.join(WEB, 'app.css'));
  assert.ok(stylesheet.indexOf(':root.' + window.SceneContext.POPPED_OUT_CLASS + ' .pop-out') >= 0);
  assert.strictEqual(stylesheet.indexOf(':root.' + window.SceneContext.SCENE_ONLY_CLASS + ' .pop-out'), -1);
});

// %% the pop-out window's url
test('the pop-out url opens the active scene alone', function () {
  loadWithSearch('?scene=g1-warehouse');
  assert.strictEqual(window.SceneContext.popOutUrl(), 'index.html?scene=g1-warehouse&layout=scene');
});

test('the pop-out url without an active scene lets the viewer pick its default', function () {
  loadWithSearch('');
  assert.strictEqual(window.SceneContext.popOutUrl(), 'index.html?layout=scene');
});

// %% the graph panel's pop-out
// The graph panel, with its knowledge tabs, pops out into a window of its own the same way
// the scene does, with ?layout=graph.
test('?layout=graph is the popped-out graph window', function () {
  loadWithSearch('?scene=g1-warehouse&layout=graph');
  assert.strictEqual(window.SceneContext.poppedOutGraph(), true);
});

test('a page without ?layout=graph is not the popped-out graph window', function () {
  loadWithSearch('?scene=g1-warehouse');
  assert.strictEqual(window.SceneContext.poppedOutGraph(), false);
});

test('the popped-out graph window does not show the scene alone', function () {
  loadWithSearch('?layout=graph');
  assert.strictEqual(window.SceneContext.sceneOnly(false), false);
  assert.strictEqual(window.SceneContext.poppedOut(), false);
});

test('the popped-out scene window is not the graph window', function () {
  loadWithSearch('?layout=scene');
  assert.strictEqual(window.SceneContext.poppedOutGraph(), false);
});

test('the graph pop-out url opens the active scene with the graph panel alone', function () {
  loadWithSearch('?scene=g1-warehouse');
  assert.strictEqual(
    window.SceneContext.popOutUrl(window.SceneContext.LAYOUT_GRAPH),
    'index.html?scene=g1-warehouse&layout=graph'
  );
});

test('the graph pop-out url without an active scene lets the viewer pick its default', function () {
  loadWithSearch('');
  assert.strictEqual(
    window.SceneContext.popOutUrl(window.SceneContext.LAYOUT_GRAPH), 'index.html?layout=graph'
  );
});

test('the class the stylesheet styles for the graph alone is the one SceneContext names', function () {
  loadWithSearch('');
  const stylesheet = BrowserSource.read(path.join(WEB, 'app.css'));
  assert.ok(stylesheet.indexOf(':root.' + window.SceneContext.GRAPH_ONLY_CLASS + ' ') >= 0);
});
