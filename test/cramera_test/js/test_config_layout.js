const BrowserSource = require('./browser_source');
// Unit tests for web/config.js (node:test): which panels the page mounts in each of the
// layouts the URL can ask for, and the classes the stylesheet styles them by.
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const path = require('path');

const WEB = path.join(__dirname, '..', '..', '..', 'cramera', 'src', 'cramera', 'web');

// the page's own scene.js and config.js, in a top-level window over a document that records
// the root element's classes
function configuredBy(search) {
  const classes = new Set();
  const scope = { location: { search: search } };
  scope.self = scope;
  scope.top = scope;
  const documentStub = {
    documentElement: { classList: { add(name) { classes.add(name); } } },
  };
  new Function('window', BrowserSource.read(path.join(WEB, 'core/scene.js')))(scope);
  const config = new Function(
    'window', 'document', 'SceneContext',
    BrowserSource.read(path.join(WEB, 'config.js')) + '; return window.CRAMERA_CONFIG;'
  )(scope, documentStub, scope.SceneContext);
  return { layout: config.layout, classes: classes };
}

test('the full page mounts the scene on the left and the EQL, designator and graph panels on the right', function () {
  const { layout } = configuredBy('?scene=g1-warehouse');
  assert.deepStrictEqual(layout, {
    left: ['robot-scene'],
    right: ['eql', 'resolution-designators', 'graph'],
  });
});

test('the popped-out scene window mounts the scene alone', function () {
  const { layout } = configuredBy('?layout=scene');
  assert.deepStrictEqual(layout, { left: ['robot-scene'] });
});

test('the popped-out graph window mounts the graph panel alone', function () {
  const { layout } = configuredBy('?layout=graph');
  assert.deepStrictEqual(layout, { left: ['graph'] });
});

test('the popped-out graph window is styled as the graph alone and as popped out', function () {
  const { classes } = configuredBy('?layout=graph');
  assert.deepStrictEqual([...classes].sort(), ['graph-only', 'popped-out']);
});

test('the full page carries no layout class', function () {
  const { classes } = configuredBy('?scene=g1-warehouse');
  assert.deepStrictEqual([...classes], []);
});
