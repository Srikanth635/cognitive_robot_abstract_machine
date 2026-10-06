// Unit tests for the browser scene-selection contract.
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const WEB = path.join(__dirname, '..', '..', '..', 'cram_viz', 'src', 'cram_viz', 'web');

function loadSceneSelection() {
  global.window = {};
  new Function(fs.readFileSync(path.join(WEB, 'core/scene-selection.js'), 'utf8'))();
  return global.window.SceneSelection;
}

test('selected scene is appended to knowledge API requests', function () {
  const SceneSelection = loadSceneSelection();

  assert.strictEqual(
    SceneSelection.apiUrl('/api/kb', '?scene=simple_pouring'),
    '/api/kb?scene=simple_pouring',
  );
  assert.strictEqual(
    SceneSelection.apiUrl('/api/kb/view?name=plan', '?scene=simple_pouring'),
    '/api/kb/view?name=plan&scene=simple_pouring',
  );
});

test('missing or invalid scene names are not forwarded', function () {
  const SceneSelection = loadSceneSelection();

  assert.strictEqual(SceneSelection.apiUrl('/api/kb', ''), '/api/kb');
  assert.strictEqual(
    SceneSelection.apiUrl('/api/kb', '?scene=../../private'),
    '/api/kb',
  );
});
