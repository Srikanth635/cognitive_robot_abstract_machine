// Tests of the address that opens a run in cram-viz: its scene, and the run to open.
import test from 'node:test';
import assert from 'node:assert';
import { cramVizAddress } from '../../../llm_reasoner_viz/src/llm_reasoner_viz/web/links.mjs';

test('the address opens the scene and the run in cram-viz', function () {
  assert.strictEqual(cramVizAddress('http://localhost:8711', 'pouring', 'abc'), 'http://localhost:8711/?scene=pouring&run=abc');
});

test('a closing slash on where cram-viz is makes no difference', function () {
  assert.strictEqual(cramVizAddress('http://localhost:8711/', 'pouring', 'abc'), cramVizAddress('http://localhost:8711', 'pouring', 'abc'));
});

test('where cram-viz is may have a path of its own', function () {
  assert.strictEqual(cramVizAddress('http://host/cram', 'pouring', 'abc'), 'http://host/cram/?scene=pouring&run=abc');
});

test('a run of a log that names no scene opens in the scene cram-viz starts with', function () {
  assert.strictEqual(cramVizAddress('http://localhost:8711', '', 'abc'), 'http://localhost:8711/?run=abc');
  assert.strictEqual(cramVizAddress('http://localhost:8711', null, 'abc'), 'http://localhost:8711/?run=abc');
});

test('names that are not plain are written so that cram-viz reads them back', function () {
  const address = new URL(cramVizAddress('http://localhost:8711', 'a&b', 'c d'));
  assert.strictEqual(address.searchParams.get('scene'), 'a&b');
  assert.strictEqual(address.searchParams.get('run'), 'c d');
});
