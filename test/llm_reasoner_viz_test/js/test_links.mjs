// Tests of the address that opens a run in cramera: its scene, and the run to open.
import test from 'node:test';
import assert from 'node:assert';
import { crameraAddress } from '../../../llm_reasoner_viz/src/llm_reasoner_viz/web/links.mjs';

test('the address opens the scene and the run in cramera', function () {
  assert.strictEqual(crameraAddress('http://localhost:8711', 'pouring', 'abc'), 'http://localhost:8711/?scene=pouring&run=abc');
});

test('a closing slash on where cramera is makes no difference', function () {
  assert.strictEqual(crameraAddress('http://localhost:8711/', 'pouring', 'abc'), crameraAddress('http://localhost:8711', 'pouring', 'abc'));
});

test('where cramera is may have a path of its own', function () {
  assert.strictEqual(crameraAddress('http://host/cram', 'pouring', 'abc'), 'http://host/cram/?scene=pouring&run=abc');
});

test('a run of a log that names no scene opens in the scene cramera starts with', function () {
  assert.strictEqual(crameraAddress('http://localhost:8711', '', 'abc'), 'http://localhost:8711/?run=abc');
  assert.strictEqual(crameraAddress('http://localhost:8711', null, 'abc'), 'http://localhost:8711/?run=abc');
});

test('names that are not plain are written so that cramera reads them back', function () {
  const address = new URL(crameraAddress('http://localhost:8711', 'a&b', 'c d'));
  assert.strictEqual(address.searchParams.get('scene'), 'a&b');
  assert.strictEqual(address.searchParams.get('run'), 'c d');
});
