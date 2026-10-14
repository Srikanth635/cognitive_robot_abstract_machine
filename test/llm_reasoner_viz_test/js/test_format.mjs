// Tests of writing numbers for reading.
import test from 'node:test';
import assert from 'node:assert';
import { formatCost, formatCount, formatDuration, formatOffset } from '../../../llm_reasoner_viz/src/llm_reasoner_viz/web/format.mjs';

test('a count is written with its thousands set apart', function () {
  assert.strictEqual(formatCount(21323), '21,323');
  assert.strictEqual(formatCount(6), '6');
  assert.strictEqual(formatCount(0), '0');
});

test('a short time is written in milliseconds and a long one in minutes', function () {
  assert.strictEqual(formatDuration(0.0001), '<1 ms');
  assert.strictEqual(formatDuration(0.25), '250 ms');
  assert.strictEqual(formatDuration(4.4), '4.40 s');
  assert.strictEqual(formatDuration(125), '2 min 5 s');
});

test('a time into a run is written with a sign', function () {
  assert.strictEqual(formatOffset(1.5), '+1.50 s');
  assert.strictEqual(formatOffset(12.34), '+12.3 s');
});

test('a cost of a few cents keeps its fractions of a cent', function () {
  assert.strictEqual(formatCost(0.0394), '$0.0394');
  assert.strictEqual(formatCost(1.5), '$1.50');
});
