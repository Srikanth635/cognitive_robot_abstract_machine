// Tests of laying spans out on a timeline: where each lies and which row it takes.
import test from 'node:test';
import assert from 'node:assert';
import { packTracks, fitScale, xOf } from '../../../llm_reasoner_viz/src/llm_reasoner_viz/web/layout.mjs';

function span(id, start, end) { return { id: id, start: start, end: end }; }
const settings = { origin: 100, pxPerSecond: 10, minWidth: 4, gap: 2, now: 200 };

test('a span lies where its time is, as wide as it lasts', function () {
  const [placed] = packTracks([span('a', 101, 103)], settings);
  assert.strictEqual(placed.x, 10);
  assert.strictEqual(placed.width, 20);
  assert.strictEqual(placed.track, 0);
});

test('a span too short to see is given a width it can be seen and pressed at', function () {
  const [placed] = packTracks([span('a', 101, 101.0001)], settings);
  assert.strictEqual(placed.width, 4);
});

test('spans that do not meet share a row', function () {
  const placed = packTracks([span('a', 100, 101), span('b', 102, 103)], settings);
  assert.deepStrictEqual(placed.map(function (p) { return p.track; }), [0, 0]);
});

test('spans that overlap take rows of their own', function () {
  const placed = packTracks([span('outer', 100, 110), span('inner', 101, 102)], settings);
  assert.deepStrictEqual(placed.map(function (p) { return p.track; }), [0, 1]);
});

test('spans too short to tell apart do not sit on one another', function () {
  const placed = packTracks([span('a', 101, 101.001), span('b', 101.01, 101.011)], settings);
  assert.deepStrictEqual(placed.map(function (p) { return p.track; }), [0, 1]);
});

test('a row is taken again once what was in it has ended', function () {
  const placed = packTracks([span('a', 100, 105), span('b', 101, 102), span('c', 103, 104)], settings);
  assert.deepStrictEqual(placed.map(function (p) { return p.track; }), [0, 1, 1]);
});

test('a span still open lasts until now', function () {
  const [placed] = packTracks([span('a', 100, null)], settings);
  assert.strictEqual(placed.width, 1000);
});

test('the rows are given in the order the spans start', function () {
  const placed = packTracks([span('late', 105, 106), span('early', 100, 101)], settings);
  assert.deepStrictEqual(placed.map(function (p) { return p.span.id; }), ['early', 'late']);
});

test('a run is scaled to fit the width it is shown in', function () {
  assert.strictEqual(fitScale(100, 1000), 10);
});

test('a run is never scaled so far down that a second is lost', function () {
  assert.strictEqual(fitScale(100000, 1000, 2), 2);
});

test('a run of no length is scaled to a width of its own', function () {
  assert.strictEqual(fitScale(0, 1000, 2), 1000);
});

test('a time is a distance from the origin', function () {
  assert.strictEqual(xOf(103, 100, 10), 30);
});
