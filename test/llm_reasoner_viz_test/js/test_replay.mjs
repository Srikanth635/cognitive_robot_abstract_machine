// Tests of replaying a run: what it looked like at any moment, and how a replay moves.
import test from 'node:test';
import assert from 'node:assert';
import { EVENT, KIND } from '../../../llm_reasoner_viz/src/llm_reasoner_viz/web/model.mjs';
import { DEFAULT_SPEED, Replay, SPEEDS } from '../../../llm_reasoner_viz/src/llm_reasoner_viz/web/replay.mjs';
import { loadFixture, modelOf, spansOfKind } from './support.mjs';

const resolved = loadFixture('resolved_run.json');

function snapshot(model) {
  return model.spans.map(function (span) { return [span.id, span.status, span.start, span.end]; });
}

test('a replay lasts from the first event of the run to the last', function () {
  const replay = new Replay(resolved);
  assert.strictEqual(replay.start, resolved[0].at);
  assert.strictEqual(replay.end, resolved[resolved.length - 1].at);
  assert.strictEqual(replay.duration, replay.end - replay.start);
});

test('a replay begins at the start, paused, at its usual speed', function () {
  const replay = new Replay(resolved);
  assert.strictEqual(replay.position, 0);
  assert.strictEqual(replay.playing, false);
  assert.strictEqual(replay.speed, DEFAULT_SPEED);
  assert.ok(SPEEDS.includes(DEFAULT_SPEED));
});

test('at the start only what was written first has happened', function () {
  const replay = new Replay(resolved);
  const model = replay.modelAt();
  assert.strictEqual(model.events.length, 1);
  assert.strictEqual(spansOfKind(model, KIND.RUN)[0].end, null);
});

test('at the end everything has happened, as in the model of the whole run', function () {
  const replay = new Replay(resolved);
  replay.seek(replay.duration);
  assert.deepStrictEqual(snapshot(replay.modelAt()), snapshot(modelOf(resolved)));
});

test('what is still going on at a moment is open, and ended later', function () {
  const replay = new Replay(resolved);
  const called = resolved.find(function (event) { return event.event === EVENT.COMPONENT_CALLED && event.operation === 'serve'; });
  replay.seek(called.at - replay.start + 1.0);
  const model = replay.modelAt();
  assert.strictEqual(model.byId.get('call:' + called.call).end, null);
  assert.ok(model.spans.length < modelOf(resolved).spans.length);
});

test('going back shows what a run looked like then, not what it looked like since', function () {
  const replay = new Replay(resolved);
  replay.seek(1.0);
  const before = snapshot(replay.modelAt());
  replay.seek(replay.duration);
  replay.seek(1.0);
  assert.deepStrictEqual(snapshot(replay.modelAt()), before);
});

test('a moment before the start or after the end is the start or the end', function () {
  const replay = new Replay(resolved);
  replay.seek(-5);
  assert.strictEqual(replay.position, 0);
  replay.seek(replay.duration + 5);
  assert.strictEqual(replay.position, replay.duration);
  assert.ok(replay.atEnd);
});

test('a replay that is paused stays where it is', function () {
  const replay = new Replay(resolved);
  replay.advance(1);
  assert.strictEqual(replay.position, 0);
});

test('a replay that is playing moves by the time that passed at its speed', function () {
  const replay = new Replay(resolved);
  replay.speed = 4;
  replay.play();
  replay.advance(0.25);
  assert.strictEqual(replay.position, 1.0);
  assert.ok(replay.playing);
});

test('a replay that reaches the end stops', function () {
  const replay = new Replay(resolved);
  replay.play();
  replay.advance(100);
  assert.strictEqual(replay.position, replay.duration);
  assert.strictEqual(replay.playing, false);
});

test('playing from the end starts again from the start', function () {
  const replay = new Replay(resolved);
  replay.seek(replay.duration);
  replay.play();
  assert.strictEqual(replay.position, 0);
  assert.ok(replay.playing);
});

test('a step goes to the moment of the next event, and of the one before', function () {
  const replay = new Replay(resolved);
  const moments = Array.from(new Set(resolved.map(function (event) { return event.at; }))).sort(function (a, b) { return a - b; });
  replay.step(1);
  assert.strictEqual(replay.time, moments[1]);
  replay.seek(replay.duration);
  replay.step(-1);
  assert.strictEqual(replay.time, moments[moments.length - 2]);
});

test('a step at either end stays there', function () {
  const replay = new Replay(resolved);
  replay.step(-1);
  assert.strictEqual(replay.position, 0);
  replay.seek(replay.duration);
  replay.step(1);
  assert.ok(replay.atEnd);
});

test('a step pauses a replay that was playing', function () {
  const replay = new Replay(resolved);
  replay.play();
  replay.step(1);
  assert.strictEqual(replay.playing, false);
});

test('the events of a log without dates make a replay of no length', function () {
  const replay = new Replay([{ event: 'Unreadable', line: 'half a line' }]);
  assert.strictEqual(replay.duration, 0);
  assert.strictEqual(replay.modelAt().spans.length, 0);
});
