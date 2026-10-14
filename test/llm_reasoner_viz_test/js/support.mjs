// What the node tests of the page's modules share: the events of real runs and a model
// built from events.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { EVENT, KIND, OUTCOME, RunModel } from '../../../llm_reasoner_viz/src/llm_reasoner_viz/web/model.mjs';

const FIXTURES = path.join(path.dirname(fileURLToPath(import.meta.url)), 'fixtures');

// The events of a log, shortened, as it was written.
export function loadFixture(name) {
  return JSON.parse(fs.readFileSync(path.join(FIXTURES, name), 'utf8'));
}

export function modelOf(events) {
  const model = new RunModel();
  events.forEach(function (event) { model.add(event); });
  return model;
}

// An event written by hand, for what the logs do not hold.
export function eventOf(kind, at, fields) {
  return Object.assign({ event: kind, at: at }, fields || {});
}

export function spansOfKind(model, kind) {
  return model.spans.filter(function (span) { return span.kind === kind; });
}

function resolutionStartedBy(model, events, position) {
  const started = events.filter(function (event) { return event.event === EVENT.RESOLUTION_STARTED; })[position];
  return spansOfKind(model, KIND.RESOLUTION).find(function (span) {
    return span.data.started.resolution === started.resolution;
  });
}

// The resolution of the action asked for, which a run starts first.
export function actionResolution(model, events) {
  return resolutionStartedBy(model, events, 0);
}

// The choice of the situation, which the resolution of the action starts next.
export function situationResolution(model, events) {
  return resolutionStartedBy(model, events, 1);
}

// The outcome of a pour as the simulation reports it when it measured where the mouth ended.
export function pourOutcome(where) {
  return {
    [OUTCOME.LANDS_IN_OPENING]: Math.hypot(where.ahead, where.right) <= where.radius && where.above > 0,
    [OUTCOME.OPENING_RADIUS]: where.radius,
    [OUTCOME.DISTANCE]: Math.hypot(where.ahead, where.right),
    [OUTCOME.AHEAD]: where.ahead,
    [OUTCOME.RIGHT]: where.right,
    [OUTCOME.ABOVE]: where.above,
  };
}

// The events of pours tried one after the other, with the outcome of each.
export function pourEvents(outcomes) {
  return outcomes.flatMap(function (outcome, index) {
    return [
      eventOf(EVENT.TRIAL_STARTED, 10 * index + 1, { trial: 'pour' + index, values: { tilt_angle: 1 + index } }),
      eventOf(EVENT.TRIAL_FINISHED, 10 * index + 2, { trial: 'pour' + index, outcome: outcome, seconds: 1 }),
    ];
  });
}
