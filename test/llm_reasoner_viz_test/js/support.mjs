// What the node tests of the page's modules share: the events of real runs and a model
// built from events.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { EVENT, KIND, RunModel } from '../../../llm_reasoner_viz/src/llm_reasoner_viz/web/model.mjs';

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
