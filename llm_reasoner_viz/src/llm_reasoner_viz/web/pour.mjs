// Where pours ended, drawn as a plan of the floor around the opening of the target seen from
// above and from the robot's base: the opening is a circle, each pour a point, ahead of the
// robot is up and to its right is right.

import { KIND, OUTCOME } from './model.mjs';

// The side of the plan, in pixels.
export const PLOT_SIZE = 220;

// The plan reaches at least this many radii of the opening from its centre, and this much
// further than the furthest pour.
const OPENING_RADII = 1.8;
const BEYOND_THE_FURTHEST = 1.2;

function isNumber(value) {
  return typeof value === 'number';
}

// Whether a pour was measured: the motion did not fail and the mouth has a place.
function hasPlace(outcome) {
  return isNumber(outcome[OUTCOME.AHEAD]) && isNumber(outcome[OUTCOME.RIGHT]) && isNumber(outcome[OUTCOME.OPENING_RADIUS]);
}

// The plan of the pours that were measured; ``trials`` are ``{id, label, outcome}``. Null
// when none was.
export function pourPlot(trials, size) {
  const placed = trials.filter(function (trial) { return hasPlace(trial.outcome); });
  if (placed.length === 0) return null;
  const radius = placed[0].outcome[OUTCOME.OPENING_RADIUS];
  const furthest = Math.max(...placed.map(function (trial) {
    return Math.hypot(trial.outcome[OUTCOME.AHEAD], trial.outcome[OUTCOME.RIGHT]);
  }));
  const reach = Math.max(radius * OPENING_RADII, furthest * BEYOND_THE_FURTHEST);
  const scale = size / 2 / reach;
  return {
    size: size,
    scale: scale,
    opening: { x: size / 2, y: size / 2, radius: radius * scale },
    points: placed.map(function (trial) {
      return {
        id: trial.id,
        label: trial.label,
        x: size / 2 + trial.outcome[OUTCOME.RIGHT] * scale,
        y: size / 2 - trial.outcome[OUTCOME.AHEAD] * scale,
        lands: trial.outcome[OUTCOME.LANDS_IN_OPENING],
        above: trial.outcome[OUTCOME.ABOVE],
      };
    }),
  };
}

// The pours tried during the attempts of a resolution that have ended, in the order they were
// tried.
export function triedPours(model, resolution) {
  if (resolution === null) return [];
  return model.childrenOf(resolution)
    .filter(function (span) { return span.kind === KIND.ATTEMPT; })
    .flatMap(function (attempt) { return model.childrenOf(attempt); })
    .filter(function (span) { return span.kind === KIND.TRIAL && span.data.finished; });
}

// The plan of the pours of some trial spans that have ended; null when none was measured.
export function plotOfTrials(trials) {
  return pourPlot(trials.map(function (trial) {
    return { id: trial.id, label: trial.label, outcome: trial.data.finished.outcome };
  }), PLOT_SIZE);
}
