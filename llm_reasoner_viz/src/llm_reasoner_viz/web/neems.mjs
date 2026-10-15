// What the page knows about the NEEMs before it draws them: where to ask for them, which
// are shown, and how each is worded.

import { formatDuration } from './format.mjs';
import { NEEM_KIND, OUTCOME, STATUS, neemSpanId } from './model.mjs';

const NEEMS_ADDRESS = '/api/neems';
const DISTANCE_DECIMALS = 3;

// Whose NEEMs are listed: those of the run shown, or all that the database holds.
export const NEEM_SCOPE = Object.freeze({ RUN: 'run', ALL: 'all' });

// Which of the NEEMs listed are shown: all of them, or those of trials or of executions.
export const NEEM_FILTER = Object.freeze({
  ALL: 'all',
  TRIALS: NEEM_KIND.TRIAL,
  EXECUTIONS: NEEM_KIND.EXECUTION,
});

// How the root of a plan ended; the steps below it are stored as not started whatever
// happened, so the plan is only as told by its root.
export const PLAN_STATUS = Object.freeze({ SUCCEEDED: 'SUCCEEDED', FAILED: 'FAILED' });

// How an execution ended, as the server says it.
const EXECUTION_RESULT = Object.freeze({ FINISHED: 'finished', FAILED: 'failed' });

const PLAN_STANDING = Object.freeze({
  [PLAN_STATUS.SUCCEEDED]: STATUS.OK,
  [PLAN_STATUS.FAILED]: STATUS.FAILED,
});

const EXECUTION_STANDING = Object.freeze({
  [EXECUTION_RESULT.FINISHED]: STATUS.OK,
  [EXECUTION_RESULT.FAILED]: STATUS.FAILED,
});

// %% where to ask

// The address that lists the NEEMs of a run, or all of them.
export function neemsAddress(scope, run) {
  return scope === NEEM_SCOPE.RUN ? NEEMS_ADDRESS + '?' + new URLSearchParams({ run: run }) : NEEMS_ADDRESS;
}

// %% which are shown

// The NEEMs of the kind asked for; a NEEM that no log names is of neither kind.
export function shownNeems(entries, filter) {
  if (filter === NEEM_FILTER.ALL) return entries;
  return entries.filter(function (entry) { return entry.origin !== null && entry.origin.kind === filter; });
}

// How many NEEMs there are of each kind, and in all.
export function kindCounts(entries) {
  return {
    [NEEM_FILTER.ALL]: entries.length,
    [NEEM_FILTER.TRIALS]: shownNeems(entries, NEEM_FILTER.TRIALS).length,
    [NEEM_FILTER.EXECUTIONS]: shownNeems(entries, NEEM_FILTER.EXECUTIONS).length,
  };
}

// %% how each is worded

function plural(count, noun) {
  return count + ' ' + noun + (count === 1 ? '' : 's');
}

// How a pour ended, as {verdict, standing, distance}; distance is how far the mouth of the
// pitcher was from the centre of the opening, when that was measured.
function pourEnding(outcome) {
  if (outcome === null) return { verdict: null, standing: null, distance: null };
  if (outcome[OUTCOME.MOTION_FAILED]) return { verdict: 'motion failed', standing: STATUS.FAILED, distance: null };
  const landed = outcome[OUTCOME.LANDS_IN_OPENING];
  const measured = outcome[OUTCOME.DISTANCE];
  return {
    verdict: landed ? 'landed' : 'missed',
    standing: landed ? STATUS.OK : STATUS.REJECTED,
    distance: measured === undefined ? null : Number(measured.toFixed(DISTANCE_DECIMALS)),
  };
}

// How the NEEM came about and ended: {verdict, standing, distance}.
function ending(entry) {
  const origin = entry.origin;
  if (origin === null) return { verdict: entry.status.toLowerCase(), standing: PLAN_STANDING[entry.status] || null, distance: null };
  if (origin.kind === NEEM_KIND.TRIAL) return pourEnding(origin.outcome);
  return { verdict: origin.result, standing: EXECUTION_STANDING[origin.result] || null, distance: null };
}

// A NEEM written for reading: its number, what it is the plan of, how that ended, the facts
// of the plan, and where it came from (``run`` and ``spanId`` are null when no log says).
export function describeNeem(entry) {
  const { verdict, standing, distance } = ending(entry);
  const origin = entry.origin;
  return {
    identifier: entry.neem,
    title: '#' + entry.neem,
    origin: origin === null ? null : origin.kind,
    verdict: verdict,
    standing: standing,
    facts: [
      distance === null ? null : distance + ' m from the centre',
      plural(entry.steps, 'step'),
      plural(entry.nodes, 'node'),
      entry.seconds === null ? null : formatDuration(entry.seconds),
    ].filter(function (fact) { return fact !== null; }),
    recordedAt: origin === null ? null : origin.recorded_at,
    run: origin === null ? null : origin.run,
    spanId: origin === null ? null : neemSpanId(origin.kind, origin.episode),
  };
}
