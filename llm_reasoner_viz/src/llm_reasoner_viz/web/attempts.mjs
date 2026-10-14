// The cards of the attempts of a resolution: what each chose, why, how it ended, and the
// pour that was tried for it.

import { trialText } from './inspector.mjs';
import { KIND } from './model.mjs';
import { isJson } from './prompts.mjs';

// The keys of an entry of the resolver's answer that are not a slot.
const ANSWER = Object.freeze({ WHY: 'why', GROUNDS: 'grounds' });

// The entry of an answer that fills the slots, or null when the answer is not data.
function entryOf(answer) {
  if (answer === null || !isJson(answer)) return null;
  const value = JSON.parse(answer);
  const entry = Array.isArray(value) ? value[0] : value;
  return entry !== null && typeof entry === 'object' ? entry : null;
}

// What an attempt answered: the answer it was rejected with, or else the last the model gave
// while it went on.
function answerOf(model, attempt) {
  if (attempt.data.rejected) return attempt.data.rejected.answer;
  const answered = model.childrenOf(attempt).filter(function (span) { return span.kind === KIND.MODEL && span.data.answered; });
  return answered.length === 0 ? null : answered[answered.length - 1].data.answered.answer;
}

function cardOf(model, attempt) {
  const entry = entryOf(answerOf(model, attempt));
  const trial = model.childrenOf(attempt).find(function (span) { return span.kind === KIND.TRIAL; });
  return {
    id: attempt.id,
    label: attempt.label,
    status: attempt.status,
    chose: entry === null ? [] : Object.keys(entry)
      .filter(function (key) { return key !== ANSWER.WHY && key !== ANSWER.GROUNDS; })
      .map(function (key) { return { slot: key.split('.').pop(), value: entry[key] }; }),
    why: entry === null ? '' : entry[ANSWER.WHY] || '',
    rejectedFor: attempt.data.rejected ? attempt.data.rejected.kind : null,
    outcome: trial && trial.data.finished ? trialText(trial.data.finished.outcome) : null,
  };
}

export function attemptCards(model, resolution) {
  if (resolution === null) return [];
  return model.childrenOf(resolution)
    .filter(function (span) { return span.kind === KIND.ATTEMPT; })
    .map(function (attempt) { return cardOf(model, attempt); });
}

// The resolution whose attempts to show: the one the chosen span belongs to, else the first
// of the run, which is the one that was asked for. Null when the run resolved nothing.
export function resolutionOfChoice(model, chosenId) {
  for (let span = model.byId.get(chosenId); span; span = model.byId.get(span.parent)) {
    if (span.kind === KIND.RESOLUTION) return span;
  }
  return model.spans.find(function (span) { return span.kind === KIND.RESOLUTION; }) || null;
}

// The card of the attempt a span was part of, or null when it was part of none.
export function cardOfChoice(model, cards, chosenId) {
  for (let span = model.byId.get(chosenId); span; span = model.byId.get(span.parent)) {
    const card = cards.find(function (candidate) { return candidate.id === span.id; });
    if (card) return card.id;
  }
  return null;
}
