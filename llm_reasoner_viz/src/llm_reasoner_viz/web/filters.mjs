// Picking out the spans of a run that matter: those that say some words, and those that
// went wrong.

import { STATUS } from './model.mjs';

const PROBLEM_STATUSES = Object.freeze([STATUS.REJECTED, STATUS.FAILED]);

// What a span says, for searching: its name and everything its events hold, prompts and
// answers included.
function textOf(span) {
  return [span.label, span.kind, span.lane, span.status, JSON.stringify(span.events)].join('\n').toLowerCase();
}

export class SpanFilter {
  constructor() {
    // Words the span must say; capital letters do not matter.
    this.text = '';
    // Whether only the spans that were rejected or failed are wanted.
    this.problemsOnly = false;
  }

  get active() {
    return this.text.trim() !== '' || this.problemsOnly;
  }

  matches(span) {
    if (this.problemsOnly && !PROBLEM_STATUSES.includes(span.status)) return false;
    const words = this.text.trim().toLowerCase();
    return words === '' || textOf(span).includes(words);
  }

  matching(model) {
    return model.spans.filter((span) => this.matches(span));
  }

  // The id of the match that starts after a span, going round to the first after the last;
  // null when nothing matches.
  next(model, afterId) {
    const matching = this.matching(model).sort(function (first, second) { return first.start - second.start; });
    if (matching.length === 0) return null;
    const after = afterId === null ? undefined : model.byId.get(afterId);
    if (after === undefined) return matching[0].id;
    const position = matching.indexOf(after);
    if (position >= 0) return matching[(position + 1) % matching.length].id;
    const later = matching.find(function (span) { return span.start > after.start; });
    return (later || matching[0]).id;
  }
}
