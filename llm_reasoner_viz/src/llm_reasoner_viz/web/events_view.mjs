// Lists the events of a run as they are written, one row each: the record the other views
// are made from.

import { el } from './dom.mjs';

const FAMILIES = Object.freeze([
  [/^Run/, 'run'],
  [/^(Model|Tool)/, 'model'],
  [/^Component/, 'call'],
  [/^(Resolution|Attempt)/, 'resolver'],
  [/^(Hypothesis|Generation|Knowledge)/, 'knowledge'],
  [/^(Trial|Step|Execution)/, 'simulation'],
]);
const PROBLEMS = /(Failed|Rejected|Refused|Unreadable)$/;
const SUMMARY_KEYS = Object.freeze(['source', 'target', 'operation', 'tool', 'style', 'attempt', 'kind',
  'step', 'status', 'error_type', 'record_id', 'purpose', 'action']);
const SUMMARY_PARTS = 4;
const NEAR_THE_BOTTOM = 80;

function familyOf(name) {
  const found = FAMILIES.find(function (family) { return family[0].test(name); });
  return found ? found[1] : 'run';
}

function summaryOf(event) {
  return SUMMARY_KEYS
    .filter(function (key) { return event[key] !== undefined && event[key] !== null; })
    .slice(0, SUMMARY_PARTS)
    .map(function (key) { return key + ' ' + event[key]; })
    .join(' · ');
}

export class EventsView {
  constructor(container) {
    this.list = el('ol', { class: 'event-rows' });
    container.replaceChildren(this.list);
    this.container = container;
    this.origin = null;
  }

  clear() {
    this.list.replaceChildren();
    this.origin = null;
  }

  add(event) {
    if (this.origin === null && typeof event.at === 'number') this.origin = event.at;
    const offset = typeof event.at === 'number' ? '+' + (event.at - this.origin).toFixed(2) + ' s' : '';
    const row = el('li', { class: 'event family-' + familyOf(event.event) + (PROBLEMS.test(event.event) ? ' problem' : '') },
      el('details', {},
        el('summary', {}, el('span', { class: 't' }, offset), el('span', { class: 'name' }, event.event),
          el('span', { class: 'detail' }, summaryOf(event))),
        el('pre', {}, JSON.stringify(event, null, 2))));
    const follows = this.container.scrollHeight - this.container.scrollTop - this.container.clientHeight < NEAR_THE_BOTTOM;
    this.list.appendChild(row);
    if (follows) this.container.scrollTop = this.container.scrollHeight;
  }
}
