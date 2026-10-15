// The NEEMs button of the top bar and the list it opens: the NEEMs of the run shown, or
// all of them, by kind, each leading to where it came from.

import { el } from './dom.mjs';
import { NEEM_FILTER, NEEM_SCOPE, describeNeem, kindCounts, shownNeems } from './neems.mjs';

const KEY_ESCAPE = 'Escape';

const SCOPE_LABEL = Object.freeze({
  [NEEM_SCOPE.RUN]: 'this run',
  [NEEM_SCOPE.ALL]: 'all runs',
});

const FILTER_LABEL = Object.freeze({
  [NEEM_FILTER.ALL]: 'all',
  [NEEM_FILTER.TRIALS]: 'trials',
  [NEEM_FILTER.EXECUTIONS]: 'executions',
});

const WHEN_FORMAT = Object.freeze({ day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' });
const MILLISECONDS_PER_SECOND = 1000;

const Say = Object.freeze({
  LOADING: 'Looking for the NEEMs…',
  NO_DATABASE: 'No NEEM database yet. A NEEM is saved when a designator is resolved or executed in cram-viz.',
  NONE_IN_RUN: 'This run saved no NEEMs.',
  NONE_AT_ALL: 'The database holds no NEEMs.',
  NONE_OF_KIND: 'None of this kind.',
  NO_ORIGIN: 'No log says which trial or execution this came from.',
});

export class NeemsView {
  // ``loadNeems(scope)`` asks for the NEEMs of a scope and resolves with the server's answer;
  // ``onChoose(neem)`` is called with the described NEEM that is picked.
  constructor(button, menu, loadNeems, onChoose) {
    this.button = button;
    this.menu = menu;
    this.loadNeems = loadNeems;
    this.onChoose = onChoose;
    this.run = null;
    this.count = 0;
    this.scope = NEEM_SCOPE.RUN;
    this.filter = NEEM_FILTER.ALL;
    this.answer = null;
    this.asking = 0;
    this.button.addEventListener('click', () => this.toggle());
    document.addEventListener('keydown', (key) => { if (key.key === KEY_ESCAPE && this.isOpen) this.close(); });
    // The path of the click is read from the event, not from where its target is now: choosing
    // in the list draws it again, and the element clicked is gone by the time this is called.
    document.addEventListener('click', (click) => {
      const inside = click.composedPath().some((node) => node === this.menu || node === this.button);
      if (this.isOpen && !inside) this.close();
    });
  }

  get isOpen() {
    return !this.menu.hidden;
  }

  // Says which run is shown and how many NEEMs it saved; the button is there only when the
  // run saved some. An open list is asked for again when that changes.
  update(run, count) {
    const changed = run !== this.run || count !== this.count;
    this.run = run;
    this.count = count;
    const available = run !== null && count > 0;
    this.button.hidden = !available;
    this.button.replaceChildren('neems ', el('span', { class: 'count' }, count));
    if (!available) this.close();
    else if (changed && this.isOpen) this.ask();
  }

  toggle() {
    if (this.isOpen) this.close();
    else this.open();
  }

  open() {
    this.menu.hidden = false;
    this.button.setAttribute('aria-expanded', 'true');
    this.button.classList.add('on');
    this.ask();
  }

  close() {
    this.menu.hidden = true;
    this.button.setAttribute('aria-expanded', 'false');
    this.button.classList.remove('on');
  }

  // Asks for the NEEMs of the scope; an answer that is overtaken by a later question is dropped.
  async ask() {
    const question = ++this.asking;
    this.answer = null;
    this.draw();
    const answer = await this.loadNeems(this.scope);
    if (question !== this.asking) return;
    this.answer = answer;
    this.draw();
  }

  choose(neem) {
    this.close();
    this.onChoose(neem);
  }

  // %% drawing

  draw() {
    this.menu.replaceChildren(this.header(), ...this.body());
  }

  header() {
    const scopes = Object.values(NEEM_SCOPE).map((scope) => el('button', {
      type: 'button', class: scope === this.scope ? 'on' : null, 'aria-pressed': String(scope === this.scope),
      onclick: () => { this.scope = scope; this.ask(); },
    }, SCOPE_LABEL[scope]));
    return el('div', { class: 'neems-head' }, el('h2', {}, 'NEEMs'), el('div', { class: 'segmented' }, ...scopes));
  }

  body() {
    if (this.answer === null) return [el('p', { class: 'neems-note' }, Say.LOADING)];
    if (!this.answer.ok) return [el('p', { class: 'neems-note problem' }, this.answer.error)];
    const entries = this.answer.neems;
    const shown = shownNeems(entries, this.filter);
    return [
      this.chips(kindCounts(entries)),
      shown.length === 0 ? el('p', { class: 'neems-note' }, this.emptiness(entries)) : this.list(shown),
      el('p', { class: 'neems-foot', title: 'The database the NEEMs are read from' }, this.answer.database),
    ];
  }

  emptiness(entries) {
    if (!this.answer.exists) return Say.NO_DATABASE;
    if (entries.length > 0) return Say.NONE_OF_KIND;
    return this.scope === NEEM_SCOPE.RUN ? Say.NONE_IN_RUN : Say.NONE_AT_ALL;
  }

  chips(counts) {
    return el('div', { class: 'chips' }, ...Object.values(NEEM_FILTER).map((filter) => el('button', {
      type: 'button', class: filter === this.filter ? 'chip on' : 'chip', 'aria-pressed': String(filter === this.filter),
      onclick: () => { this.filter = filter; this.draw(); },
    }, FILTER_LABEL[filter] + ' ', el('span', { class: 'count' }, counts[filter]))));
  }

  list(entries) {
    return el('ul', { class: 'neem-list' }, ...entries.map((entry) => this.row(describeNeem(entry))));
  }

  row(neem) {
    const when = neem.recordedAt === null ? null : new Date(neem.recordedAt * MILLISECONDS_PER_SECOND).toLocaleString(undefined, WHEN_FORMAT);
    return el('li', {}, el('button', {
      type: 'button', class: 'neem-row' + (neem.standing === null ? '' : ' status-' + neem.standing),
      title: neem.run === null ? Say.NO_ORIGIN : null, onclick: () => this.choose(neem),
    },
    el('span', { class: 'neem-line' },
      el('strong', {}, neem.title),
      el('span', {}, neem.origin === null ? 'unknown origin' : neem.origin),
      neem.verdict === null ? null : el('span', { class: 'neem-verdict' }, neem.verdict)),
    el('span', { class: 'neem-facts' }, [...neem.facts, when].filter(Boolean).join(' · '))));
  }
}
