// The screen of one NEEM, in place of the run's panels: where the plan came from and how it
// ended, the tree of its steps, what the chosen step was asked to do, and how it compares with
// another NEEM when one is chosen.

import { el } from './dom.mjs';
import { InspectorView } from './inspector_view.mjs';
import { describeNode, firstStep, planRows, trailOf } from './neem.mjs';
import { MARK, compareNeems, comparisonSummary, describeComparedNode, markOf } from './neem_compare.mjs';
import { describeNeem } from './neems.mjs';

const RUN_NAME_LENGTH = 8;
const INDENT_PIXELS = 16;
const WHEN_FORMAT = Object.freeze({ day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' });
const MILLISECONDS_PER_SECOND = 1000;
const NO_COMPARISON = '';

// How a mark in the tree is worded.
const MARK_TEXT = Object.freeze({ [MARK.CHANGED]: 'differs', [MARK.ONLY_HERE]: 'only here' });

export class NeemScreen {
  // ``actions`` are what the screen asks of the page: ``onBack()`` to return to the run;
  // ``onOpenOrigin(neem)`` to go to the trial or the execution the NEEM is the plan of;
  // ``onCompare(number)`` to tell which NEEM is compared with, null for none; ``loadNeems()`` for
  // all the NEEMs listed, and ``loadNeem(number)`` for one with the steps of its plan, both
  // answering with the server's payload.
  constructor(container, actions) {
    this.container = container;
    this.actions = actions;
    this.nodes = [];
    this.identifier = null;
    this.chosenId = null;
    this.comparison = null;
    this.header = el('header', { class: 'neem-header' });
    this.summary = el('p', { class: 'neem-summary' });
    this.compared = el('p', { class: 'neem-summary' });
    this.chooser = el('select', { class: 'compare', 'aria-label': 'Compare with another NEEM',
      onchange: () => this.compareWith(this.chooser.value === NO_COMPARISON ? null : Number(this.chooser.value)) });
    this.tree = el('nav', { class: 'plan-tree', 'aria-label': 'Steps of the plan' });
    this.onlyThere = el('div', { class: 'plan-extra' });
    this.detail = new InspectorView(el('div', { id: 'neem-detail' }), (id) => this.choose(id), () => {});
    this.detail.empty('Pick a step of the plan.');
    container.replaceChildren(this.header, this.summary, this.compared, el('div', { class: 'neem-body' },
      el('section', { class: 'panel' }, el('h2', {}, 'Plan'), this.tree, this.onlyThere),
      el('section', { class: 'panel' }, el('h2', {}, 'Step'), this.detail.container)));
  }

  get isOpen() {
    return !this.container.hidden;
  }

  // Shows a NEEM as the server sends it, compared with another when one is named.
  show(payload, comparedWith) {
    this.nodes = payload.plan;
    this.identifier = payload.neem.neem;
    this.comparison = null;
    this.drawHeader(describeNeem(payload.neem), payload.neem);
    this.container.hidden = false;
    this.choose(firstStep(this.nodes).id);
    this.fillChooser(comparedWith);
    if (comparedWith !== null) this.compareWith(comparedWith);
  }

  // Says why there is no NEEM to show, or that one is being read.
  showMessage(text) {
    this.nodes = [];
    this.comparison = null;
    this.header.replaceChildren(this.backButton(), el('p', { class: 'neem-note' }, text));
    this.summary.replaceChildren();
    this.compared.replaceChildren();
    this.tree.replaceChildren();
    this.onlyThere.replaceChildren();
    this.detail.empty('');
    this.container.hidden = false;
  }

  hide() {
    this.container.hidden = true;
  }

  choose(id) {
    this.chosenId = id;
    this.drawTree();
    const node = this.nodes.find((candidate) => candidate.id === id);
    const view = this.comparison === null ? describeNode(node) : describeComparedNode(node, this.comparison);
    this.detail.show(view, trailOf(this.nodes, id));
  }

  // %% comparing

  // Lists the other NEEMs to compare with, the one given selected.
  async fillChooser(selected) {
    const answer = await this.actions.loadNeems();
    const others = answer.ok ? answer.neems.filter((entry) => entry.neem !== this.identifier) : [];
    this.chooser.replaceChildren(el('option', { value: NO_COMPARISON }, 'compare with…'), ...others.map((entry) => {
      const neem = describeNeem(entry);
      return el('option', { value: neem.identifier, selected: neem.identifier === selected },
        [neem.title, neem.origin === null ? 'unknown origin' : neem.origin, neem.verdict].filter(Boolean).join(' · '));
    }));
  }

  // Compares this NEEM with another, or stops comparing when none is given.
  async compareWith(identifier) {
    this.actions.onCompare(identifier);
    if (identifier === null) {
      this.comparison = null;
      this.chooser.value = NO_COMPARISON;
    } else {
      const other = await this.actions.loadNeem(identifier);
      if (!other.ok) return this.compared.replaceChildren('Could not read NEEM #' + identifier + ': ' + other.error);
      this.comparison = compareNeems(this.nodes, other.plan, '#' + identifier);
    }
    this.drawComparison();
    this.choose(this.chosenId);
  }

  // %% drawing

  backButton() {
    return el('button', { type: 'button', class: 'back', onclick: this.actions.onBack }, '‹ back to the run');
  }

  drawHeader(neem, entry) {
    const when = neem.recordedAt === null ? null
      : new Date(neem.recordedAt * MILLISECONDS_PER_SECOND).toLocaleString(undefined, WHEN_FORMAT);
    const origin = [neem.origin === null ? 'unknown origin' : neem.origin,
      neem.run === null ? null : 'run ' + neem.run.slice(0, RUN_NAME_LENGTH), when].filter(Boolean).join(' · ');
    this.header.replaceChildren(
      this.backButton(),
      el('h2', {}, 'NEEM ' + neem.title),
      neem.verdict === null ? null : el('span', { class: 'badge' + (neem.standing === null ? '' : ' status-' + neem.standing) }, neem.verdict),
      el('span', { class: 'neem-origin' }, origin),
      neem.run === null ? null : el('button', { type: 'button', class: 'neem-link',
        title: 'Go to the bar of the ' + neem.origin + ' in the run that saved this',
        onclick: () => this.actions.onOpenOrigin(neem) }, 'open the ' + neem.origin + ' in its run ›'),
      this.chooser);
    this.summary.textContent = [...neem.facts, 'plan ' + entry.status.toLowerCase()].join(' · ');
    this.compared.replaceChildren();
  }

  drawComparison() {
    this.compared.textContent = this.comparison === null ? '' : 'Compared with ' + this.comparison.title + ': ' + comparisonSummary(this.comparison);
    const only = this.comparison === null ? [] : this.comparison.onlyThere;
    this.onlyThere.replaceChildren(...(only.length === 0 ? [] : [
      el('h3', {}, 'Only in ' + this.comparison.title),
      ...only.map((step) => el('p', {}, step.label, el('span', { class: 'plan-id' }, ' #' + step.id))),
    ]));
  }

  drawTree() {
    this.tree.replaceChildren(...planRows(this.nodes).map((row) => {
      const mark = this.comparison === null ? null : markOf(this.comparison, row.id);
      return el('button', {
        type: 'button', class: ['plan-row', row.muted ? 'muted' : null, row.id === this.chosenId ? 'on' : null, mark === null ? null : 'marked'].filter(Boolean).join(' '),
        style: 'padding-left: ' + (10 + row.depth * INDENT_PIXELS) + 'px', 'aria-current': row.id === this.chosenId ? 'true' : null,
        onclick: () => this.choose(row.id),
      }, el('span', { class: 'plan-label' }, row.label),
      mark === null ? null : el('span', { class: 'plan-mark ' + (mark === MARK.CHANGED ? 'changed' : 'only-here') }, MARK_TEXT[mark]),
      el('span', { class: 'plan-id' }, '#' + row.id));
    }));
  }
}
