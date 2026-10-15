// The screen of one NEEM, in place of the run's panels: where the plan came from and how it
// ended, the tree of its steps, and what the chosen step was asked to do.

import { el } from './dom.mjs';
import { InspectorView } from './inspector_view.mjs';
import { describeNode, firstStep, planRows, trailOf } from './neem.mjs';
import { describeNeem } from './neems.mjs';

const RUN_NAME_LENGTH = 8;
const INDENT_PIXELS = 16;
const WHEN_FORMAT = Object.freeze({ day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' });
const MILLISECONDS_PER_SECOND = 1000;

export class NeemScreen {
  // ``onBack()`` returns to the run; ``onOpenOrigin(neem)`` goes to the trial or the execution
  // the NEEM is the plan of, with the described NEEM.
  constructor(container, onBack, onOpenOrigin) {
    this.container = container;
    this.onBack = onBack;
    this.onOpenOrigin = onOpenOrigin;
    this.nodes = [];
    this.chosenId = null;
    this.header = el('header', { class: 'neem-header' });
    this.summary = el('p', { class: 'neem-summary' });
    this.tree = el('nav', { class: 'plan-tree', 'aria-label': 'Steps of the plan' });
    this.detail = new InspectorView(el('div', { id: 'neem-detail' }), (id) => this.choose(id), () => {});
    this.detail.empty('Pick a step of the plan.');
    container.replaceChildren(this.header, this.summary, el('div', { class: 'neem-body' },
      el('section', { class: 'panel' }, el('h2', {}, 'Plan'), this.tree),
      el('section', { class: 'panel' }, el('h2', {}, 'Step'), this.detail.container)));
  }

  get isOpen() {
    return !this.container.hidden;
  }

  show(payload) {
    this.nodes = payload.plan;
    this.drawHeader(describeNeem(payload.neem), payload.neem);
    this.container.hidden = false;
    this.choose(firstStep(this.nodes).id);
  }

  // Says why there is no NEEM to show, or that one is being read.
  showMessage(text) {
    this.nodes = [];
    this.header.replaceChildren(this.backButton(), el('p', { class: 'neem-note' }, text));
    this.summary.replaceChildren();
    this.tree.replaceChildren();
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
    this.detail.show(describeNode(node), trailOf(this.nodes, id));
  }

  // %% drawing

  backButton() {
    return el('button', { type: 'button', class: 'back', onclick: this.onBack }, '‹ back to the run');
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
        onclick: () => this.onOpenOrigin(neem) }, 'open the ' + neem.origin + ' in its run ›'));
    this.summary.textContent = [...neem.facts, 'plan ' + entry.status.toLowerCase()].join(' · ');
  }

  drawTree() {
    this.tree.replaceChildren(...planRows(this.nodes).map((row) => el('button', {
      type: 'button', class: 'plan-row' + (row.muted ? ' muted' : '') + (row.id === this.chosenId ? ' on' : ''),
      style: 'padding-left: ' + (10 + row.depth * INDENT_PIXELS) + 'px', 'aria-current': row.id === this.chosenId ? 'true' : null,
      onclick: () => this.choose(row.id),
    }, el('span', { class: 'plan-label' }, row.label), el('span', { class: 'plan-id' }, '#' + row.id))));
  }
}
