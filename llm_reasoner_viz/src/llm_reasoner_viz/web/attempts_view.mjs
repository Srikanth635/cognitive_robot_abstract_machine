// Draws the attempts of a resolution as a strip of cards, with where the pours ended beside
// them: what each attempt chose, how it ended, and the pour that was tried.

import { el } from './dom.mjs';
import { drawPlot } from './pour_view.mjs';

function choiceLine(choice) {
  return el('li', {}, el('span', { class: 'slot' }, choice.slot), el('span', { class: 'value' }, String(choice.value)));
}

export class AttemptsView {
  // ``onChoose`` is called with the id of the span a card or a pour stands for.
  constructor(container, onChoose) {
    this.container = container;
    this.onChoose = onChoose;
  }

  // Shows ``cards`` under ``title``, the one with id ``chosen`` marked, and a plan of the
  // pours beside them unless ``plot`` is null.
  show(title, cards, plot, chosen) {
    this.container.hidden = cards.length === 0;
    const onChoose = this.onChoose;
    const strip = cards.map(function (card) {
      return el('button', { type: 'button', class: 'card status-' + card.status + (card.id === chosen ? ' chosen' : ''),
        onclick: function () { onChoose(card.id); } },
      el('span', { class: 'card-head' }, el('strong', {}, card.label), el('span', { class: 'badge status-' + card.status }, card.status)),
      el('ul', { class: 'choices' }, card.chose.map(choiceLine)),
      card.why ? el('p', { class: 'why' }, card.why) : null,
      card.outcome ? el('p', { class: 'outcome' }, 'pour ' + card.outcome) : null,
      card.rejectedFor ? el('p', { class: 'rejection' }, 'rejected: ' + card.rejectedFor) : null);
    });
    this.container.replaceChildren(
      el('h2', {}, title),
      el('div', { class: 'strip' }, strip, plot ? el('div', { class: 'pours' }, drawPlot(plot, null, onChoose)) : null));
  }
}
