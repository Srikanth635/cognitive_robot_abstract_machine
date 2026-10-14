// The controls that pick out spans: a search of what spans say, a switch to show only
// what went wrong, how many spans match, and a step to the next match.

import { el } from './dom.mjs';

export class FilterView {
  // ``filter`` is changed by the controls; ``onChange`` is called after, and ``onNext``
  // when the next match is asked for.
  constructor(container, filter, onChange, onNext) {
    this.search = el('input', { type: 'search', placeholder: 'find in prompts, answers, names…',
      'aria-label': 'Find spans that say', oninput: () => { filter.text = this.search.value; onChange(); },
      onkeydown: (key) => { if (key.key === 'Enter') onNext(); } });
    this.problems = el('input', { type: 'checkbox', onchange: () => { filter.problemsOnly = this.problems.checked; onChange(); } });
    this.count = el('span', { class: 'count' });
    this.next = el('button', { type: 'button', title: 'Go to the next match', onclick: onNext }, 'next ›');
    container.replaceChildren(this.search, el('label', { class: 'switch' }, this.problems, 'problems only'), this.count, this.next);
  }

  // Says how many of the spans match; ``matching`` is null when nothing is asked for.
  show(matching, total) {
    this.count.textContent = matching === null ? '' : matching.length + ' of ' + total;
    this.next.disabled = matching === null || matching.length === 0;
  }
}
