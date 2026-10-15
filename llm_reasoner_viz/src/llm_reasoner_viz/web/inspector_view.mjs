// Draws what the inspector shows of a span: its title, how it ended, a few facts, and the
// blocks of detail, such as the prompts of a model call as the sections they are made of.

import { el } from './dom.mjs';
import { formatCount } from './format.mjs';
import { BLOCK } from './inspector.mjs';
import { drawPlot } from './pour_view.mjs';
import { isJson, prettyJson } from './prompts.mjs';

const SECTION_OPEN_BELOW = 1200;
const JSON_OPEN_BELOW_LINES = 28;

function copyButton(text) {
  const button = el('button', { type: 'button', class: 'copy', title: 'Copy to the clipboard' }, 'copy');
  button.addEventListener('click', function (click) {
    click.preventDefault();
    navigator.clipboard.writeText(text).then(function () {
      button.textContent = 'copied';
      setTimeout(function () { button.textContent = 'copy'; }, 1200);
    });
  });
  return button;
}

function writtenOut(text) {
  return isJson(text) ? prettyJson(text) : text;
}

function textBlock(block) {
  const text = writtenOut(block.text);
  return el('div', { class: 'body' }, copyButton(text), el('pre', {}, text));
}

function jsonBlock(block) {
  const text = JSON.stringify(block.value, null, 2);
  const lines = text.split('\n').length;
  if (lines <= JSON_OPEN_BELOW_LINES) return el('div', { class: 'body' }, copyButton(text), el('pre', {}, text));
  return el('details', { class: 'body' }, el('summary', {}, lines + ' lines', copyButton(text)), el('pre', {}, text));
}

function sectionsBlock(block) {
  const sections = block.sections.map(function (section) {
    const label = '<' + section.name + (section.attributes ? ' ' + section.attributes : '') + '>';
    return el('details', { class: 'section', open: section.length < SECTION_OPEN_BELOW },
      el('summary', {}, el('span', { class: 'tag' }, label), el('span', { class: 'size' }, section.length + ' characters'),
        copyButton(section.content)),
      el('pre', {}, section.content));
  });
  const loose = block.loose ? el('pre', { class: 'loose' }, block.loose) : null;
  return el('div', { class: 'body' }, loose, sections);
}

function listBlock(block, choose) {
  const items = block.items.map(function (item) {
    const label = item.id
      ? el('button', { type: 'button', class: 'link status-' + item.status, onclick: function () { choose(item.id); } }, item.label)
      : el('span', { class: 'plain' }, item.label);
    return el('li', {}, label, item.note ? el('span', { class: 'note' }, item.note) : null);
  });
  return el('ul', { class: 'items' }, items);
}

function tableBlock(block) {
  return el('table', {}, el('thead', {}, el('tr', {}, block.columns.map(function (name) { return el('th', {}, name); }))),
    el('tbody', {}, block.rows.map(function (row) {
      return el('tr', {}, row.map(function (cell) { return el('td', {}, String(cell)); }));
    })));
}

function factsList(facts) {
  return el('dl', { class: 'facts' }, facts.map(function ([name, value]) {
    return [el('dt', {}, name), el('dd', {}, typeof value === 'number' ? formatCount(value) : String(value))];
  }));
}

const BLOCKS = new Map([
  [BLOCK.TEXT, textBlock],
  [BLOCK.JSON, jsonBlock],
  [BLOCK.SECTIONS, sectionsBlock],
  [BLOCK.LIST, listBlock],
  [BLOCK.TABLE, tableBlock],
  [BLOCK.PLOT, function (block, choose) { return el('div', { class: 'plan-block' }, drawPlot(block.plot, block.emphasised, choose)); }],
  [BLOCK.FACTS, function (block) { return factsList(block.facts); }],
]);

export class InspectorView {
  // ``onChoose(id)`` is called with the id of a span that is picked, and ``onOpenNeem(number)``
  // with the number of the NEEM a span kept when its link is followed.
  constructor(container, onChoose, onOpenNeem) {
    this.container = container;
    this.onChoose = onChoose;
    this.onOpenNeem = onOpenNeem;
    this.empty('Pick a bar on the timeline to see what happened there.');
  }

  empty(message) {
    this.container.replaceChildren(el('p', { class: 'empty' }, message));
  }

  // Shows a view of a span, with the spans it is inside of as a trail to follow upward.
  show(view, trail) {
    const choose = this.onChoose;
    const openNeem = this.onOpenNeem;
    const header = el('header', {},
      trail.length ? el('nav', { class: 'trail' }, trail.map(function (above) {
        return el('button', { type: 'button', onclick: function () { choose(above.id); } }, above.label);
      })) : null,
      el('h2', {}, view.title, el('span', { class: 'badge status-' + view.status }, view.status)),
      view.neem === null ? null : el('button', { type: 'button', class: 'neem-link', title: 'Open the plan this kept, as a NEEM',
        onclick: function () { openNeem(view.neem); } }, 'open NEEM #' + view.neem + ' ›'));
    const blocks = view.blocks.map(function (block) {
      return el('section', { class: 'block' }, el('h3', {}, block.title), BLOCKS.get(block.kind)(block, choose));
    });
    this.container.replaceChildren(header, view.facts.length ? factsList(view.facts) : null, ...blocks);
  }
}
