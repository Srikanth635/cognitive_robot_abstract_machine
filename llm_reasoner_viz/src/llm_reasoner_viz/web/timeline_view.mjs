// Draws the timeline: a row for each component, with a bar for each piece of work, as
// wide as it lasted. Bars of a run that is still going grow as it goes.

import { el } from './dom.mjs';
import { formatDuration } from './format.mjs';
import { fitScale, packTracks } from './layout.mjs';
import { LANE, spanEnd } from './model.mjs';

const LANE_ROWS = Object.freeze([
  { lane: LANE.RUN, label: 'run' },
  { lane: LANE.RESOLVER, label: 'resolver' },
  { lane: LANE.MODEL, label: 'language model' },
  { lane: LANE.WORLD, label: 'world injector' },
  { lane: LANE.KNOWLEDGE, label: 'knowledge' },
  { lane: LANE.SIMULATION, label: 'simulation' },
]);

export const LABEL_WIDTH = 124;
const ROW_HEIGHT = 24;
const LANE_PADDING = 8;
const MINIMUM_BAR_WIDTH = 3;
const GAP_BETWEEN_BARS = 2;
const LABEL_FITS_FROM = 48;
const RIGHT_MARGIN = 40;
const LEAST_PIXELS_PER_SECOND_WHEN_FITTED = 2;
const ZOOM_FACTOR = 1.5;
const FALLBACK_WIDTH = 600;
const TICK_SPACING = 90;
const TICK_STEPS = Object.freeze([0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 1800, 3600]);

function tickStep(pixelsPerSecond) {
  const wanted = TICK_SPACING / pixelsPerSecond;
  return TICK_STEPS.find(function (step) { return step >= wanted; }) || TICK_STEPS[TICK_STEPS.length - 1];
}

function tickText(seconds) {
  return seconds < 1 && seconds > 0 ? seconds.toFixed(2) + ' s' : formatDuration(seconds).replace(/\.00 s$/, ' s');
}

function barTitle(span, now) {
  return span.label + ' · ' + span.status + ' · ' + formatDuration(spanEnd(span, now) - span.start);
}

export class TimelineView {
  constructor(container, onChoose) {
    this.zoom = null;
    this.follows = true;
    this.pixelsPerSecond = 1;
    this.drawn = null;
    this.canvas = el('div', { class: 'timeline-canvas' });
    this.scroller = el('div', { class: 'timeline-scroll' }, this.canvas);
    this.canvas.addEventListener('click', function (click) {
      const bar = click.target.closest('.bar');
      if (bar) onChoose(bar.dataset.id);
    });
    const button = (text, label, action) => el('button', { type: 'button', title: label, 'aria-label': label, onclick: action }, text);
    this.followButton = button('follow', 'Keep the newest work in view while the run goes on', () => {
      this.follows = !this.follows;
      this.redraw();
    });
    const controls = el('div', { class: 'timeline-controls' },
      button('fit', 'Show the whole run', () => { this.zoom = null; this.redraw(); }),
      button('−', 'Zoom out', () => { this.zoomBy(1 / ZOOM_FACTOR); }),
      button('+', 'Zoom in', () => { this.zoomBy(ZOOM_FACTOR); }),
      this.followButton);
    container.replaceChildren(controls, this.scroller);
  }

  zoomBy(factor) {
    this.zoom = this.pixelsPerSecond * factor;
    this.redraw();
  }

  redraw() {
    if (this.drawn) this.render(this.drawn.model, this.drawn.view);
  }

  axis(duration, width) {
    const step = tickStep(this.pixelsPerSecond);
    const ticks = [];
    for (let seconds = 0; seconds <= duration + step; seconds += step) {
      ticks.push(el('span', { class: 'tick', style: 'left:' + seconds * this.pixelsPerSecond + 'px' }, tickText(seconds)));
    }
    return el('div', { class: 'axis' }, el('div', { class: 'lane-label' }),
      el('div', { class: 'lane-track', style: 'width:' + width + 'px' }, ticks));
  }

  // Draws the run as of ``view.now``, on an axis that reaches to ``view.until``; ``view``
  // also says which span is chosen, which component's row to single out, and whether the
  // run is going on.
  render(model, view) {
    this.drawn = { model: model, view: view };
    const origin = model.firstAt === null ? view.now : model.firstAt;
    const duration = Math.max(view.until - origin, 0.001);
    const available = (this.scroller.clientWidth || FALLBACK_WIDTH) - LABEL_WIDTH - RIGHT_MARGIN;
    this.pixelsPerSecond = this.zoom === null
      ? fitScale(duration, available, LEAST_PIXELS_PER_SECOND_WHEN_FITTED) : this.zoom;
    const width = Math.max(available, duration * this.pixelsPerSecond);
    const settings = { origin: origin, pxPerSecond: this.pixelsPerSecond, minWidth: MINIMUM_BAR_WIDTH,
      gap: GAP_BETWEEN_BARS, now: view.now };
    const chosen = view.chosenId === null ? null : model.byId.get(view.chosenId);
    const related = new Set();
    if (chosen) {
      related.add(chosen.parent);
      model.childrenOf(chosen).forEach(function (child) { related.add(child.id); });
    }
    const lanes = LANE_ROWS.map(function (row) {
      const placed = packTracks(model.spans.filter(function (span) { return span.lane === row.lane; }), settings);
      const tracks = placed.reduce(function (most, one) { return Math.max(most, one.track + 1); }, 1);
      const bars = placed.map(function (one) {
        const span = one.span;
        const classes = ['bar', 'status-' + span.status, 'kind-' + span.kind];
        if (span.id === view.chosenId) classes.push('chosen');
        if (related.has(span.id)) classes.push('related');
        return el('div', { class: classes.join(' '), 'data-id': span.id, title: barTitle(span, view.now),
          style: 'left:' + one.x + 'px;top:' + (LANE_PADDING / 2 + one.track * ROW_HEIGHT) + 'px;width:' + one.width + 'px' },
        one.width >= LABEL_FITS_FROM ? span.label : null);
      });
      const dimmed = view.chosenLane !== null && view.chosenLane !== row.lane;
      return el('div', { class: 'lane lane-' + row.lane + (dimmed ? ' dimmed' : '') },
        el('div', { class: 'lane-label' }, row.label),
        el('div', { class: 'lane-track', style: 'width:' + width + 'px;height:' + (tracks * ROW_HEIGHT + LANE_PADDING) + 'px' }, bars));
    });
    const now = view.live ? el('div', { class: 'now-line', style: 'left:' + (LABEL_WIDTH + (view.now - origin) * this.pixelsPerSecond) + 'px' }) : null;
    this.canvas.style.width = LABEL_WIDTH + width + RIGHT_MARGIN + 'px';
    this.canvas.replaceChildren(this.axis(duration, width), ...lanes, ...(now ? [now] : []));
    this.followButton.classList.toggle('on', this.follows);
    if (this.zoom !== null && this.follows && view.live) this.scroller.scrollLeft = this.scroller.scrollWidth;
  }
}
