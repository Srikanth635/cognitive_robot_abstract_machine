// Draws a plan of where pours ended: the opening as a circle, each pour as a numbered point,
// green where it landed in the opening and amber where it missed.

import { svg } from './dom.mjs';

// The lengths a scale bar may have, in metres, and how it is named.
const SCALE_LENGTHS = Object.freeze([0.01, 0.02, 0.05, 0.1, 0.2, 0.5]);
const CENTIMETRES_PER_METRE = 100;
const POINT_RADIUS = 6;
const LABEL_OFFSET = 9;
const EDGE_GAP = 6;
const TRIAL_PREFIX = /^trial /;

function scaleBar(plot) {
  const reach = plot.size / 2 / plot.scale;
  const length = SCALE_LENGTHS.filter(function (candidate) { return candidate <= reach / 2; }).pop() || SCALE_LENGTHS[0];
  const y = plot.size - EDGE_GAP * 2;
  const width = length * plot.scale;
  return svg('g', { class: 'scale' },
    svg('line', { x1: EDGE_GAP, x2: EDGE_GAP + width, y1: y, y2: y }),
    svg('text', { x: EDGE_GAP, y: y - 4 }, Math.round(length * CENTIMETRES_PER_METRE) + ' cm'));
}

// ``emphasised`` is the id of the pour to stand out; ``onChoose`` is called with the id of a
// pour that is pressed.
export function drawPlot(plot, emphasised, onChoose) {
  const middle = plot.size / 2;
  const points = plot.points.map(function (point) {
    const classes = 'pour ' + (point.lands ? 'lands' : 'misses') + (point.id === emphasised ? ' emphasised' : '');
    return svg('g', { class: classes, onclick: function () { onChoose(point.id); } },
      svg('title', {}, point.label + (point.lands ? ': landed' : ': missed')),
      svg('circle', { cx: point.x, cy: point.y, r: POINT_RADIUS }),
      svg('text', { x: point.x, y: point.y - LABEL_OFFSET, 'text-anchor': 'middle' }, point.label.replace(TRIAL_PREFIX, '')));
  });
  return svg('svg', { class: 'plan', viewBox: '0 0 ' + plot.size + ' ' + plot.size, role: 'img',
    'aria-label': 'Where the mouth of the container ended, seen from above and from the robot' },
  svg('line', { class: 'axis', x1: middle, x2: middle, y1: 0, y2: plot.size }),
  svg('line', { class: 'axis', x1: 0, x2: plot.size, y1: middle, y2: middle }),
  svg('circle', { class: 'opening', cx: plot.opening.x, cy: plot.opening.y, r: plot.opening.radius }),
  svg('text', { class: 'direction', x: middle + 5, y: 12 }, 'ahead'),
  svg('text', { class: 'direction', x: plot.size - EDGE_GAP, y: middle - 5, 'text-anchor': 'end' }, 'right'),
  scaleBar(plot), points);
}
