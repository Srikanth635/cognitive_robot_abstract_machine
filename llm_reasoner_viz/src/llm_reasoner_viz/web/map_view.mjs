// Draws the map of the components: a box for each, an arrow for each connection, and what
// is going on in them.

import { svg } from './dom.mjs';
import { edgeKey } from './model.mjs';
import { ABOVE, EDGES, MAP_HEIGHT, MAP_WIDTH, NODES, NODE_HEIGHT, NODE_WIDTH } from './map.mjs';

const CURVE_OFFSET = 20;
const ACTIVITY_LENGTH = 30;
const ACTIVITY_OFFSET_ABOVE = 8;
const ACTIVITY_OFFSET_BELOW = 17;

const centres = new Map(NODES.map(function (node) { return [node.lane, node]; }));

// Where a line from a box's centre toward a point leaves the box.
function leavingPoint(centre, toward) {
  const dx = toward.x - centre.x;
  const dy = toward.y - centre.y;
  const scale = 1 / Math.max(Math.abs(dx) / (NODE_WIDTH / 2), Math.abs(dy) / (NODE_HEIGHT / 2));
  return { x: centre.x + dx * scale, y: centre.y + dy * scale };
}

// The curve of a connection, bent to the right of its direction so that two connections
// between the same boxes in opposite directions do not lie on one another.
function curve(from, to) {
  const length = Math.hypot(to.x - from.x, to.y - from.y);
  const control = {
    x: (from.x + to.x) / 2 + ((to.y - from.y) / length) * CURVE_OFFSET,
    y: (from.y + to.y) / 2 - ((to.x - from.x) / length) * CURVE_OFFSET,
  };
  const start = leavingPoint(from, control);
  const end = leavingPoint(to, control);
  const middle = {
    x: 0.25 * start.x + 0.5 * control.x + 0.25 * end.x,
    y: 0.25 * start.y + 0.5 * control.y + 0.25 * end.y,
  };
  return {
    path: 'M' + start.x + ',' + start.y + ' Q' + control.x + ',' + control.y + ' ' + end.x + ',' + end.y,
    middle: middle,
  };
}

function shortened(text) {
  return text.length > ACTIVITY_LENGTH ? text.slice(0, ACTIVITY_LENGTH - 1) + '…' : text;
}

export class MapView {
  // The map is drawn into the container once; ``update`` then changes what it shows.
  constructor(container, onChooseLane) {
    this.edges = new Map();
    this.nodes = new Map();
    const arrow = svg('marker', { id: 'arrow', viewBox: '0 0 8 8', refX: 7, refY: 4,
      markerWidth: 7, markerHeight: 7, orient: 'auto' }, svg('path', { d: 'M0,0 L8,4 L0,8 z' }));
    const edgeLayer = svg('g', { class: 'edges' }, EDGES.map(this.buildEdge, this));
    const nodeLayer = svg('g', { class: 'nodes' }, NODES.map(function (node) {
      return this.buildNode(node, onChooseLane);
    }, this));
    container.replaceChildren(svg('svg', { class: 'map', viewBox: '0 0 ' + MAP_WIDTH + ' ' + MAP_HEIGHT,
      role: 'img', 'aria-label': 'The components and the calls between them' },
    svg('defs', {}, arrow), edgeLayer, nodeLayer));
  }

  buildEdge(edge) {
    const shape = curve(centres.get(edge.from), centres.get(edge.to));
    const path = svg('path', { d: shape.path, class: 'edge-line', 'marker-end': 'url(#arrow)' });
    const count = svg('text', { class: 'edge-count', x: shape.middle.x, y: shape.middle.y + 3 });
    const group = svg('g', { class: 'edge' }, path, count);
    this.edges.set(edgeKey(edge.from, edge.to), { group: group, count: count });
    return group;
  }

  buildNode(node, onChooseLane) {
    const box = svg('rect', { x: -NODE_WIDTH / 2, y: -NODE_HEIGHT / 2, width: NODE_WIDTH,
      height: NODE_HEIGHT, rx: 10 });
    const name = svg('text', { class: 'node-name', y: -3 }, node.label);
    const count = svg('text', { class: 'node-count', y: 13 });
    const activityY = node.activitySide === ABOVE ? -NODE_HEIGHT / 2 - ACTIVITY_OFFSET_ABOVE
      : NODE_HEIGHT / 2 + ACTIVITY_OFFSET_BELOW;
    const activity = svg('text', { class: 'node-activity', y: activityY });
    const group = svg('g', { class: 'node lane-' + node.lane, transform: 'translate(' + node.x + ',' + node.y + ')',
      onclick: function () { onChooseLane(node.lane); } }, box, name, count, activity);
    this.nodes.set(node.lane, { group: group, count: count, activity: activity, unit: node.unit });
    return group;
  }

  update(state, chosenLane) {
    state.nodes.forEach(function (node) {
      const drawn = this.nodes.get(node.lane);
      drawn.count.textContent = node.calls + ' ' + drawn.unit + (node.calls === 1 ? '' : 's');
      drawn.activity.textContent = node.activity === null ? '' : shortened(node.activity);
      drawn.group.classList.toggle('active', node.open > 0);
      drawn.group.classList.toggle('chosen', chosenLane === node.lane);
      drawn.group.classList.toggle('dimmed', chosenLane !== null && chosenLane !== node.lane);
    }, this);
    state.edges.forEach(function (edge) {
      const drawn = this.edges.get(edgeKey(edge.from, edge.to));
      drawn.count.textContent = edge.count > 0 ? edge.count : '';
      drawn.group.classList.toggle('used', edge.count > 0);
      drawn.group.classList.toggle('pulsing', edge.pulsing);
    }, this);
  }
}
