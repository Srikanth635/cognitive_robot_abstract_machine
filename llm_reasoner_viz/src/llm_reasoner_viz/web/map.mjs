// The components of the system and the calls between them: how much each has done, what it
// is doing, and which connection has just been used.

import { KIND, LANE, edgeKey } from './model.mjs';

export const MAP_WIDTH = 420;
export const MAP_HEIGHT = 250;
export const NODE_WIDTH = 108;
export const NODE_HEIGHT = 46;

// Which side of a node what it is doing is written on.
export const ABOVE = -1;
export const BELOW = 1;

// Each component with what it is called, where it stands on the map and what its count
// counts.
export const NODES = Object.freeze([
  { lane: LANE.RESOLVER, label: 'resolver', unit: 'task', x: 110, y: 52, activitySide: ABOVE },
  { lane: LANE.SIMULATION, label: 'simulation', unit: 'task', x: 310, y: 52, activitySide: ABOVE },
  { lane: LANE.WORLD, label: 'world injector', unit: 'call', x: 62, y: 186, activitySide: BELOW },
  { lane: LANE.MODEL, label: 'language model', unit: 'call', x: 210, y: 186, activitySide: BELOW },
  { lane: LANE.KNOWLEDGE, label: 'knowledge', unit: 'task', x: 358, y: 186, activitySide: BELOW },
]);

// The connections a call can go over, from the component that calls to the one called.
export const EDGES = Object.freeze([
  { from: LANE.RESOLVER, to: LANE.MODEL },
  { from: LANE.RESOLVER, to: LANE.WORLD },
  { from: LANE.RESOLVER, to: LANE.KNOWLEDGE },
  { from: LANE.RESOLVER, to: LANE.SIMULATION },
  { from: LANE.KNOWLEDGE, to: LANE.MODEL },
  { from: LANE.MODEL, to: LANE.WORLD },
  { from: LANE.MODEL, to: LANE.KNOWLEDGE },
]);

// How long a connection shows that it has just been used, in seconds.
export const PULSE_SECONDS = 1.5;

const ACTIVITIES = new Map([
  [KIND.MODEL, (span) => span.data.asked.source + ' asks'],
  [KIND.TOOL, (span) => span.label],
  [KIND.CALL, (span) => span.data.called.source + ' calls ' + span.label],
  [KIND.RESOLUTION, (span) => 'resolving ' + span.label],
  [KIND.ATTEMPT, (span) => span.label],
  [KIND.HYPOTHESIS, (span) => 'generating ' + span.label],
  [KIND.GENERATION_ATTEMPT, (span) => 'generation ' + span.label],
  [KIND.TRIAL, (span) => span.label + ' running'],
  [KIND.STEP, (span) => span.label],
]);

function activityOf(span) {
  const describe = ACTIVITIES.get(span.kind);
  return describe ? describe(span) : span.label;
}

function nodeState(node, model) {
  const spans = model.spans.filter((span) => span.lane === node.lane);
  const open = spans.filter((span) => span.end === null);
  return Object.assign({}, node, {
    calls: spans.length,
    open: open.length,
    activity: open.length === 0 ? null : activityOf(open[open.length - 1]),
  });
}

function edgeState(edge, model, now) {
  const key = edgeKey(edge.from, edge.to);
  const lastUsed = model.edgeLastAt.get(key);
  const sinceUse = now - lastUsed;
  return Object.assign({}, edge, {
    count: model.edges.get(key) || 0,
    pulsing: lastUsed !== undefined && sinceUse >= 0 && sinceUse < PULSE_SECONDS,
  });
}

// The map as of a moment: a node for each component and an edge for each connection.
export function mapState(model, now) {
  return {
    nodes: NODES.map((node) => nodeState(node, model)),
    edges: EDGES.map((edge) => edgeState(edge, model, now)),
  };
}
