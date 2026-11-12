// What the page shows of one NEEM: the tree of the nodes of the statechart its plan was
// performed as, and for each step how it ended, what it was asked to do and the things and
// places it refers to, in the blocks the inspector draws.

import { factsBlock, jsonBlock } from './inspector.mjs';
import { formatOffset } from './format.mjs';
import { STATUS, shortName } from './model.mjs';

const NEEMS_ADDRESS = '/api/neems/';
const SIGNIFICANT_DIGITS = 6;
const NODE_SUFFIX = /Node$/;
const WORD_BOUNDARY = /([a-z])([A-Z])/g;

// How a value that was not set is said.
export const NOT_SET = 'not set';

// How a step stands on the page, by the status it ended in. A step not begun, paused, or
// interrupted because the step above it ended stands nowhere.
const STEP_STANDING = Object.freeze({
  RUNNING: STATUS.RUNNING,
  SUCCEEDED: STATUS.OK,
  FAILED: STATUS.FAILED,
});

// What is said of a step that never began, and of one that has not ended.
export const NEVER_STARTED = 'never';
export const NOT_ENDED = 'not yet';

// What stands for a class or for a member of an enumeration in a value: the path of its module,
// the name of the class, and for a member its name, such as ``coraplex.datastructures.enums.Arms.RIGHT``.
const CLASS_PATH = /^[a-z_]\w*(\.[a-z_]\w*)+\.[A-Z]\w*(\.\w+)?$/;

// %% where to ask

export function neemAddress(identifier) {
  return NEEMS_ADDRESS + identifier;
}

// %% the tree

function plainName(type) {
  return type.replace(NODE_SUFFIX, '').replace(WORD_BOUNDARY, '$1 $2').toLowerCase();
}

// What a step is called in the tree: the action it was asked to perform, else its class.
export function nodeLabel(node) {
  if (node.designator !== null) return node.designator;
  return plainName(node.type);
}

// How a step stands on the page; null for one not begun or paused.
export function standingOf(node) {
  return STEP_STANDING[node.status] || null;
}

// Which steps are of the plan rather than of what its executor added to watch it: the first
// root and every step below it.
function inPlan(nodes) {
  const found = new Map();
  nodes.forEach(function (node) {
    found.set(node.id, node.parent === null ? node.id === nodes[0].id : found.get(node.parent));
  });
  return found;
}

// The rows of the tree, one for each step, in the order of the plan; what the executor added to
// watch the plan is set apart from it.
export function planRows(nodes) {
  const plan = inPlan(nodes);
  return nodes.map(function (node) {
    return {
      id: node.id, depth: node.depth, label: nodeLabel(node), type: node.type,
      standing: standingOf(node), muted: !plan.get(node.id),
    };
  });
}

// The step the view opens on: the first action, which says most of what was done; the root
// of a plan that has none.
export function firstStep(nodes) {
  return nodes.find(function (node) { return node.designator !== null; }) || nodes[0];
}

// The steps above a step, from the root, as links to follow upward.
export function trailOf(nodes, id) {
  const byId = new Map(nodes.map(function (node) { return [node.id, node]; }));
  const trail = [];
  for (let above = byId.get(byId.get(id).parent); above !== undefined; above = byId.get(above.parent)) {
    trail.unshift({ id: above.id, label: nodeLabel(above) });
  }
  return trail;
}

// %% values written for reading

function plainNumber(number) {
  return Number.isInteger(number) ? String(number) : String(Number(number.toPrecision(SIGNIFICANT_DIGITS)));
}

// A value as it is read: a number with what matters of it, a class or a member of an
// enumeration by its name, and a value that was not set as such.
export function displayValue(value) {
  if (value === null) return NOT_SET;
  if (typeof value === 'boolean') return String(value);
  if (typeof value === 'number') return plainNumber(value);
  return CLASS_PATH.test(value) ? shortName(value) : value;
}

function coordinates(numbers) {
  return '(' + numbers.map(plainNumber).join(', ') + ')';
}

function poseParts(pose) {
  const { frame, position, orientation } = pose;
  return [
    frame === null ? 'no frame' : frame,
    position === null ? null : coordinates([position.x, position.y, position.z]),
    orientation === null ? null : 'facing ' + coordinates([orientation.x, orientation.y, orientation.z, orientation.w]),
  ].filter(function (part) { return part !== null; });
}

// A thing or a place a step refers to: its kind, and the name the world gives it or where it is.
export function referenceText(reference) {
  const parts = [reference.type];
  if (reference.label !== null) parts.push(reference.label);
  if (reference.pose !== null) parts.push(...poseParts(reference.pose));
  return parts.join(' · ');
}

// %% what a step is

function valueFacts(parameters) {
  return parameters.map(function (parameter) { return [parameter.name, displayValue(parameter.value)]; });
}

// The view of a step, as the inspector draws a span: its title, how it stands, a few facts
// and blocks of detail.
export function describeNode(node) {
  const withValues = node.references.filter(function (reference) { return reference.parameters.length > 0; });
  const blocks = [
    node.parameters.length > 0 ? factsBlock('Values', valueFacts(node.parameters)) : null,
    node.references.length > 0
      ? factsBlock('Things and places', node.references.map(function (reference) { return [reference.name, referenceText(reference)]; }))
      : null,
    ...withValues.map(function (reference) { return factsBlock('Values of ' + reference.name, valueFacts(reference.parameters)); }),
    jsonBlock('As it is kept', node.stored),
  ].filter(function (block) { return block !== null; });
  return {
    title: node.designator !== null ? node.designator : nodeLabel(node),
    status: standingOf(node) || '',
    neem: null,
    facts: [
      ['step', '#' + node.id], ['type', node.type], ['status', node.status],
      ['started', node.started === null ? NEVER_STARTED : formatOffset(node.started)],
      ['ended', node.ended === null ? NOT_ENDED : formatOffset(node.ended)],
    ],
    blocks: blocks,
  };
}
