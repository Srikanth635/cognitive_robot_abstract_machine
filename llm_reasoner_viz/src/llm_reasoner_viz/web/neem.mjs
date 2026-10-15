// What the page shows of one NEEM: the tree of the steps of its plan, and for each step what
// it was asked to do and the things and places it refers to, in the blocks the inspector draws.

import { factsBlock, jsonBlock } from './inspector.mjs';
import { shortName } from './model.mjs';

const NEEMS_ADDRESS = '/api/neems/';
const SIGNIFICANT_DIGITS = 6;
const NODE_SUFFIX = /Node$/;
const WORD_BOUNDARY = /([a-z])([A-Z])/g;

// How a value that was not set is said.
export const NOT_SET = 'not set';

// The kinds of step that the page names itself, the others being named for what they were
// asked to do or for their kind.
export const NODE_TYPE = Object.freeze({ SEQUENTIAL: 'SequentialNode', ACTION: 'ActionNode', CONDITION: 'ConditionNode' });

const PRE_CONDITION = 'pre_condition';

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

function isCheckedBefore(node) {
  const flag = node.parameters.find(function (parameter) { return parameter.name === PRE_CONDITION; });
  return flag !== undefined && flag.value === true;
}

// What a step is called in the tree: what it was asked to do, else what kind of step it is.
export function nodeLabel(node) {
  if (node.designator !== null) return node.designator;
  if (node.type === NODE_TYPE.SEQUENTIAL) return 'sequence';
  if (node.type === NODE_TYPE.CONDITION) return isCheckedBefore(node) ? 'condition before' : 'condition after';
  return plainName(node.type);
}

// The rows of the tree, one for each step, in the order of the plan.
export function planRows(nodes) {
  return nodes.map(function (node) {
    return { id: node.id, depth: node.depth, label: nodeLabel(node), type: node.type, muted: node.type === NODE_TYPE.CONDITION };
  });
}

// The step the view opens on: the first action, which says most of what was done; the root
// of a plan that has none.
export function firstStep(nodes) {
  return nodes.find(function (node) { return node.type === NODE_TYPE.ACTION; }) || nodes[0];
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

// The view of a step, as the inspector draws a span: its title, a few facts and blocks of
// detail. A step has no status of its own, since the plan stores only its root as ended.
export function describeNode(node) {
  const withValues = node.references.filter(function (reference) { return reference.parameters.length > 0; });
  const blocks = [
    node.parameters.length > 0 ? factsBlock('Values', valueFacts(node.parameters)) : null,
    node.references.length > 0
      ? factsBlock('Things and places', node.references.map(function (reference) { return [reference.name, referenceText(reference)]; }))
      : null,
    ...withValues.map(function (reference) { return factsBlock('Values of ' + reference.name, valueFacts(reference.parameters)); }),
    jsonBlock('Rows it is kept in', node.rows),
  ].filter(function (block) { return block !== null; });
  return {
    title: node.designator !== null ? node.designator : nodeLabel(node),
    status: '',
    neem: null,
    facts: [['step', '#' + node.id], ['type', node.type]],
    blocks: blocks,
  };
}
