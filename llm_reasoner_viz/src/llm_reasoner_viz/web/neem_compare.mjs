// Comparing two NEEMs: which steps of one are those of the other, and what differs in the steps
// that are. Steps are the same step when they are named the same under the same step, the first
// of those with a name matching the first of the other's.

import { textBlock, factsBlock } from './inspector.mjs';
import { describeNode, displayValue, nodeLabel, referenceText } from './neem.mjs';

// How a step stands in a comparison, when it is not the same as in the other.
export const MARK = Object.freeze({ CHANGED: 'changed', ONLY_HERE: 'only here' });

// What is said of a value the other step does not have.
const ABSENT = 'absent';
const ARROW = ' → ';
const KEY_SEPARATOR = '/';

// %% which steps are which

// A key for each step that two steps of different plans share when they are the same step:
// the names from the root down, and how many steps of that name came before under its parent.
function keysOf(nodes) {
  const keys = new Map();
  const counts = new Map();
  nodes.forEach(function (node) {
    const above = node.parent === null ? '' : keys.get(node.parent);
    const named = above + KEY_SEPARATOR + nodeLabel(node);
    const before = counts.get(named) || 0;
    counts.set(named, before + 1);
    keys.set(node.id, named + '#' + before);
  });
  return keys;
}

// %% what differs

function textsOfValues(node) {
  return new Map(node.parameters.map(function (parameter) { return [parameter.name, displayValue(parameter.value)]; }));
}

// A thing a step refers to as one text: what it is and, for an object that is only values, its values.
function textsOfReferences(node) {
  return new Map(node.references.map(function (reference) {
    const values = reference.parameters.map(function (parameter) { return parameter.name + '=' + displayValue(parameter.value); });
    return [reference.name, referenceText(reference) + (values.length > 0 ? ' {' + values.join(', ') + '}' : '')];
  }));
}

// The names in the one and then those only in the other.
function namesOf(here, other) {
  return [...here.keys(), ...[...other.keys()].filter(function (name) { return !here.has(name); })];
}

function differences(here, other) {
  return namesOf(here, other)
    .map(function (name) { return { name: name, here: here.get(name) || ABSENT, other: other.get(name) || ABSENT }; })
    .filter(function (difference) { return difference.here !== difference.other; });
}

// How a step differs from the same step of the other plan: its values and the things it refers to
// that differ, as they are here and as they are there.
function difference(here, other) {
  return {
    parameters: differences(textsOfValues(here), textsOfValues(other)),
    references: differences(textsOfReferences(here), textsOfReferences(other)),
  };
}

// The comparison of this plan with another, called ``title``: the steps that differ with how
// they differ, the steps only this plan has (their ids), and those only the other has.
export function compareNeems(here, other, title) {
  const hereKeys = keysOf(here);
  const otherKeys = keysOf(other);
  const otherByKey = new Map(other.map(function (node) { return [otherKeys.get(node.id), node]; }));
  const changed = new Map();
  const onlyHere = new Set();
  here.forEach(function (node) {
    const same = otherByKey.get(hereKeys.get(node.id));
    if (same === undefined) {
      onlyHere.add(node.id);
      return;
    }
    const found = difference(node, same);
    if (found.parameters.length > 0 || found.references.length > 0) changed.set(node.id, found);
  });
  const hereHas = new Set(hereKeys.values());
  const onlyThere = other
    .filter(function (node) { return !hereHas.has(otherKeys.get(node.id)); })
    .map(function (node) { return { id: node.id, label: nodeLabel(node) }; });
  return { title: title, changed: changed, onlyHere: onlyHere, onlyThere: onlyThere };
}

// %% how it is shown

// How a step of this plan stands in the comparison; null when it is the same as in the other.
export function markOf(comparison, id) {
  if (comparison.onlyHere.has(id)) return MARK.ONLY_HERE;
  return comparison.changed.has(id) ? MARK.CHANGED : null;
}

function plural(count, noun) {
  return count + ' ' + noun + (count === 1 ? '' : 's');
}

// The comparison in a few words: how many steps differ and how many only one plan has.
export function comparisonSummary(comparison) {
  const parts = [
    comparison.changed.size > 0 ? plural(comparison.changed.size, 'step') + (comparison.changed.size === 1 ? ' differs' : ' differ') : null,
    comparison.onlyHere.size > 0 ? plural(comparison.onlyHere.size, 'step') + ' only here' : null,
    comparison.onlyThere.length > 0 ? plural(comparison.onlyThere.length, 'step') + ' only in ' + comparison.title : null,
  ].filter(function (part) { return part !== null; });
  return parts.length === 0 ? 'the same as ' + comparison.title : parts.join(' · ');
}

function differenceFacts(found) {
  return [...found.parameters, ...found.references].map(function (change) { return [change.name, change.here + ARROW + change.other]; });
}

// The view of a step with, after what it was asked to do, how it compares with the other plan.
export function describeComparedNode(node, comparison) {
  const view = describeNode(node);
  const title = 'Compared with ' + comparison.title;
  const mark = markOf(comparison, node.id);
  let compared = textBlock(title, 'The same as in ' + comparison.title + '.');
  if (mark === MARK.ONLY_HERE) compared = textBlock(title, 'Not in ' + comparison.title + '.');
  if (mark === MARK.CHANGED) compared = factsBlock(title, differenceFacts(comparison.changed.get(node.id)));
  return Object.assign({}, view, { blocks: [...view.blocks, compared] });
}
