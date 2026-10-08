// Unit tests for panels/graph/designator.js: the action designator's Lisp and JSON
// spellings, which descriptions start collapsed, and finding a referenced description.
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const WEB = path.join(__dirname, '..', '..', '..', 'cram_viz', 'src', 'cram_viz', 'web');

function loadDesignatorView() {
  global.window = {};
  new Function(fs.readFileSync(path.join(WEB, 'panels/graph/designator.js'), 'utf8'))();
  return global.window.DesignatorView;
}

function value(kind, text, data, extra) {
  return Object.assign({ node: 'value', kind: kind, text: text, data: data, reference: '', note: '' }, extra || {});
}

function description(article, kind, id, type, properties) {
  return { node: 'description', article: article, kind: kind, id: id, type: type, note: '', properties: properties };
}

// an action with one open slot, one defaulted slot and a hypothesis whose structure
// holds a phase that refers back to the role described before it
function fixtureDesignator() {
  const role = description('a', 'manipulation-role', 'jug', 'pouring:SourceContainer', [
    { key: 'label', values: [value('literal', '"the jug"', 'the jug')] },
  ]);
  const phase = description('an', 'action-phase', 'lift', 'prehension:LiftPhase', [
    { key: 'has-coupling-regime', values: [value('term', 'core:Coupled', 'core:Coupled')] },
    { key: 'involves-role', many: true, values: [value('reference', '#jug', { $ref: 'jug' }, { reference: 'jug' })] },
  ]);
  const structure = description('an', 'action-phase-structure', 'pour', 'pouring:PouringPhaseStructure', [
    { key: 'defines-role', many: true, values: [role] },
    { key: 'defines-phase', many: true, values: [phase] },
  ]);
  return description('an', 'action', 'PouringAction', 'pouring', [
    { key: 'source-container', values: [value('variable', '?source-container', '?source-container')] },
    { key: 'tilt-angle', values: [value('literal', '1.85', 1.85)] },
    { key: 'hypothesis', values: [structure] },
  ]);
}

// %% Lisp
test('the designator is written as nested Lisp descriptions', function () {
  const { lispText } = loadDesignatorView();
  assert.strictEqual(lispText(fixtureDesignator()), [
    '(an action (type pouring)',
    '  (source-container ?source-container)',
    '  (tilt-angle 1.85)',
    '  (hypothesis',
    '    (an action-phase-structure (type pouring:PouringPhaseStructure)',
    '      (defines-role',
    '        (a manipulation-role (type pouring:SourceContainer)',
    '          (label "the jug")))',
    '      (defines-phase',
    '        (an action-phase (type prehension:LiftPhase)',
    '          (has-coupling-regime core:Coupled)',
    '          (involves-role #jug))))))',
  ].join('\n'));
});

// %% JSON
test('the designator is written as nested JSON objects', function () {
  const { toJson } = loadDesignatorView();
  assert.deepStrictEqual(toJson(fixtureDesignator()), {
    action: 'pouring',
    'source-container': '?source-container',
    'tilt-angle': 1.85,
    hypothesis: {
      'action-phase-structure': 'pouring:PouringPhaseStructure',
      id: 'pour',
      'defines-role': [{ 'manipulation-role': 'pouring:SourceContainer', id: 'jug', label: 'the jug' }],
      'defines-phase': [{
        'action-phase': 'prehension:LiftPhase',
        id: 'lift',
        'has-coupling-regime': 'core:Coupled',
        'involves-role': [{ $ref: 'jug' }],
      }],
    },
  });
});

// %% collapsing and references
test('descriptions below the hypothesis structure start collapsed', function () {
  const { initiallyCollapsed } = loadDesignatorView();
  assert.deepStrictEqual([...initiallyCollapsed(fixtureDesignator())].sort(), ['jug', 'lift']);
});

test('a reference is found by the descriptions enclosing its target', function () {
  const { pathTo } = loadDesignatorView();
  assert.deepStrictEqual(pathTo(fixtureDesignator(), 'jug'), ['PouringAction', 'pour', 'jug']);
});
