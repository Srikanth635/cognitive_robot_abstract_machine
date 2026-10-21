// Tests what the designator box offers: the action designators by name, and inside the
// argument list of one, the parameters it takes.
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const ROOT = path.join(__dirname, '..', '..', '..');
const COMPLETION = path.join(
  ROOT, 'cramera', 'src', 'cramera', 'web', 'core', 'completion.js',
);
const DESIGNATOR_COMPLETION = path.join(
  ROOT, 'cramera_resolution', 'src', 'cramera_resolution', 'web', 'addon',
  'designator_completion.js',
);

global.window = {};
new Function(fs.readFileSync(COMPLETION, 'utf8'))();
global.Completion = global.window.Completion;
new Function(fs.readFileSync(DESIGNATOR_COMPLETION, 'utf8'))();
const { entriesAt } = global.window.DesignatorCompletion;

const VOCABULARY = [
  { name: 'PouringAction', kind: 'class', detail: 'action designator', module: 'coraplex', owner: '' },
  { name: 'source_container', kind: 'field', detail: 'Tool', module: 'PouringAction', owner: 'PouringAction' },
  { name: 'target_container', kind: 'field', detail: 'Body', module: 'PouringAction', owner: 'PouringAction' },
  { name: 'tilt_angle', kind: 'field', detail: 'float = 1.85', module: 'PouringAction', owner: 'PouringAction' },
  { name: 'PlacingAction', kind: 'class', detail: 'action designator', module: 'coraplex', owner: '' },
  { name: 'surface', kind: 'field', detail: 'Body', module: 'PlacingAction', owner: 'PlacingAction' },
];

function offeredAtEnd(text) {
  return entriesAt(VOCABULARY, text, text.length).map(function (entry) { return entry.name; });
}

test('an empty box offers the action designators', function () {
  assert.deepStrictEqual(offeredAtEnd(''), ['PouringAction', 'PlacingAction']);
});

test('the start of a designator name is offered the designators', function () {
  assert.deepStrictEqual(offeredAtEnd('the(Pou'), ['PouringAction', 'PlacingAction']);
});

test('a designator typed in full offers nothing more', function () {
  assert.deepStrictEqual(offeredAtEnd('the(PlacingAction'), ['PouringAction']);
});

test('the argument list of a designator offers its parameters', function () {
  assert.deepStrictEqual(
    offeredAtEnd('the(PouringAction)('), ['source_container', 'target_container', 'tilt_angle'],
  );
});

test('a designator written as a call offers its parameters too', function () {
  assert.deepStrictEqual(offeredAtEnd('PlacingAction('), ['surface']);
});

test('a parameter being typed is matched by the menu against the parameters', function () {
  const text = 'the(PouringAction)(source_container=pouring_cup, ta';
  const offered = entriesAt(VOCABULARY, text, text.length);
  assert.deepStrictEqual(
    global.Completion.suggest(offered, global.Completion.tokenAt(text, text.length)).map(function (entry) { return entry.name; }),
    ['target_container', 'source_container'],
  );
});

test('only the parameters of the designator being written are offered', function () {
  assert.deepStrictEqual(offeredAtEnd('the(PlacingAction)('), ['surface']);
});

test('a parameter after the last designator named is that designator\'s', function () {
  assert.deepStrictEqual(
    offeredAtEnd('the(PouringAction)(source_container=x)\nthe(PlacingAction)('), ['surface'],
  );
});

test('the value of a parameter is not a name to complete', function () {
  assert.deepStrictEqual(offeredAtEnd('the(PouringAction)(source_container='), []);
});

test('inside a call in a value nothing is offered', function () {
  assert.deepStrictEqual(offeredAtEnd('the(PouringAction)(arm=variable('), []);
});

test('after a closed argument list the designators are offered again', function () {
  assert.deepStrictEqual(
    offeredAtEnd('the(PouringAction)(arm=x)\nthe(P'), ['PouringAction', 'PlacingAction'],
  );
});

test('a member behind a dot is not a designator or a parameter', function () {
  assert.deepStrictEqual(offeredAtEnd('the(PouringAction)(source_container=bowl.'), []);
});

test('the caret decides, not the end of the text', function () {
  const text = 'the(PouringAction)(ti, arm=x)\nthe(PlacingAction)(surface=y)';
  const caret = text.indexOf('(ti,') + 3;
  assert.deepStrictEqual(
    entriesAt(VOCABULARY, text, caret).map(function (entry) { return entry.name; }),
    ['source_container', 'target_container', 'tilt_angle'],
  );
});

test('a vocabulary with no designators offers nothing', function () {
  assert.deepStrictEqual(entriesAt([], 'the(', 4), []);
});
