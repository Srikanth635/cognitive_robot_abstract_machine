// Tests of reading a prompt as the tagged sections it is made of.
import test from 'node:test';
import assert from 'node:assert';
import { parseSections, isJson, prettyJson } from '../../../llm_reasoner_viz/src/llm_reasoner_viz/web/prompts.mjs';

test('a prompt is the tagged sections it is made of, in order', function () {
  const parsed = parseSections('<role>\nFill the slots.\n</role>\n\n<inputs>\nThe request.\n</inputs>\n');
  assert.deepStrictEqual(parsed.sections.map(function (s) { return s.name; }), ['role', 'inputs']);
  assert.strictEqual(parsed.sections[0].content, 'Fill the slots.');
  assert.strictEqual(parsed.sections[1].content, 'The request.');
  assert.strictEqual(parsed.loose, '');
});

test('the attributes of a tag are kept with its section', function () {
  const parsed = parseSections('<description type="PouringPhaseStructure">\nA phase.\n</description>');
  assert.strictEqual(parsed.sections[0].name, 'description');
  assert.strictEqual(parsed.sections[0].attributes, 'type="PouringPhaseStructure"');
});

test('a tag named inside a section neither starts nor ends one', function () {
  const text = '<inputs>\n- <request>: the query in English.\n- <task>: what the caller wants.\n</inputs>\n<request>\nthe(Action)()\n</request>';
  const parsed = parseSections(text);
  assert.deepStrictEqual(parsed.sections.map(function (s) { return s.name; }), ['inputs', 'request']);
  assert.ok(parsed.sections[0].content.includes('- <request>: the query in English.'));
  assert.strictEqual(parsed.sections[1].content, 'the(Action)()');
});

test('a section nesting another is one section', function () {
  const parsed = parseSections('<outer>\n<inner>\nx\n</inner>\n</outer>');
  assert.strictEqual(parsed.sections.length, 1);
  assert.strictEqual(parsed.sections[0].content, '<inner>\nx\n</inner>');
});

test('a section written on one line is a section', function () {
  const parsed = parseSections('<role>Fill the slots.</role>');
  assert.strictEqual(parsed.sections[0].name, 'role');
  assert.strictEqual(parsed.sections[0].content, 'Fill the slots.');
});

test('text outside the sections is kept apart', function () {
  const parsed = parseSections('Some words.\n<role>\nr\n</role>\nMore words.');
  assert.strictEqual(parsed.sections.length, 1);
  assert.strictEqual(parsed.loose, 'Some words.\nMore words.');
});

test('a prompt without sections is all loose text', function () {
  const parsed = parseSections('Just words.');
  assert.deepStrictEqual(parsed.sections, []);
  assert.strictEqual(parsed.loose, 'Just words.');
});

test('a section that is never closed runs to the end', function () {
  const parsed = parseSections('<role>\nunfinished');
  assert.strictEqual(parsed.sections[0].content, 'unfinished');
});

test('a section says how long it is', function () {
  const parsed = parseSections('<role>\nabcd\n</role>');
  assert.strictEqual(parsed.sections[0].length, 4);
});

test('json is recognised and written out with indentation', function () {
  assert.ok(isJson('{"a": [1, 2]}'));
  assert.ok(!isJson('not json'));
  assert.ok(!isJson('42'));
  assert.strictEqual(prettyJson('{"a":1}'), '{\n  "a": 1\n}');
});
