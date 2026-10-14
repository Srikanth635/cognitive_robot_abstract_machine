// Tests of the cards of the attempts of a resolution: what each chose and how it ended.
import test from 'node:test';
import assert from 'node:assert';
import { EVENT, KIND, STATUS } from '../../../llm_reasoner_viz/src/llm_reasoner_viz/web/model.mjs';
import { attemptCards, cardOfChoice, resolutionOfChoice } from '../../../llm_reasoner_viz/src/llm_reasoner_viz/web/attempts.mjs';
import { trialText } from '../../../llm_reasoner_viz/src/llm_reasoner_viz/web/inspector.mjs';
import { actionResolution, eventOf, loadFixture, modelOf, situationResolution, spansOfKind } from './support.mjs';

const resolved = loadFixture('resolved_run.json');

function answers() {
  return resolved.filter(function (event) { return event.event === EVENT.MODEL_ANSWERED; });
}

test('a resolution has a card for each of its attempts, in order', function () {
  const model = modelOf(resolved);
  const cards = attemptCards(model, actionResolution(model, resolved));
  assert.deepStrictEqual(cards.map(function (card) { return card.status; }), [STATUS.REJECTED, STATUS.OK]);
  assert.deepStrictEqual(cards.map(function (card) { return card.label; }), ['attempt 1', 'attempt 2']);
  cards.forEach(function (card) { assert.strictEqual(model.byId.get(card.id).kind, KIND.ATTEMPT); });
});

test('a card says what the attempt chose, slot by slot, without the name of the class', function () {
  const model = modelOf(resolved);
  const [first] = attemptCards(model, actionResolution(model, resolved));
  const rejection = resolved.find(function (event) { return event.event === EVENT.ATTEMPT_REJECTED; });
  const chosen = JSON.parse(rejection.answer)[0];
  const slots = Object.keys(chosen).filter(function (key) { return key.includes('.'); });
  assert.deepStrictEqual(first.chose.map(function (choice) { return choice.slot; }), slots.map(function (slot) { return slot.split('.').pop(); }));
  first.chose.forEach(function (choice, index) { assert.strictEqual(choice.value, chosen[slots[index]]); });
});

test('the last attempt, which was accepted, says what it chose from the answer of the model', function () {
  const model = modelOf(resolved);
  const cards = attemptCards(model, actionResolution(model, resolved));
  const lastAnswer = JSON.parse(answers()[answers().length - 1].answer)[0];
  const tilt = cards[1].chose.find(function (choice) { return choice.slot === 'tilt_angle'; });
  assert.strictEqual(tilt.value, lastAnswer['PouringAction.tilt_angle']);
});

test('a card says why the model chose as it did', function () {
  const model = modelOf(resolved);
  const [first] = attemptCards(model, actionResolution(model, resolved));
  const rejection = resolved.find(function (event) { return event.event === EVENT.ATTEMPT_REJECTED; });
  assert.strictEqual(first.why, JSON.parse(rejection.answer)[0].why);
});

test('a card of a rejected attempt says what it was rejected for, and the pour that was tried', function () {
  const model = modelOf(resolved);
  const [first, second] = attemptCards(model, actionResolution(model, resolved));
  const rejection = resolved.find(function (event) { return event.event === EVENT.ATTEMPT_REJECTED; });
  const trials = resolved.filter(function (event) { return event.event === EVENT.TRIAL_FINISHED; });
  assert.strictEqual(first.rejectedFor, rejection.kind);
  assert.strictEqual(first.outcome, trialText(trials[0].outcome));
  assert.strictEqual(second.rejectedFor, null);
  assert.strictEqual(second.outcome, trialText(trials[1].outcome));
});

test('an attempt whose answer is not data has nothing it chose', function () {
  const model = modelOf([
    eventOf(EVENT.RESOLUTION_STARTED, 1, { resolution: 'r', purpose: 'action', action: 'a.B', slots: [], max_repairs: 1 }),
    eventOf(EVENT.ATTEMPT_STARTED, 2, { resolution: 'r', attempt: 0 }),
    eventOf(EVENT.ATTEMPT_REJECTED, 3, { resolution: 'r', attempt: 0, kind: 'Unreadable', details: {}, answer: 'no idea' }),
  ]);
  const [card] = attemptCards(model, model.byId.get('resolution:r'));
  assert.deepStrictEqual(card.chose, []);
  assert.strictEqual(card.why, '');
});

test('without a resolution there are no cards', function () {
  assert.deepStrictEqual(attemptCards(modelOf([]), null), []);
});

test('the attempts shown are those of the resolution the chosen span belongs to', function () {
  const model = modelOf(resolved);
  const action = actionResolution(model, resolved);
  const attempt = model.childrenOf(action).find(function (span) { return span.kind === KIND.ATTEMPT; });
  const call = model.childrenOf(attempt).find(function (span) { return span.kind === KIND.MODEL; });
  assert.strictEqual(resolutionOfChoice(model, call.id), action);
  assert.strictEqual(resolutionOfChoice(model, attempt.id), action);
  assert.strictEqual(resolutionOfChoice(model, action.id), action);
});

test('the choice of the situation shows the attempts of that choice', function () {
  const model = modelOf(resolved);
  const situation = situationResolution(model, resolved);
  const attempt = model.childrenOf(situation).find(function (span) { return span.kind === KIND.ATTEMPT; });
  assert.strictEqual(resolutionOfChoice(model, attempt.id), situation);
});

test('with nothing chosen, or something outside any resolution, the first resolution is shown', function () {
  const model = modelOf(resolved);
  const action = actionResolution(model, resolved);
  assert.strictEqual(resolutionOfChoice(model, null), action);
  assert.strictEqual(resolutionOfChoice(model, spansOfKind(model, KIND.RUN)[0].id), action);
});

test('a run that resolved nothing has no resolution to show', function () {
  assert.strictEqual(resolutionOfChoice(modelOf([]), null), null);
});

test('the card of what is chosen is that of the attempt it was part of', function () {
  const model = modelOf(resolved);
  const action = actionResolution(model, resolved);
  const cards = attemptCards(model, action);
  const [attempt] = model.childrenOf(action).filter(function (span) { return span.kind === KIND.ATTEMPT; });
  const call = model.childrenOf(attempt).find(function (span) { return span.kind === KIND.MODEL; });
  assert.strictEqual(cardOfChoice(model, cards, call.id), cards[0].id);
  assert.strictEqual(cardOfChoice(model, cards, attempt.id), cards[0].id);
});

test('no card is that of what is chosen when it is not part of an attempt', function () {
  const model = modelOf(resolved);
  const action = actionResolution(model, resolved);
  const cards = attemptCards(model, action);
  assert.strictEqual(cardOfChoice(model, cards, action.id), null);
  assert.strictEqual(cardOfChoice(model, cards, null), null);
});
