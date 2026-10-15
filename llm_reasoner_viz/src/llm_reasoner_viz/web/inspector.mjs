// What the inspector shows of a span: a title, a few facts, and blocks of detail such as
// the prompts of a model call, laid out as the sections they are made of.

import { formatCost, formatDuration } from './format.mjs';
import { KIND, OUTCOME, STATUS, USAGE, inputTokens, shortName } from './model.mjs';
import { plotOfTrials, triedPours } from './pour.mjs';
import { isJson, parseSections } from './prompts.mjs';

const NO_COMPONENT = 'no component';

// The table of the NEEM database that holds the plan of each NEEM, which its identifier names.
const NEEM_PLAN_TABLE = 'PlanMappingDAO';

// %% blocks

// The kinds of block of detail a view is made of.
export const BLOCK = Object.freeze({
  TEXT: 'text',
  JSON: 'json',
  LIST: 'list',
  FACTS: 'facts',
  TABLE: 'table',
  SECTIONS: 'sections',
  PLOT: 'plot',
});

const textBlock = (title, text) => ({ title: title, kind: BLOCK.TEXT, text: text });
export const jsonBlock = (title, value) => ({ title: title, kind: BLOCK.JSON, value: value });
const listBlock = (title, items) => ({ title: title, kind: BLOCK.LIST, items: items });
export const factsBlock = (title, facts) => ({ title: title, kind: BLOCK.FACTS, facts: facts });
const tableBlock = (title, columns, rows) => ({ title: title, kind: BLOCK.TABLE, columns: columns, rows: rows });
// Where pours ended on a plan of the opening, one of them emphasised (null when none is); no
// block when none of the pours was measured.
function plotBlock(title, trials, emphasised) {
  const plot = plotOfTrials(trials);
  return plot === null ? null : { title: title, kind: BLOCK.PLOT, plot: plot, emphasised: emphasised };
}

function finishedTrials(spans) {
  return spans.filter(function (span) { return span.kind === KIND.TRIAL && span.data.finished; });
}

const sectionsBlock = (title, text) => Object.assign({ title: title, kind: BLOCK.SECTIONS }, parseSections(text));

// Data as the object it spells when it is JSON, as text otherwise.
function dataBlock(title, value) {
  if (typeof value !== 'string') return jsonBlock(title, value);
  return isJson(value) ? jsonBlock(title, JSON.parse(value)) : textBlock(title, value);
}

// The name an identifier ends in: the part after the "#" of an address, else its last part.
function nameOfIdentifier(identifier) {
  const parts = identifier.split('#');
  return parts.length > 1 ? parts.pop() : shortName(identifier.split('/').pop());
}

function linkOf(span) {
  return { label: span.label, id: span.id, status: span.status };
}

// The kinds of work whose name alone does not say what they were, among others in a list.
const WORK_NAMES = new Map([[KIND.MODEL, 'model call'], [KIND.CALL, 'call']]);

function namedLinkOf(span) {
  const work = WORK_NAMES.get(span.kind);
  return Object.assign(linkOf(span), { label: work ? work + ' · ' + span.label : span.label });
}

function linksTo(model, span, kinds, describe = linkOf) {
  return model.childrenOf(span).filter((child) => kinds.includes(child.kind)).map(describe);
}

function present(block) {
  return block !== null && !(block.items && block.items.length === 0);
}

function lasted(span) {
  return span.end === null ? [] : [['lasted', formatDuration(span.end - span.start)]];
}

function failureFacts(failure) {
  return failure ? [['failed with', failure.error_type]] : [];
}

function failureBlocks(failure) {
  return failure ? [textBlock('Failure', failure.reason)] : [];
}

// The number of the NEEM a span kept, which the view leads to; null when it kept none.
function keptNeem(span) {
  return span.data.neem ? span.data.neem.database_id : null;
}

function view(span, title, facts, blocks) {
  return { title: title, kind: span.kind, status: span.status, neem: keptNeem(span), facts: facts, blocks: blocks.filter(present) };
}

// %% the NEEM kept of what the simulation performed

function neemFacts(span) {
  const { neem, neemFailure } = span.data;
  if (neem) return [['NEEM', 'saved as plan ' + neem.database_id]];
  if (neemFailure) return [['NEEM', 'not saved: ' + neemFailure.error_type]];
  return [];
}

// Where a NEEM is kept and how to read it, or why it was not kept; nothing when none was
// asked for.
function neemBlocks(span) {
  const { neem, neemFailure } = span.data;
  if (neem) {
    return [
      factsBlock('NEEM', [['plan', neem.database_id], ['database', neem.database], ['saved in', formatDuration(neem.seconds)]]),
      textBlock('Read it', 'sqlite3 "' + neem.database + '" "select * from ' + NEEM_PLAN_TABLE +
        ' where database_id = ' + neem.database_id + ';"'),
    ];
  }
  return neemFailure ? [textBlock('NEEM not saved', neemFailure.reason)] : [];
}

// %% a pour tried in the simulation

// A pour written as it ended: where the mouth of the pitcher was against the opening.
export function trialText(outcome) {
  if (outcome[OUTCOME.MOTION_FAILED]) return 'the motion failed: ' + outcome[OUTCOME.MOTION_FAILED];
  const verdict = outcome[OUTCOME.LANDS_IN_OPENING] ? 'landed' : 'missed';
  if (outcome[OUTCOME.DISTANCE] === undefined) return verdict;
  return verdict + ': the mouth ' + outcome[OUTCOME.DISTANCE] + ' m from the opening\'s centre (' +
    outcome[OUTCOME.AHEAD] + ' m ahead, ' + outcome[OUTCOME.RIGHT] + ' m right), ' +
    outcome[OUTCOME.ABOVE] + ' m above it';
}

// %% views by kind

function describeModel(span, model) {
  const { asked, answered, failure } = span.data;
  const usage = answered && answered.usage;
  const facts = [['asked by', asked.source], ['model', asked.identity], ['style', asked.style]];
  if (answered) facts.push(['took', formatDuration(answered.seconds)]);
  if (usage) {
    facts.push(['input tokens', inputTokens(usage)]);
    if (usage[USAGE.CACHE_READ]) facts.push(['read from the cache', usage[USAGE.CACHE_READ]]);
    if (usage[USAGE.CACHE_WRITTEN]) facts.push(['written to the cache', usage[USAGE.CACHE_WRITTEN]]);
    facts.push(['output tokens', usage[USAGE.OUTPUT]]);
  }
  if (answered && typeof answered.cost === 'number') facts.push(['cost', formatCost(answered.cost)]);
  facts.push(...failureFacts(failure));
  return view(span, 'Model call by the ' + asked.source, facts, [
    sectionsBlock('System prompt', asked.system),
    sectionsBlock('User prompt', asked.user),
    asked.schema ? jsonBlock('Schema', asked.schema) : null,
    listBlock('Tools offered', asked.tools.map((name) => ({ label: name }))),
    listBlock('Tool calls', linksTo(model, span, [KIND.TOOL, KIND.REFUSED])),
    answered ? textBlock('Answer', answered.answer) : null,
    ...failureBlocks(failure),
  ].filter((block) => block !== null));
}

function describeTool(span) {
  const { called, returned, failure } = span.data;
  const facts = [['tool', called.tool], ['belongs to', called.owner || NO_COMPONENT]];
  if (returned) facts.push(['took', formatDuration(returned.seconds)]);
  facts.push(...failureFacts(failure));
  return view(span, 'Tool call: ' + called.tool, facts, [
    jsonBlock('Input', called.arguments),
    returned ? dataBlock('Returned', returned.content) : null,
    ...failureBlocks(failure),
  ].filter((block) => block !== null));
}

function describeRefusal(span) {
  const { refused } = span.data;
  return view(span, 'Tool call refused: ' + refused.tool, [['tool', refused.tool]], [
    jsonBlock('Input', refused.arguments),
    dataBlock('Refusal', refused.content),
  ]);
}

function describeCall(span) {
  const { called, returned, failure } = span.data;
  const facts = [['from', called.source], ['to', called.target], ['operation', called.operation]];
  if (returned) facts.push(['took', formatDuration(returned.seconds)]);
  facts.push(...failureFacts(failure));
  return view(span, called.source + ' calls ' + called.target + ': ' + called.operation, facts, [
    jsonBlock('Asked', called.arguments),
    returned ? dataBlock('Answered', returned.result) : null,
    ...failureBlocks(failure),
  ].filter((block) => block !== null));
}

function describeResolution(span, model) {
  const { started, finished, consulted, failure } = span.data;
  const facts = [['purpose', started.purpose], ['action', started.action],
    ['repairs allowed', started.max_repairs]];
  if (finished) facts.push(['attempts', finished.attempts]);
  facts.push(...lasted(span), ...failureFacts(failure));
  const slots = tableBlock('Slots', ['Slot', 'Kind', 'Type', 'Candidates'],
    started.slots.map((slot) => [slot.path, slot.kind, slot.type_name, slot.candidates === null ? '' : slot.candidates]));
  const knowledge = consulted && factsBlock('Knowledge consulted', [
    ['record', consulted.record_id || 'none'],
    ['designator', consulted.designator],
    ['situation features', Object.entries(consulted.situation).map(([key, value]) => key + ' = ' + value).join(', ') || 'none'],
  ]);
  const why = finished && listBlock('Why', finished.justifications.map((justification) => ({
    label: justification.why,
    note: justification.grounds.map((ground) => ground.source + ': ' + nameOfIdentifier(ground.identifier)).join(', '),
  })));
  return view(span, 'Resolution of ' + span.label, facts, [
    slots,
    knowledge || null,
    listBlock('Attempts', linksTo(model, span, [KIND.ATTEMPT])),
    plotBlock('Pours tried', triedPours(model, span), null),
    finished ? listBlock('Resolved to', finished.instances.map((name) => ({ label: name }))) : null,
    why || null,
    ...failureBlocks(failure),
  ].filter((block) => block !== null));
}

function describeAttempt(span, model) {
  const { rejected } = span.data;
  const facts = rejected ? [['rejected for', rejected.kind]] : [];
  facts.push(...lasted(span));
  return view(span, 'Attempt ' + span.label.replace('attempt ', '') + ' at ' + model.byId.get(span.parent).label, facts, [
    rejected ? jsonBlock('What the failure found', rejected.details) : null,
    rejected ? textBlock('Answer rejected', rejected.answer) : null,
    listBlock('During the attempt', linksTo(model, span, [KIND.MODEL, KIND.CALL, KIND.TRIAL], namedLinkOf)),
  ].filter((block) => block !== null));
}

function describeHypothesis(span, model) {
  const { requested, served, failure } = span.data;
  const facts = [['kind', shortName(requested.root_type)]];
  if (served) {
    facts.push(['from the store', served.from_store ? 'yes' : 'no'], ['record', served.record_id],
      ['attempts', served.attempts]);
  }
  facts.push(...lasted(span), ...failureFacts(failure));
  return view(span, 'Knowledge of ' + shortName(requested.root_type), facts, [
    textBlock('Instruction', requested.instruction),
    jsonBlock('Fixed values', requested.fixed_values),
    listBlock('Attempts', linksTo(model, span, [KIND.GENERATION_ATTEMPT])),
    served ? listBlock('Warnings', served.warnings.map((warning) => ({ label: warning }))) : null,
    failure ? listBlock('Problems', failure.problems.map((problem) => ({ label: problem }))) : null,
  ].filter((block) => block !== null));
}

function describeGenerationAttempt(span, model) {
  const { rejected } = span.data;
  return view(span, 'Generation ' + span.label, lasted(span), [
    rejected ? listBlock('Problems found', rejected.problems.map((problem) => ({ label: problem }))) : null,
    listBlock('During the attempt', linksTo(model, span, [KIND.MODEL], namedLinkOf)),
  ].filter((block) => block !== null));
}

function describeTrial(span, model) {
  const { started, finished } = span.data;
  const facts = [];
  if (finished) facts.push(['result', trialText(finished.outcome)], ['took', formatDuration(finished.seconds)]);
  facts.push(...neemFacts(span));
  return view(span, 'Pour tried in the simulation', facts, [
    finished ? plotBlock('Where the mouth ended', finishedTrials(model.spans), span.id) : null,
    jsonBlock('Values tried', started.values),
    finished ? jsonBlock('Outcome', finished.outcome) : null,
    ...neemBlocks(span),
  ].filter((block) => block !== null));
}

function describeStep(span) {
  return view(span, 'Step: ' + span.label, [['status', span.data.changed.status], ...lasted(span)], []);
}

function describeExecution(span, model) {
  const { started, failure } = span.data;
  return view(span, 'Execution of the plan', [['run', started.run], ...lasted(span), ...neemFacts(span), ...failureFacts(failure)], [
    listBlock('Steps', linksTo(model, span, [KIND.STEP])),
    ...neemBlocks(span),
    ...failureBlocks(failure),
  ]);
}

function describeRun(span) {
  const { started, failure } = span.data;
  return view(span, 'Run', [['scene', started.scene], ['run', started.run], ...lasted(span), ...failureFacts(failure)], [
    textBlock('Query', started.query),
    ...failureBlocks(failure),
  ]);
}

function describeUnknown(span) {
  return view(span, span.label, lasted(span), [jsonBlock('Events', span.events)]);
}

const DESCRIBERS = new Map([
  [KIND.MODEL, describeModel],
  [KIND.TOOL, describeTool],
  [KIND.REFUSED, describeRefusal],
  [KIND.CALL, describeCall],
  [KIND.RESOLUTION, describeResolution],
  [KIND.ATTEMPT, describeAttempt],
  [KIND.HYPOTHESIS, describeHypothesis],
  [KIND.GENERATION_ATTEMPT, describeGenerationAttempt],
  [KIND.TRIAL, describeTrial],
  [KIND.STEP, describeStep],
  [KIND.EXECUTION, describeExecution],
  [KIND.RUN, describeRun],
]);

// The view of a span: its title, how it ended, a few facts, and blocks of detail.
export function describeSpan(span, model) {
  const describe = DESCRIBERS.get(span.kind) || describeUnknown;
  return describe(span, model);
}
