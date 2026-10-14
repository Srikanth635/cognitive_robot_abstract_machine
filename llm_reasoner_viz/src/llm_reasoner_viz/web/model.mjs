// What a run did, built from its events: every piece of work is a span with a start, an
// end and how it ended, nested in the span it happened during. The model takes events one
// at a time, so a run that is still going is followed as it grows.

export const LANE = Object.freeze({
  RUN: 'run',
  RESOLVER: 'resolver',
  WORLD: 'world_injector',
  KNOWLEDGE: 'knowledge',
  MODEL: 'language_model',
  SIMULATION: 'simulation',
});

// The component a call is made to, as an event names it, where it is not a lane's name.
const CALLER = 'caller';

export const STATUS = Object.freeze({
  RUNNING: 'running',
  OK: 'ok',
  REJECTED: 'rejected',
  FAILED: 'failed',
});

export const KIND = Object.freeze({
  RUN: 'run',
  EXECUTION: 'execution',
  STEP: 'step',
  RESOLUTION: 'resolution',
  ATTEMPT: 'attempt',
  MODEL: 'model',
  TOOL: 'tool',
  REFUSED: 'refused',
  CALL: 'call',
  HYPOTHESIS: 'hypothesis',
  GENERATION_ATTEMPT: 'generation',
  TRIAL: 'trial',
});

export const EVENT = Object.freeze({
  RUN_STARTED: 'RunStarted',
  RUN_FINISHED: 'RunFinished',
  RUN_FAILED: 'RunFailed',
  EXECUTION_STARTED: 'ExecutionStarted',
  EXECUTION_FINISHED: 'ExecutionFinished',
  EXECUTION_FAILED: 'ExecutionFailed',
  STEP_CHANGED: 'StepChanged',
  RESOLUTION_STARTED: 'ResolutionStarted',
  KNOWLEDGE_CONSULTED: 'KnowledgeConsulted',
  ATTEMPT_STARTED: 'AttemptStarted',
  ATTEMPT_REJECTED: 'AttemptRejected',
  RESOLUTION_FINISHED: 'ResolutionFinished',
  RESOLUTION_FAILED: 'ResolutionFailed',
  MODEL_ASKED: 'ModelAsked',
  MODEL_ANSWERED: 'ModelAnswered',
  MODEL_FAILED: 'ModelFailed',
  TOOL_CALLED: 'ToolCalled',
  TOOL_RETURNED: 'ToolReturned',
  TOOL_FAILED: 'ToolFailed',
  TOOL_REFUSED: 'ToolRefused',
  COMPONENT_CALLED: 'ComponentCalled',
  COMPONENT_RETURNED: 'ComponentReturned',
  COMPONENT_FAILED: 'ComponentFailed',
  HYPOTHESIS_REQUESTED: 'HypothesisRequested',
  GENERATION_ATTEMPT_STARTED: 'GenerationAttemptStarted',
  GENERATION_ATTEMPT_REJECTED: 'GenerationAttemptRejected',
  GENERATION_FAILED: 'GenerationFailed',
  HYPOTHESIS_SERVED: 'HypothesisServed',
  TRIAL_STARTED: 'TrialStarted',
  TRIAL_FINISHED: 'TrialFinished',
});

// The keys of the outcome of a pour tried in the simulation. Lengths are in metres, as seen
// from the robot's base: ahead along its heading, right to its right.
export const OUTCOME = Object.freeze({
  LANDS_IN_OPENING: 'lands_in_opening',
  MOTION_FAILED: 'motion_failed',
  OPENING_RADIUS: 'opening_radius',
  DISTANCE: 'mouth_distance_from_opening_centre',
  AHEAD: 'mouth_ahead_of_opening_centre',
  RIGHT: 'mouth_right_of_opening_centre',
  ABOVE: 'mouth_above_opening',
});

// The statuses of a step of the plan that end it; any other leaves it as it is.
const STEP_STATUS = Object.freeze({
  RUNNING: 'RUNNING',
  SUCCEEDED: 'SUCCEEDED',
  FAILED: 'FAILED',
  INTERRUPTED: 'INTERRUPTED',
});

// The keys of the tokens a model answered with: the input read without the cache, the tokens
// generated, and those written to the cache and read from it.
export const USAGE = Object.freeze({
  INPUT: 'input_tokens',
  OUTPUT: 'output_tokens',
  CACHE_WRITTEN: 'cache_creation_input_tokens',
  CACHE_READ: 'cache_read_input_tokens',
});

// Everything the model read: the input, with what went to the cache and came from it.
export function inputTokens(usage) {
  return (usage[USAGE.INPUT] || 0) + (usage[USAGE.CACHE_WRITTEN] || 0) + (usage[USAGE.CACHE_READ] || 0);
}

const ACTION_SUFFIX = /Action$/;
const WORD_BOUNDARY = /([a-z])([A-Z])/g;

// The last part of a dotted name: a class from its module path.
export function shortName(dotted) {
  return dotted.split('.').pop();
}

// The name of an action written for reading: ParkArmsAction as "park arms".
export function humanize(actionName) {
  return actionName.replace(ACTION_SUFFIX, '').replace(WORD_BOUNDARY, '$1 $2').toLowerCase();
}

export function spanEnd(span, now) {
  return span.end === null ? now : span.end;
}

function laneOfComponent(component) {
  return component === CALLER ? LANE.RUN : component;
}

// How a connection from one component to another is named in ``RunModel.edges``.
export function edgeKey(from, to) {
  return from + '>' + to;
}

export class RunModel {
  constructor() {
    this.events = [];
    this.spans = [];
    this.byId = new Map();
    // How many calls went from one component to another, by "from>to", and when the last did.
    this.edges = new Map();
    this.edgeLastAt = new Map();
    this.totals = { modelCalls: 0, cost: 0, inputTokens: 0, outputTokens: 0 };
    this.firstAt = null;
    this.lastAt = null;
    // The spans that work happens inside of, innermost last.
    this.enclosing = [];
    this.execution = null;
    this.step = null;
    this.hypotheses = 0;
    this.trials = 0;
    this.steps = 0;
  }

  add(event) {
    this.events.push(event);
    if (typeof event.at === 'number') this.date(event.at);
    const handle = HANDLERS.get(event.event);
    if (handle) handle.call(this, event);
  }

  date(at) {
    if (this.firstAt === null) this.firstAt = at;
    this.lastAt = this.lastAt === null ? at : Math.max(this.lastAt, at);
  }

  childrenOf(span) {
    return this.spans.filter(function (other) { return other.parent === span.id; });
  }

  // %% spans

  innermost() {
    return this.enclosing.length === 0 ? null : this.enclosing[this.enclosing.length - 1];
  }

  innermostOfKind(kind) {
    for (let index = this.enclosing.length - 1; index >= 0; index -= 1) {
      const span = this.byId.get(this.enclosing[index]);
      if (span.kind === kind) return span;
    }
    return null;
  }

  uniqueId(wanted) {
    let id = wanted;
    let count = 1;
    while (this.byId.has(id)) {
      count += 1;
      id = wanted + '#' + count;
    }
    return id;
  }

  begin(kind, id, lane, label, event, options = {}) {
    const span = {
      id: this.uniqueId(id),
      kind: kind,
      lane: lane,
      label: label,
      start: event.at,
      end: null,
      status: STATUS.RUNNING,
      parent: options.parent === undefined ? this.innermost() : options.parent,
      events: [event],
      data: {},
    };
    this.spans.push(span);
    this.byId.set(span.id, span);
    if (options.encloses) this.enclosing.push(span.id);
    return span;
  }

  end(span, event, status) {
    span.end = event.at;
    span.status = status;
    span.events.push(event);
    this.enclosing = this.enclosing.filter(function (id) { return id !== span.id; });
  }

  // Ends a span that is still going when the work it was part of ends, unless it has ended.
  endUnlessEnded(span, event, status) {
    if (span !== null && span.end === null) this.end(span, event, status);
  }

  bump(from, to, at) {
    const key = edgeKey(from, to);
    this.edges.set(key, (this.edges.get(key) || 0) + 1);
    this.edgeLastAt.set(key, at);
  }

  // %% a run

  runStarted(event) {
    const span = this.begin(KIND.RUN, 'run:' + event.run, LANE.RUN, 'run', event, { parent: null, encloses: true });
    span.data.started = event;
  }

  runFinished(event) {
    const span = this.byId.get('run:' + event.run);
    if (!span) return;
    span.data.finished = event;
    this.end(span, event, STATUS.OK);
  }

  runFailed(event) {
    const span = this.byId.get('run:' + event.run);
    if (!span) return;
    span.data.failure = event;
    this.end(span, event, STATUS.FAILED);
  }

  // %% the execution of the plan

  executionStarted(event) {
    this.execution = this.begin(KIND.EXECUTION, 'execution:' + event.run, LANE.RUN, 'execution', event, { parent: null });
    this.execution.data.started = event;
    this.step = null;
  }

  stepChanged(event) {
    const name = humanize(event.step);
    if (event.status === STEP_STATUS.RUNNING) {
      if (this.step !== null && this.step.end === null && this.step.label === name) return;
      this.endUnlessEnded(this.step, event, STATUS.OK);
      this.steps += 1;
      this.step = this.begin(KIND.STEP, 'step:' + this.steps, LANE.SIMULATION, name, event,
        { parent: this.execution === null ? null : this.execution.id });
      this.step.data.changed = event;
      return;
    }
    if (this.step === null || this.step.label !== name) return;
    if (event.status === STEP_STATUS.SUCCEEDED) this.endUnlessEnded(this.step, event, STATUS.OK);
    if (event.status === STEP_STATUS.FAILED || event.status === STEP_STATUS.INTERRUPTED) {
      this.endUnlessEnded(this.step, event, STATUS.FAILED);
    }
  }

  executionFinished(event) {
    this.endUnlessEnded(this.step, event, STATUS.OK);
    if (this.execution === null) return;
    this.execution.data.finished = event;
    this.end(this.execution, event, STATUS.OK);
  }

  executionFailed(event) {
    this.endUnlessEnded(this.step, event, STATUS.FAILED);
    if (this.execution === null) return;
    this.execution.data.failure = event;
    this.end(this.execution, event, STATUS.FAILED);
  }

  // %% a resolution

  resolutionStarted(event) {
    const name = shortName(event.action);
    const span = this.begin(KIND.RESOLUTION, 'resolution:' + event.resolution, LANE.RESOLVER,
      name, event, { encloses: true });
    span.data.started = event;
  }

  knowledgeConsulted(event) {
    const span = this.byId.get('resolution:' + event.resolution);
    if (!span) return;
    span.data.consulted = event;
    span.events.push(event);
  }

  attemptStarted(event) {
    const resolution = 'resolution:' + event.resolution;
    const span = this.begin(KIND.ATTEMPT, 'attempt:' + event.resolution + ':' + event.attempt,
      LANE.RESOLVER, 'attempt ' + (event.attempt + 1), event,
      { parent: this.byId.has(resolution) ? resolution : null, encloses: true });
    span.data.started = event;
  }

  attemptRejected(event) {
    const span = this.byId.get('attempt:' + event.resolution + ':' + event.attempt);
    if (!span) return;
    span.data.rejected = event;
    this.end(span, event, STATUS.REJECTED);
  }

  endAttemptsOf(resolution, event, status) {
    const model = this;
    this.childrenOf(resolution).forEach(function (child) {
      if (child.kind === KIND.ATTEMPT) model.endUnlessEnded(child, event, status);
    });
  }

  resolutionFinished(event) {
    const span = this.byId.get('resolution:' + event.resolution);
    if (!span) return;
    this.endAttemptsOf(span, event, STATUS.OK);
    span.data.finished = event;
    this.end(span, event, STATUS.OK);
  }

  resolutionFailed(event) {
    const span = this.byId.get('resolution:' + event.resolution);
    if (!span) return;
    this.endAttemptsOf(span, event, STATUS.FAILED);
    span.data.failure = event;
    this.end(span, event, STATUS.FAILED);
  }

  // %% the language model and its tools

  modelAsked(event) {
    const span = this.begin(KIND.MODEL, 'model:' + event.call, LANE.MODEL, event.source, event);
    span.data.asked = event;
    this.bump(event.source, LANE.MODEL, event.at);
  }

  modelAnswered(event) {
    this.totals.modelCalls += 1;
    this.totals.cost += event.cost || 0;
    if (event.usage) {
      this.totals.inputTokens += inputTokens(event.usage);
      this.totals.outputTokens += event.usage[USAGE.OUTPUT] || 0;
    }
    const span = this.byId.get('model:' + event.call);
    if (!span) return;
    span.data.answered = event;
    this.end(span, event, STATUS.OK);
  }

  modelFailed(event) {
    const span = this.byId.get('model:' + event.call);
    if (!span) return;
    span.data.failure = event;
    this.end(span, event, STATUS.FAILED);
  }

  toolCalled(event) {
    const lane = event.owner ? laneOfComponent(event.owner) : LANE.MODEL;
    const span = this.begin(KIND.TOOL, 'tool:' + event.call, lane, event.tool, event,
      { parent: 'model:' + event.parent });
    span.data.called = event;
    if (event.owner) this.bump(LANE.MODEL, laneOfComponent(event.owner), event.at);
  }

  toolReturned(event) {
    const span = this.byId.get('tool:' + event.call);
    if (!span) return;
    span.data.returned = event;
    this.end(span, event, event.is_error ? STATUS.REJECTED : STATUS.OK);
  }

  toolFailed(event) {
    const span = this.byId.get('tool:' + event.call);
    if (!span) return;
    span.data.failure = event;
    this.end(span, event, STATUS.FAILED);
  }

  toolRefused(event) {
    const span = this.begin(KIND.REFUSED, 'refused:' + event.call, LANE.MODEL, event.tool, event,
      { parent: 'model:' + event.parent });
    span.data.refused = event;
    this.end(span, event, STATUS.REJECTED);
  }

  // %% calls between components

  componentCalled(event) {
    const span = this.begin(KIND.CALL, 'call:' + event.call, laneOfComponent(event.target),
      event.operation, event);
    span.data.called = event;
    this.bump(event.source, event.target, event.at);
  }

  componentReturned(event) {
    const span = this.byId.get('call:' + event.call);
    if (!span) return;
    span.data.returned = event;
    this.end(span, event, STATUS.OK);
  }

  componentFailed(event) {
    const span = this.byId.get('call:' + event.call);
    if (!span) return;
    span.data.failure = event;
    this.end(span, event, STATUS.FAILED);
  }

  // %% the knowledge

  hypothesisRequested(event) {
    this.hypotheses += 1;
    const span = this.begin(KIND.HYPOTHESIS, 'hypothesis:' + this.hypotheses, LANE.KNOWLEDGE,
      shortName(event.root_type), event, { encloses: true });
    span.data.requested = event;
  }

  generationAttemptStarted(event) {
    const hypothesis = this.innermostOfKind(KIND.HYPOTHESIS);
    const span = this.begin(KIND.GENERATION_ATTEMPT,
      'generation:' + this.hypotheses + ':' + event.attempt, LANE.KNOWLEDGE,
      'attempt ' + (event.attempt + 1), event,
      { parent: hypothesis === null ? null : hypothesis.id, encloses: true });
    span.data.started = event;
  }

  generationAttemptRejected(event) {
    const span = this.innermostOfKind(KIND.GENERATION_ATTEMPT);
    if (span === null) return;
    span.data.rejected = event;
    this.end(span, event, STATUS.REJECTED);
  }

  hypothesisServed(event) {
    const hypothesis = this.innermostOfKind(KIND.HYPOTHESIS);
    if (hypothesis === null) return;
    this.endUnlessEnded(this.innermostOfKind(KIND.GENERATION_ATTEMPT), event, STATUS.OK);
    hypothesis.data.served = event;
    this.end(hypothesis, event, STATUS.OK);
  }

  generationFailed(event) {
    const hypothesis = this.innermostOfKind(KIND.HYPOTHESIS);
    if (hypothesis === null) return;
    this.endUnlessEnded(this.innermostOfKind(KIND.GENERATION_ATTEMPT), event, STATUS.FAILED);
    hypothesis.data.failure = event;
    this.end(hypothesis, event, STATUS.FAILED);
  }

  // %% the simulation

  trialStarted(event) {
    this.trials += 1;
    const span = this.begin(KIND.TRIAL, 'trial:' + event.trial, LANE.SIMULATION,
      'trial ' + this.trials, event);
    span.data.started = event;
    this.bump(LANE.RESOLVER, LANE.SIMULATION, event.at);
  }

  trialFinished(event) {
    const span = this.byId.get('trial:' + event.trial);
    if (!span) return;
    span.data.finished = event;
    let status = event.outcome[OUTCOME.LANDS_IN_OPENING] ? STATUS.OK : STATUS.REJECTED;
    if (event.outcome[OUTCOME.MOTION_FAILED]) status = STATUS.FAILED;
    this.end(span, event, status);
  }
}

const HANDLERS = new Map([
  [EVENT.RUN_STARTED, RunModel.prototype.runStarted],
  [EVENT.RUN_FINISHED, RunModel.prototype.runFinished],
  [EVENT.RUN_FAILED, RunModel.prototype.runFailed],
  [EVENT.EXECUTION_STARTED, RunModel.prototype.executionStarted],
  [EVENT.STEP_CHANGED, RunModel.prototype.stepChanged],
  [EVENT.EXECUTION_FINISHED, RunModel.prototype.executionFinished],
  [EVENT.EXECUTION_FAILED, RunModel.prototype.executionFailed],
  [EVENT.RESOLUTION_STARTED, RunModel.prototype.resolutionStarted],
  [EVENT.KNOWLEDGE_CONSULTED, RunModel.prototype.knowledgeConsulted],
  [EVENT.ATTEMPT_STARTED, RunModel.prototype.attemptStarted],
  [EVENT.ATTEMPT_REJECTED, RunModel.prototype.attemptRejected],
  [EVENT.RESOLUTION_FINISHED, RunModel.prototype.resolutionFinished],
  [EVENT.RESOLUTION_FAILED, RunModel.prototype.resolutionFailed],
  [EVENT.MODEL_ASKED, RunModel.prototype.modelAsked],
  [EVENT.MODEL_ANSWERED, RunModel.prototype.modelAnswered],
  [EVENT.MODEL_FAILED, RunModel.prototype.modelFailed],
  [EVENT.TOOL_CALLED, RunModel.prototype.toolCalled],
  [EVENT.TOOL_RETURNED, RunModel.prototype.toolReturned],
  [EVENT.TOOL_FAILED, RunModel.prototype.toolFailed],
  [EVENT.TOOL_REFUSED, RunModel.prototype.toolRefused],
  [EVENT.COMPONENT_CALLED, RunModel.prototype.componentCalled],
  [EVENT.COMPONENT_RETURNED, RunModel.prototype.componentReturned],
  [EVENT.COMPONENT_FAILED, RunModel.prototype.componentFailed],
  [EVENT.HYPOTHESIS_REQUESTED, RunModel.prototype.hypothesisRequested],
  [EVENT.GENERATION_ATTEMPT_STARTED, RunModel.prototype.generationAttemptStarted],
  [EVENT.GENERATION_ATTEMPT_REJECTED, RunModel.prototype.generationAttemptRejected],
  [EVENT.GENERATION_FAILED, RunModel.prototype.generationFailed],
  [EVENT.HYPOTHESIS_SERVED, RunModel.prototype.hypothesisServed],
  [EVENT.TRIAL_STARTED, RunModel.prototype.trialStarted],
  [EVENT.TRIAL_FINISHED, RunModel.prototype.trialFinished],
]);
