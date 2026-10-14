// The screen: follows one run's events as they are written and keeps the map of the
// components, the timeline and the inspector in step with them.

import { formatCost, formatDuration } from './format.mjs';
import { describeSpan } from './inspector.mjs';
import { EventsView } from './events_view.mjs';
import { InspectorView } from './inspector_view.mjs';
import { mapState } from './map.mjs';
import { MapView } from './map_view.mjs';
import { KIND, RunModel, STATUS } from './model.mjs';
import { Replay } from './replay.mjs';
import { ReplayView } from './replay_view.mjs';
import { TimelineView } from './timeline_view.mjs';

const RUN_POLL_MILLISECONDS = 2000;
const REFRESH_MILLISECONDS = 250;
const STALE_SECONDS = 900;
const RUN_NAME_LENGTH = 8;
const QUERY_LENGTH = 48;
const FRAME_MILLISECONDS = 33;
const LONGEST_FRAME_SECONDS = 0.25;

const Address = Object.freeze({
  RUNS: '/api/runs',
  stream: function (run) { return '/api/runs/' + encodeURIComponent(run) + '/stream'; },
});

const Parameter = Object.freeze({ RUN: 'run', SPAN: 'span' });

const Connection = Object.freeze({
  CONNECTING: 'connecting…',
  FOLLOWING: 'following',
  RECONNECTING: 'reconnecting…',
  UNREACHABLE: 'the viewer server cannot be reached',
});

// How a run stands, as the top bar says it; the first two are the statuses of work.
const Standing = Object.freeze({
  RUNNING: STATUS.RUNNING,
  FAILED: STATUS.FAILED,
  UNFINISHED: 'unfinished',
  FINISHED: 'finished',
  NOTHING: '',
});

const SECONDS_PER_MILLISECOND = 1 / 1000;

function wallClock() {
  return Date.now() * SECONDS_PER_MILLISECOND;
}

function shortened(text, length) {
  const line = text.split('\n')[0];
  return line.length > length ? line.slice(0, length - 1) + '…' : line;
}

class Viewer {
  constructor() {
    const find = (id) => document.getElementById(id);
    this.runSelect = find('run-select');
    this.statusBadge = find('run-status');
    this.totals = find('totals');
    this.connection = find('connection');
    this.drawer = find('events-drawer');
    this.eventsToggle = find('events-toggle');
    this.replayToggle = find('replay-toggle');
    this.replayBar = find('replay-bar');
    this.replayView = new ReplayView(this.replayBar, () => this.replayChanged());
    this.map = new MapView(find('map'), (lane) => this.chooseLane(lane));
    this.timeline = new TimelineView(find('timeline'), (id) => this.choose(id));
    this.inspector = new InspectorView(find('inspector'), (id) => this.choose(id));
    this.events = new EventsView(find('events'));
    const parameters = new URLSearchParams(window.location.search);
    this.wantedRun = parameters.get(Parameter.RUN);
    this.wantedSpan = parameters.get(Parameter.SPAN);
    this.runs = [];
    this.runOptions = '';
    this.run = null;
    this.model = new RunModel();
    this.stream = null;
    this.chosenId = null;
    this.chosenLane = null;
    this.inspected = null;
    this.changed = true;
    this.replay = null;
    this.lastFrame = null;
    this.framing = false;
  }

  start() {
    this.runSelect.addEventListener('change', () => this.followRun(this.runSelect.value));
    this.eventsToggle.addEventListener('click', () => {
      this.drawer.hidden = !this.drawer.hidden;
      this.eventsToggle.setAttribute('aria-expanded', String(!this.drawer.hidden));
      this.eventsToggle.classList.toggle('on', !this.drawer.hidden);
    });
    this.replayToggle.addEventListener('click', () => this.toggleReplay());
    this.loadRuns();
    setInterval(() => this.loadRuns(), RUN_POLL_MILLISECONDS);
    setInterval(() => { if (this.changed || this.isLive()) this.render(); }, REFRESH_MILLISECONDS);
  }

  // %% runs

  async loadRuns() {
    const response = await fetch(Address.RUNS).catch(() => null);
    if (response === null) {
      this.showConnection(Connection.UNREACHABLE, false);
      return;
    }
    const payload = await response.json();
    if (!payload.ok) return;
    this.runs = payload.runs;
    this.showRuns();
    if (this.run === null && this.runs.length > 0) {
      const wanted = this.runs.find((run) => run.run === this.wantedRun);
      this.followRun((wanted || this.runs[0]).run);
    }
    if (this.runs.length === 0) this.inspector.empty('No run yet. Resolve a designator in cram-viz, or point the viewer at a directory of runs.');
  }

  showRuns() {
    const signature = JSON.stringify([this.run, this.runs.map((run) => [run.run, run.resolution, run.execution])]);
    if (signature === this.runOptions) return;
    this.runOptions = signature;
    this.runSelect.replaceChildren(...this.runs.map((run) => {
      const stage = run.execution ? 'executed' : run.resolution;
      const option = document.createElement('option');
      option.value = run.run;
      option.selected = run.run === this.run;
      option.textContent = [run.run.slice(0, RUN_NAME_LENGTH), run.scene, stage, shortened(run.query, QUERY_LENGTH)]
        .filter(Boolean).join(' · ');
      return option;
    }));
  }

  followRun(run) {
    if (this.stream) this.stream.close();
    this.replay = null;
    this.run = run;
    this.model = new RunModel();
    this.events.clear();
    this.chosenId = null;
    this.chosenLane = null;
    this.inspected = null;
    this.setAddress(Parameter.RUN, run);
    this.setAddress(Parameter.SPAN, null);
    this.showConnection(Connection.CONNECTING, false);
    this.stream = new EventSource(Address.stream(run));
    this.stream.onopen = () => this.showConnection(Connection.FOLLOWING + ' ' + run.slice(0, RUN_NAME_LENGTH), true);
    this.stream.onerror = () => this.showConnection(Connection.RECONNECTING, false);
    this.stream.onmessage = (message) => {
      const event = JSON.parse(message.data);
      this.model.add(event);
      this.events.add(event);
      this.changed = true;
    };
    this.runOptions = '';
    this.showRuns();
  }

  // %% replay

  toggleReplay() {
    if (this.replay !== null) {
      this.replay = null;
    } else {
      this.replay = new Replay(this.model.events);
      this.replay.play();
    }
    this.replayChanged();
  }

  replayChanged() {
    this.render();
    if (this.replay !== null && this.replay.playing && !this.framing) this.runFrames();
  }

  // Moves a replay that is playing on with the time that passes, a frame at a time.
  runFrames() {
    this.framing = true;
    this.lastFrame = null;
    const frame = (timestamp) => {
      if (this.replay === null || !this.replay.playing) {
        this.framing = false;
        return;
      }
      const passed = this.lastFrame === null ? 0 : timestamp - this.lastFrame;
      if (this.lastFrame !== null && passed < FRAME_MILLISECONDS) {
        requestAnimationFrame(frame);
        return;
      }
      this.lastFrame = timestamp;
      this.replay.advance(Math.min(passed * SECONDS_PER_MILLISECOND, LONGEST_FRAME_SECONDS));
      this.render();
      requestAnimationFrame(frame);
    };
    requestAnimationFrame(frame);
  }

  // %% what is chosen

  choose(id) {
    this.chosenId = id;
    this.setAddress(Parameter.SPAN, id);
    this.render();
  }

  chooseLane(lane) {
    this.chosenLane = this.chosenLane === lane ? null : lane;
    this.render();
  }

  // Chooses what was asked for in the address once it is there, and a span to start with.
  // A replay keeps the choice though what is chosen has not happened yet.
  settleChoice(model) {
    if (this.wantedSpan !== null && model.byId.has(this.wantedSpan)) {
      this.chosenId = this.wantedSpan;
      this.wantedSpan = null;
    }
    if (this.chosenId !== null && (model.byId.has(this.chosenId) || this.replay !== null)) return;
    const first = model.spans.find((span) => span.kind === KIND.RUN) || model.spans[0];
    this.chosenId = first ? first.id : null;
  }

  // Puts a choice in the address, so that reloading the page returns to it; null removes it.
  setAddress(name, value) {
    const address = new URL(window.location.href);
    if (value === null) address.searchParams.delete(name);
    else address.searchParams.set(name, value);
    window.history.replaceState(null, '', address);
  }

  // %% drawing

  isLive() {
    const going = this.model.spans.some((span) => span.end === null);
    return going && this.model.lastAt !== null && wallClock() - this.model.lastAt < STALE_SECONDS;
  }

  showConnection(text, following) {
    this.connection.textContent = text;
    this.connection.classList.toggle('following', following);
  }

  // What is shown: the run as of a moment, that moment, how far the axis of the timeline
  // reaches, and whether the run is going on at that moment.
  moment() {
    if (this.replay !== null) {
      return { model: this.replay.modelAt(), now: this.replay.time, until: this.replay.end, live: !this.replay.atEnd };
    }
    const live = this.isLive();
    const now = live ? wallClock() : (this.model.lastAt === null ? 0 : this.model.lastAt);
    return { model: this.model, now: now, until: now, live: live };
  }

  statusOf(model, live) {
    const roots = model.spans.filter((span) => span.kind === KIND.RUN || span.kind === KIND.EXECUTION);
    if (roots.some((span) => span.status === STATUS.FAILED)) return Standing.FAILED;
    if (roots.some((span) => span.end === null)) return live ? Standing.RUNNING : Standing.UNFINISHED;
    return roots.length ? Standing.FINISHED : Standing.NOTHING;
  }

  showTotals(model, now) {
    const { modelCalls, cost, inputTokens, outputTokens } = model.totals;
    const elapsed = model.firstAt === null ? 0 : now - model.firstAt;
    const parts = [
      modelCalls + (modelCalls === 1 ? ' model call' : ' model calls'),
      inputTokens + ' tokens in, ' + outputTokens + ' out',
      formatCost(cost),
      formatDuration(elapsed),
    ];
    this.totals.textContent = parts.join(' · ');
  }

  showInspector(model) {
    const span = model.byId.get(this.chosenId);
    if (!span) {
      this.inspected = null;
      return this.inspector.empty(this.replay === null ? 'Nothing to inspect yet.' : 'This has not happened yet at this point of the replay.');
    }
    const children = model.childrenOf(span);
    const signature = [span.id, span.status, span.end, span.events.length, children.length,
      children.map((child) => child.status).join()].join('|');
    if (signature === this.inspected) return;
    this.inspected = signature;
    const trail = [];
    for (let above = model.byId.get(span.parent); above; above = model.byId.get(above.parent)) {
      trail.unshift({ id: above.id, label: above.label });
    }
    this.inspector.show(describeSpan(span, model), trail);
  }

  render() {
    this.changed = false;
    const { model, now, until, live } = this.moment();
    this.settleChoice(model);
    const status = this.statusOf(model, live);
    this.statusBadge.textContent = status;
    this.statusBadge.className = 'badge status-' + status;
    this.replayBar.hidden = this.replay === null;
    this.replayToggle.classList.toggle('on', this.replay !== null);
    this.replayToggle.disabled = this.replay === null && (this.isLive() || this.model.events.length === 0);
    if (this.replay !== null) this.replayView.show(this.replay);
    this.showTotals(model, now);
    this.map.update(mapState(model, live ? now : Number.POSITIVE_INFINITY), this.chosenLane);
    this.timeline.render(model, { now: now, until: until, live: live, chosenId: this.chosenId, chosenLane: this.chosenLane });
    this.showInspector(model);
  }
}

new Viewer().start();
