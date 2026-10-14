'use strict';

// The events of a run as they are written, one row each; the screen that makes sense of
// them is built on this.

const RUN_POLL_MILLISECONDS = 2000;
const NEAR_THE_BOTTOM = 80;

const FAMILIES = [
  [/^Run/, 'run'],
  [/^(Model|Tool)/, 'model'],
  [/^Component/, 'call'],
  [/^(Resolution|Attempt)/, 'resolver'],
  [/^(Hypothesis|Generation|Knowledge)/, 'knowledge'],
  [/^(Trial|Step|Execution)/, 'simulation'],
];
const PROBLEMS = /(Failed|Rejected|Refused|Unreadable)$/;
const DETAIL_KEYS = ['source', 'target', 'operation', 'tool', 'style', 'attempt', 'kind',
  'step', 'status', 'error_type', 'record_id', 'purpose', 'action'];

const runList = document.getElementById('run-list');
const eventList = document.getElementById('events');
const connection = document.getElementById('connection');
const runTitle = document.getElementById('run-title');
const runSummary = document.getElementById('run-summary');
const runQuery = document.getElementById('run-query');

let selected = new URLSearchParams(window.location.search).get('run');
let stream = null;
let firstEventAt = null;

function familyOf(name) {
  const found = FAMILIES.find(function (family) { return family[0].test(name); });
  return found ? found[1] : 'run';
}

function describe(event) {
  return DETAIL_KEYS
    .filter(function (key) { return event[key] !== undefined && event[key] !== null; })
    .slice(0, 4)
    .map(function (key) { return key + ' ' + event[key]; })
    .join(' · ');
}

function seconds(at) {
  if (firstEventAt === null) firstEventAt = at;
  return '+' + (at - firstEventAt).toFixed(1) + 's';
}

function addEvent(event) {
  const empty = eventList.querySelector('.empty');
  if (empty) empty.remove();
  const row = document.createElement('li');
  row.className = 'event family-' + familyOf(event.event) +
    (PROBLEMS.test(event.event) ? ' problem' : '');
  const details = document.createElement('details');
  const summary = document.createElement('summary');
  [['t', seconds(event.at)], ['name', event.event], ['detail', describe(event)]]
    .forEach(function (part) {
      const cell = document.createElement('span');
      cell.className = part[0];
      cell.textContent = part[1];
      summary.appendChild(cell);
    });
  const body = document.createElement('pre');
  body.textContent = JSON.stringify(event, null, 2);
  details.append(summary, body);
  row.appendChild(details);
  const follow = eventList.scrollHeight - eventList.scrollTop - eventList.clientHeight < NEAR_THE_BOTTOM;
  eventList.appendChild(row);
  if (follow) eventList.scrollTop = eventList.scrollHeight;
}

function select(run) {
  selected = run;
  const url = new URL(window.location.href);
  url.searchParams.set('run', run);
  window.history.replaceState(null, '', url);
  if (stream) stream.close();
  eventList.innerHTML = '';
  firstEventAt = null;
  runTitle.textContent = 'Run ' + run;
  connection.textContent = 'connecting…';
  connection.className = '';
  stream = new EventSource('/api/runs/' + encodeURIComponent(run) + '/stream');
  stream.onopen = function () { connection.textContent = 'following ' + run; connection.className = 'live'; };
  stream.onerror = function () { connection.textContent = 'reconnecting…'; connection.className = ''; };
  stream.onmessage = function (message) { addEvent(JSON.parse(message.data)); };
  renderRuns(lastRuns);
}

let lastRuns = [];

function renderRuns(runs) {
  lastRuns = runs;
  runList.innerHTML = '';
  runs.forEach(function (run) {
    const item = document.createElement('li');
    if (run.run === selected) item.className = 'selected';
    const title = document.createElement('div');
    title.className = 'run-title';
    title.textContent = (run.query || run.run).split('\n')[0];
    const meta = document.createElement('div');
    meta.className = 'run-meta';
    const stage = run.execution || run.resolution || 'running';
    const badge = document.createElement('span');
    badge.className = 'badge ' + stage;
    badge.textContent = (run.execution ? 'executed ' : '') + stage;
    meta.append(badge, document.createTextNode(
      run.scene + ' · ' + run.events + ' events · $' + run.cost.toFixed(3)));
    item.append(title, meta);
    item.addEventListener('click', function () { select(run.run); });
    runList.appendChild(item);
    if (run.run === selected) showRun(run);
  });
}

function showRun(run) {
  runSummary.textContent = run.model_calls + ' model calls · $' + run.cost.toFixed(3) +
    ' · ' + run.input_tokens + ' input and ' + run.output_tokens + ' output tokens · ' + run.events + ' events';
  runQuery.hidden = !run.query;
  runQuery.textContent = run.query;
}

async function loadRuns() {
  const response = await fetch('/api/runs');
  const payload = await response.json();
  if (payload.ok) renderRuns(payload.runs);
}

loadRuns().then(function () { if (selected) select(selected); });
setInterval(loadRuns, RUN_POLL_MILLISECONDS);
