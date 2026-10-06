// Tests for the semantic graph projected from a running CRAM plan.
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const WEB_ROOT = path.join(
  __dirname,
  '..',
  '..',
  '..',
  'cram_viz',
  'src',
  'cram_viz',
  'web',
);

// %% Live knowledge fixture
function loadLiveKnowledgeGraph() {
  global.window = {};
  const source = fs.readFileSync(
    path.join(WEB_ROOT, 'panels/graph/live-knowledge.js'),
    'utf8',
  );
  new Function(source)();
  return global.window.LiveKnowledgeGraph;
}

function runningPlan() {
  return {
    sig: 'running-plan',
    nodes: [
      {
        id: 'plan-root',
        parent: null,
        kind: 'SequentialNode',
        label: 'SequentialNode',
        status: 'RUNNING',
        derived: false,
      },
      {
        id: 'park-arms',
        parent: 'plan-root',
        kind: 'ActionNode',
        label: 'ParkArmsAction',
        status: 'RUNNING',
        derived: true,
        arm: 'BOTH',
      },
      {
        id: 'navigate',
        parent: 'plan-root',
        kind: 'ActionNode',
        label: 'NavigateAction',
        status: 'RUNNING',
        derived: true,
      },
      {
        id: 'navigate-second-step',
        parent: 'plan-root',
        kind: 'ActionNode',
        label: 'NavigateAction',
        status: 'RUNNING',
        derived: true,
      },
      {
        id: 'pickup',
        parent: 'plan-root',
        kind: 'ActionNode',
        label: 'PickUpAction',
        status: 'RUNNING',
        derived: true,
        arm: 'RIGHT',
        target: 'Static_MilkPitcher.stl',
      },
      {
        id: 'reach',
        parent: 'pickup',
        kind: 'ActionNode',
        label: 'ReachAction',
        status: 'RUNNING',
        derived: true,
        arm: 'RIGHT',
        target: 'Static_MilkPitcher.stl',
      },
      {
        id: 'pouring',
        parent: 'plan-root',
        kind: 'ActionNode',
        label: 'PouringAction',
        status: 'CREATED',
        derived: false,
        arm: 'RIGHT',
        target: 'bowl.stl',
      },
    ],
  };
}

function bridgeInformation() {
  return {
    robot: 'PR2',
    objects: [
      'milk.stl',
      'breakfast_cereal.stl',
      'Static_MilkPitcher.stl',
      'bowl.stl',
    ],
  };
}

// %% Live graph structure
test('live knowledge uses the running plan instead of stale recorded episodes', function () {
  const LiveKnowledgeGraph = loadLiveKnowledgeGraph();
  const payload = new LiveKnowledgeGraph(runningPlan(), bridgeInformation()).build();
  const labels = payload.nodes.map(function (node) { return node.label; });

  assert.ok(labels.includes('PR2'));
  assert.ok(labels.includes('Park Arms'));
  assert.ok(labels.includes('Navigate'));
  assert.ok(labels.includes('Pick Up'));
  assert.ok(labels.includes('Reach'));
  assert.ok(labels.includes('Pouring'));
  assert.ok(labels.includes('Static MilkPitcher'));
  assert.ok(labels.includes('bowl'));
  assert.ok(!labels.includes('transport milk'));
});

test('live knowledge merges repeated action types into one semantic node', function () {
  const LiveKnowledgeGraph = loadLiveKnowledgeGraph();
  const payload = new LiveKnowledgeGraph(runningPlan(), bridgeInformation()).build();
  const navigateNodes = payload.nodes.filter(function (node) {
    return node.label === 'Navigate';
  });
  const navigateEdges = payload.edges.filter(function (edge) {
    return edge.to === 'live-action:navigate';
  });

  assert.strictEqual(navigateNodes.length, 1);
  assert.strictEqual(navigateNodes[0].id, 'live-action:navigate');
  assert.strictEqual(navigateEdges.length, 1);
  assert.ok(!payload.nodes.some(function (node) { return /Navigate \d+/.test(node.label); }));
});

test('live knowledge requests wide spacing for the connected semantic graph', function () {
  const LiveKnowledgeGraph = loadLiveKnowledgeGraph();
  const payload = new LiveKnowledgeGraph(runningPlan(), bridgeInformation()).build();

  assert.strictEqual(payload.spacing, 'wide');
});

test('live knowledge highlights only the current action chain and its entities', function () {
  const LiveKnowledgeGraph = loadLiveKnowledgeGraph();
  const payload = new LiveKnowledgeGraph(runningPlan(), bridgeInformation()).build();

  assert.deepStrictEqual(
    new Set(payload.highlight),
    new Set([
      'live-plan',
      'live-robot',
      'live-action:pickup',
      'live-action:reach',
      'live-arm:right',
      'live-object:Static_MilkPitcher.stl',
    ]),
  );
  assert.ok(!payload.highlight.includes('live-action:navigate'));
  assert.ok(!payload.highlight.includes('live-action:park-arms'));
});

test('completed live knowledge clears the active spotlight', function () {
  const LiveKnowledgeGraph = loadLiveKnowledgeGraph();
  const completedPlan = runningPlan();
  completedPlan.nodes[0].status = 'SUCCEEDED';

  const payload = new LiveKnowledgeGraph(completedPlan, bridgeInformation()).build();

  assert.deepStrictEqual(payload.highlight, []);
});
