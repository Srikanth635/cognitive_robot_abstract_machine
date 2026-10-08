// Unit tests for panels/graph/ontology-graph.js: the model deciding which classes and
// links the Ontology Graph tab shows at each level of detail, what a search matches,
// and what the panel says about a selected class.
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const WEB = path.join(__dirname, '..', '..', '..', 'cram_viz', 'src', 'cram_viz', 'web');

function loadOntologyGraph() {
  global.window = {};
  new Function(fs.readFileSync(path.join(WEB, 'panels/graph/ontology-graph.js'), 'utf8'))();
  return global.window.OntologyGraph;
}

// a used phase class specialising a microworld class linked to CRAM-AICOR, citing an
// introduced concept, related to a role class and disjoint from another phase class
function fixturePayload() {
  return {
    nodes: [
      { id: 'pre:LiftPhase', label: 'LiftPhase', category: 'used', count: 2 },
      { id: 'pre:TransportPhase', label: 'TransportPhase', category: 'used', count: 1 },
      { id: 'core:ActionPhase', label: 'ActionPhase', category: 'microworld', count: 0 },
      { id: 'aicor:Motion', label: 'aicor:Motion', category: 'linked', count: 0 },
      { id: 'hypothesis:Gentle', label: 'hypothesis:Gentle', category: 'introduced', count: 0 },
      { id: 'pour:SourceContainer', label: 'SourceContainer', category: 'used', count: 1 },
      { id: 'core:CoupledPhase', label: 'CoupledPhase', category: 'microworld', count: 0 },
    ],
    edges: [
      { source: 'pre:LiftPhase', target: 'core:ActionPhase', kind: 'is_a', label: '', count: 1 },
      { source: 'core:ActionPhase', target: 'aicor:Motion', kind: 'is_a', label: '', count: 1 },
      { source: 'pre:LiftPhase', target: 'hypothesis:Gentle', kind: 'cites', label: 'core:hasCategory', count: 1 },
      { source: 'pre:LiftPhase', target: 'pour:SourceContainer', kind: 'relation', label: 'core:involvesRole', count: 3 },
      { source: 'pre:LiftPhase', target: 'pre:TransportPhase', kind: 'disjoint', label: '', count: 1 },
      { source: 'pre:LiftPhase', target: 'core:CoupledPhase', kind: 'classified_as', label: '', count: 1 },
    ],
    details: {
      'pre:LiftPhase': {
        prefixed: 'prehension:LiftPhase',
        source: 'predictive manipulation',
        evidence: 'core:EmpiricalFinding',
        description: 'Raising the object <clear> of its support.',
        requirements: ['terminated_by has a LiftClearanceReached'],
        claims: [{ id: 'core:JF09-07', text: 'Grasp stability prevents slips.', evidence: 'core:SupportedInterpretation' }],
        instances: ['lift'],
        count: 2,
      },
    },
  };
}

function visibleIds(model) {
  return model.nodes.filter((node) => model.isNodeVisible(node)).map((node) => node.id);
}

// %% levels of detail
test('the graph opens on the classes the hypothesis uses and their backing', function () {
  const { OntologyGraphModel, INITIAL_LEVEL, LEVELS } = loadOntologyGraph();
  const model = new OntologyGraphModel(fixturePayload());
  assert.strictEqual(model.level, INITIAL_LEVEL);
  assert.deepStrictEqual(
    visibleIds(model),
    fixturePayload().nodes
      .filter((node) => !LEVELS[INITIAL_LEVEL].hidden.includes(node.category))
      .map((node) => node.id),
  );
});

test('the first level shows only the classes the hypothesis instantiates', function () {
  const { OntologyGraphModel } = loadOntologyGraph();
  const model = new OntologyGraphModel(fixturePayload());
  model.setLevel(0);
  assert.deepStrictEqual(visibleIds(model), ['pre:LiftPhase', 'pre:TransportPhase', 'pour:SourceContainer']);
});

test('the last level shows every class and concept', function () {
  const { OntologyGraphModel } = loadOntologyGraph();
  const model = new OntologyGraphModel(fixturePayload());
  model.setLevel(2);
  assert.strictEqual(visibleIds(model).length, fixturePayload().nodes.length);
});

// %% legend toggles
test('relations and disjointness start switched off and switch on from the legend', function () {
  const { OntologyGraphModel, HIDDEN_KINDS } = loadOntologyGraph();
  const model = new OntologyGraphModel(fixturePayload());
  const relation = model.edges.find((edge) => edge.kind === 'relation');
  assert.ok(HIDDEN_KINDS.includes('relation') && HIDDEN_KINDS.includes('disjoint'));
  assert.strictEqual(model.isEdgeVisible(relation), false);
  model.toggleKind('relation');
  assert.strictEqual(model.isEdgeVisible(relation), true);
});

test('a link to a hidden class is hidden with it', function () {
  const { OntologyGraphModel } = loadOntologyGraph();
  const model = new OntologyGraphModel(fixturePayload());
  const linked = model.edges.find((edge) => edge.target === 'aicor:Motion');
  assert.strictEqual(model.isEdgeVisible(linked), true);
  model.toggleCategory('linked');
  assert.strictEqual(model.isEdgeVisible(linked), false);
});

// %% search
test('search matches a class by name, prefixed name or description', function () {
  const { OntologyGraphModel } = loadOntologyGraph();
  const model = new OntologyGraphModel(fixturePayload());
  const lift = model.byId['pre:LiftPhase'];
  assert.ok(model.matches(lift, 'liftph'));
  assert.ok(model.matches(lift, 'PREHENSION:'));
  assert.ok(model.matches(lift, 'support'));
  assert.ok(!model.matches(lift, 'pouring'));
});

test('matches are listed with the classes the hypothesis uses first, hidden ones included', function () {
  const { OntologyGraphModel } = loadOntologyGraph();
  const model = new OntologyGraphModel(fixturePayload());
  model.setLevel(0);
  assert.deepStrictEqual(
    model.matching('phase').map((node) => node.id),
    ['pre:LiftPhase', 'pre:TransportPhase', 'core:ActionPhase', 'core:CoupledPhase'],
  );
});

test('an empty search matches nothing', function () {
  const { OntologyGraphModel } = loadOntologyGraph();
  assert.deepStrictEqual(new OntologyGraphModel(fixturePayload()).matching('  '), []);
});

// %% selection
test('selecting a class reaches every class it is linked to, whatever the legend hides', function () {
  const { OntologyGraphModel } = loadOntologyGraph();
  const model = new OntologyGraphModel(fixturePayload());
  assert.deepStrictEqual(
    [...model.connected('pre:LiftPhase')].sort(),
    ['core:ActionPhase', 'core:CoupledPhase', 'hypothesis:Gentle', 'pour:SourceContainer', 'pre:LiftPhase', 'pre:TransportPhase'],
  );
});

test('the links of a selected class are drawn even when their kind is switched off', function () {
  const { OntologyGraphModel } = loadOntologyGraph();
  const model = new OntologyGraphModel(fixturePayload());
  const relation = model.edges.find((edge) => edge.kind === 'relation');
  assert.strictEqual(model.isEdgeVisible(relation), false);
  assert.strictEqual(model.isEdgeVisible(relation, 'pre:LiftPhase'), true);
  assert.strictEqual(model.isEdgeVisible(relation, 'core:ActionPhase'), false);
});

// %% labels
test('a used class shows its instance count and a link its count', function () {
  const { OntologyGraphModel } = loadOntologyGraph();
  const model = new OntologyGraphModel(fixturePayload());
  assert.strictEqual(model.nodeLabel(model.byId['pre:LiftPhase']), 'LiftPhase ×2');
  assert.strictEqual(model.nodeLabel(model.byId['core:ActionPhase']), 'ActionPhase');
  const relation = model.edges.find((edge) => edge.kind === 'relation');
  assert.strictEqual(model.edgeLabel(relation), 'core:involvesRole ×3');
});

// %% class panel
test('the panel names the defined classes the instances meet, and how many do', function () {
  const { OntologyGraphModel } = loadOntologyGraph();
  const html = new OntologyGraphModel(fixturePayload()).panelHtml('pre:LiftPhase');
  assert.ok(html.includes('classified as'));
  assert.ok(html.includes('data-select="core:CoupledPhase"'));
  assert.ok(html.includes('1 of 2 instances'));
});

test('a defined class lists the classes whose instances it classifies', function () {
  const { OntologyGraphModel } = loadOntologyGraph();
  const html = new OntologyGraphModel(fixturePayload()).panelHtml('core:CoupledPhase');
  assert.ok(html.includes('classifies'));
  assert.ok(html.includes('data-select="pre:LiftPhase"'));
});

test('the panel says what the ontology states and how the hypothesis uses the class', function () {
  const { OntologyGraphModel } = loadOntologyGraph();
  const model = new OntologyGraphModel(fixturePayload());
  const html = model.panelHtml('pre:LiftPhase');
  const detail = fixturePayload().details['pre:LiftPhase'];
  assert.ok(html.includes(detail.prefixed));
  assert.ok(html.includes(detail.requirements[0]));
  assert.ok(html.includes(detail.claims[0].text));
  assert.ok(html.includes('data-select="core:ActionPhase"'));
  assert.ok(html.includes('core:involvesRole ×3'));
  assert.ok(html.includes('… and 1 more'));
});

test('the panel escapes text from the ontology', function () {
  const { OntologyGraphModel } = loadOntologyGraph();
  const html = new OntologyGraphModel(fixturePayload()).panelHtml('pre:LiftPhase');
  assert.ok(html.includes('&lt;clear&gt;'));
  assert.ok(!html.includes('<clear>'));
});

test('a link between classes that are not both in the graph is dropped', function () {
  const { OntologyGraphModel } = loadOntologyGraph();
  const payload = fixturePayload();
  payload.edges.push({ source: 'pre:LiftPhase', target: 'missing:Class', kind: 'is_a', label: '', count: 1 });
  assert.strictEqual(new OntologyGraphModel(payload).edges.length, fixturePayload().edges.length);
});
