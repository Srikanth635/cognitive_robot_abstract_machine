"""0.22 regression checks for pouring, pickup, placement and navigation.

Mutations and numbers are synthetic in-memory test inputs. They neither populate
the knowledge store nor establish physical truth, observations or executability.
"""
import hashlib
import json
from pathlib import Path
from rdflib import Graph, Namespace, RDF, RDFS, OWL, BNode, Literal, XSD
from pyshacl import validate
from ontology_imports import ROOT, closure
from aspect_runtime import A, C, P, one
from condition_signatures import signature
from object_manipulation import N as OM, manipulation_errors, assess as manipulation_assess
from mobility_navigation import N as MN, mobility_errors, assess as navigation_assess
from tool_material_transfer import MT, TC, material_errors, assess as transfer_assess
from context_review import K

B = 'http://aicor.knowledge/ontologies/'
E = Namespace(B + 'primary-action-tests.owl#')
schema, imports = closure(B + 'all-action-knowledge.owl')
schema -= Graph().parse(ROOT / 'DUL.owl')
data = Graph()
for file in ['object-manipulation-profiles.ttl', 'mobility-navigation-profiles.ttl', 'tool-material-transfer-profiles.ttl']:
    data.parse(ROOT / 'examples' / file)
metadata = set(data.subjects(RDF.type, A.KnowledgeClaim)) | set(data.subjects(RDF.type, A.Statement))
descriptive = Graph()
for s, p, o in data:
    if s not in metadata and p != A.hasClaim:
        descriptive.add((s, p, o))
g = schema + descriptive
ME = Namespace(B + 'object-manipulation-examples.owl#')
NE = Namespace(B + 'mobility-examples.owl#')
TE = Namespace(B + 'tool-transfer-examples.owl#')
profiles = [
    ('pickup', ME.pickup, manipulation_errors, manipulation_assess, OM.Holds),
    ('place', ME.placement, manipulation_errors, manipulation_assess, OM.AtTarget),
    ('navigate', NE.navigation, mobility_errors, navigation_assess, MN.InRegion),
    ('pour', TE.pouringTransfer, material_errors, transfer_assess, P.ContainedIn),
]
checks = []
def check(name, result):
    checks.append({'case': name, 'passed': bool(result)})

def expression(graph, action, prop):
    return one(graph, one(graph, action, prop), C.hasExpression)

def reject(name, checker, mutation):
    h = Graph() + g
    mutation(h)
    try:
        rejected = bool(checker(h))
    except ValueError:
        rejected = True
    check(name, rejected)

def atom_for(graph, action, relation, prop=A.hasSuccessCriterion):
    seen, todo = set(), [expression(graph, action, prop)]
    while todo:
        node = todo.pop()
        if node in seen:
            continue
        seen.add(node)
        if (node, RDF.type, C.AtomicCondition) in graph and one(graph, one(graph, node, C.hasPattern), P.usesRelation) == relation:
            return node
        todo.extend(graph.objects(node, C.hasOperand))
    raise ValueError('Expected atom not found')

def contradictory(graph, action, relation):
    node = atom_for(graph, action, relation)
    neg, conjunction = E['neg' + str(action).split('#')[-1]], E['and' + str(action).split('#')[-1]]
    graph.add((neg, RDF.type, C.Negation)); graph.add((neg, C.hasOperand, node))
    graph.add((conjunction, RDF.type, C.Conjunction))
    graph.add((conjunction, C.hasOperand, expression(graph, action, A.hasSuccessCriterion)))
    graph.add((conjunction, C.hasOperand, neg))
    graph.set((one(graph, action, A.hasSuccessCriterion), C.hasExpression, conjunction))

def foreign_binding(graph, action, relation, prop):
    node = atom_for(graph, action, relation, prop)
    pattern = one(graph, node, C.hasPattern)
    binding = sorted(graph.objects(pattern, P.hasArgumentBinding), key=str)[0]
    graph.set((binding, P.bindsEntity, E.foreignEntity))

def supplied(action, prop):
    atoms, pairs = signature(g, expression(g, action, prop))
    facts = {(scope, rel, args): not negative for scope, rel, args, negative in atoms}
    quantities = {}
    for op, scope, left, rscope, right in pairs:
        lv, rv = ('2', '1') if op == C.GreaterOrEqual else ('0.01', '0.02')
        quantities[(scope, left)] = (lv, str(one(g, left, A.unitIRI)))
        quantities[(rscope, right)] = (rv, str(one(g, right, A.unitIRI)))
    return facts, quantities

for name, action, checker, assess, relation in profiles:
    check(name + ':existing-profiles-valid', not checker(g))
    reject(name + ':success-cannot-substitute-for-prerequisite', checker,
           lambda h, a=action: h.set((one(h, a, A.hasPrecondition), C.hasExpression, expression(h, a, A.hasSuccessCriterion))))
    reject(name + ':success-must-establish-declared-goal', checker,
           lambda h, a=action: h.set((one(h, a, A.hasGoal), C.hasExpression, expression(h, a, A.hasPrecondition))))
    reject(name + ':success-is-not-fault', checker,
           lambda h, a=action: h.set((one(h, a, A.hasFailureCriterion), C.hasExpression, expression(h, a, A.hasSuccessCriterion))))
    reject(name + ':contradictory-outcome-rejected', checker,
           lambda h, a=action, r=relation: contradictory(h, a, r))
    reject(name + ':goal-cannot-change-object-or-destination', checker,
           lambda h, a=action: h.set((one(h, a, A.hasGoal), C.hasExpression, expression(h, NE.facing, A.hasGoal))))
    success_facts, quantities = supplied(action, A.hasSuccessCriterion)
    pre_facts, _ = supplied(action, A.hasPrecondition)
    fault_facts, _ = supplied(action, A.hasFailureCriterion)
    def call(facts):
        return assess(g, action, facts=facts) if name in ['pickup', 'place'] else assess(g, action, facts=facts, quantities=quantities)
    out = call(success_facts)
    check(name + ':outcome-alone-leaves-combined-criteria-unknown', out['success'] is True and out['criteriaSatisfied'] is None)
    complete = {**success_facts, **pre_facts, **{key: False for key in fault_facts}}
    check(name + ':qualified-supplied-criteria-satisfied', call(complete)['criteriaSatisfied'] is True)
    incomplete = dict(complete)
    for key in fault_facts:
        incomplete.pop(key)
    check(name + ':missing-fault-evidence-is-not-no-fault', call(incomplete)['criteriaSatisfied'] is None)
    bad = dict(complete); bad[next(iter(pre_facts))] = False
    check(name + ':false-prerequisite-blocks-combined-criteria', call(bad)['success'] is True and call(bad)['criteriaSatisfied'] is False)
    bad = dict(complete); bad.update(fault_facts)
    check(name + ':success-and-fault-remain-visible', call(bad)['successAndFaultReported'] and call(bad)['criteriaSatisfied'] is False)
    check(name + ':no-observation-acceptance-or-execution', call(complete)['acceptedObservations'] == 0 and not call(complete)['worldUpdated'] and not call(complete)['executed'])
    h = Graph() + g
    h.set((one(h, action, A.hasGoal), C.hasExpression, atom_for(h, action, relation)))
    check(name + ':weaker-goal-permitted', not checker(h))

for action, relation, checker, label in [
    (ME.pickup, P.Supports, manipulation_errors, 'pickup-support'),
    (ME.placement, OM.Holds, manipulation_errors, 'place-held-object'),
    (NE.navigation, MN.NavigableTo, mobility_errors, 'navigation-footprint'),
    (TE.pouringTransfer, MT.TransferReady, material_errors, 'pour-source-destination-material')]:
    reject(label + ':foreign-prerequisite-binding', checker,
           lambda h, a=action, r=relation: foreign_binding(h, a, r, A.hasPrecondition))
reject('pickup:current-state-is-not-initial-support', manipulation_errors,
       lambda h: h.set((atom_for(h, ME.pickup, P.Supports, A.hasPrecondition), C.inScope, C.CurrentState)))
reject('place:current-hold-is-not-initial-hold', manipulation_errors,
       lambda h: h.set((atom_for(h, ME.placement, OM.Holds, A.hasPrecondition), C.inScope, C.CurrentState)))
reject('pickup:optional-context-cannot-belong-to-navigation', manipulation_errors,
       lambda h: h.add((ME.pickup, OM.evaluationContext, NE.navigationContext)))
reject('pour:mixed-technique-profile-needs-explicit-contract', material_errors,
       lambda h: h.add((TE.pouringTransfer, RDF.type, MT.CompleteTransferDescription)))
reject('navigation:roles-cannot-disagree-with-target', mobility_errors,
       lambda h: h.remove((NE.navigation, A.hasRoleSpecification, NE.navigationTargetRoleSpecification)))

# Complementary comparisons within one scope are inconsistent; different state
# scopes and weaker goals are intentionally not rejected as contradictions.
h = Graph() + g
root = expression(h, TE.pouringTransfer, A.hasSuccessCriterion)
comparison = next(n for n in h.objects(root, C.hasOperand) if (n, RDF.type, C.Comparison) in h and one(h, n, C.operator) == C.LessOrEqual)
for p, o in list(h.predicate_objects(comparison)):
    h.add((E.complementaryComparison, p, o))
h.set((E.complementaryComparison, C.operator, C.GreaterThan)); h.add((root, C.hasOperand, E.complementaryComparison))
try:
    material_errors(h)
except ValueError:
    check('pour:complementary-spill-comparisons-rejected', True)
else:
    check('pour:complementary-spill-comparisons-rejected', False)
h = Graph() + g
old = expression(h, ME.placement, A.hasSuccessCriterion)
h.add((E.scopedConjunction, RDF.type, C.Conjunction)); h.add((E.scopedConjunction, C.hasOperand, old))
h.add((E.scopedConjunction, C.hasOperand, expression(h, ME.placement, A.hasPrecondition)))
h.set((one(h, ME.placement, A.hasSuccessCriterion), C.hasExpression, E.scopedConjunction))
check('place:held-before-and-released-after-are-consistent', not manipulation_errors(h))

demo = Graph().parse(ROOT / 'demo/comprehensive-pouring.ttl')
check('pour:existing-concrete-readiness-and-spill-comparison-remain-valid', not material_errors(g + demo))

# Check the newly added shape constraints separately from Python semantics.
all_shapes = Graph()
for file in ['object-manipulation-shapes.ttl', 'mobility-navigation-shapes.ttl', 'tool-transfer-shapes.ttl']:
    all_shapes.parse(ROOT / 'domains' / file)
SH = Namespace('http://www.w3.org/ns/shacl#')
shapes = Graph(); seen = set()
todo = [s for s in all_shapes.subjects(RDF.type, SH.NodeShape) if 'Primary' in str(s)]
while todo:
    n = todo.pop()
    if n in seen:
        continue
    seen.add(n)
    for p, o in all_shapes.predicate_objects(n):
        shapes.add((n, p, o))
        if isinstance(o, BNode):
            todo.append(o)
def shape_valid(graph, focus=None):
    ok, _, _ = validate(graph, shacl_graph=shapes, ont_graph=schema, inference='rdfs', advanced=True,
                        focus_nodes=focus)
    return bool(ok)
check('primary-shapes:existing-profiles-conform', shape_valid(descriptive))
for name, subject, predicate, value, focus in [
    ('pickup-self-support', ME.pickup, OM.initialSupport, ME.pickupObject, ME.pickup),
    ('place-self-support', ME.placement, OM.destinationSupport, ME.placementObject, ME.placement),
    ('navigation-self-target', NE.navigation, MN.targetRegion, NE.navigationRobot, NE.navigation),
    ('pour-identical-endpoints', TE.pouringTransferContract, TC.destination, TE.pouringTransferSource, TE.pouringTransferContract),
    ('pour-unsupported-basis', TE.pouringTransfer, MT.quantityBasis, Literal('count', datatype=XSD.string), TE.pouringTransfer),
    ('pickup-wrong-context', ME.pickup, OM.evaluationContext, NE.navigationContext, ME.pickup),
    ('navigate-wrong-context', NE.navigationContext, K.scopedDescription, ME.pickup, NE.navigation),
    ('pour-wrong-context', TE.pouringTransferContext, K.scopedDescription, NE.navigation, TE.pouringTransfer)]:
    h = Graph() + descriptive; h.set((subject, predicate, value))
    check('primary-shapes:reject-' + name, not shape_valid(h, [focus]))

for cls, prop in [(OM.ObjectManipulationDescription, OM.object), (OM.PickupDescription, OM.initialSupport),
                  (OM.PlacementDescription, OM.destinationSupport), (MT.MaterialTransferDescription, TC.hasTransferContract),
                  (MN.MobilityDescription, MN.targetRegion)]:
    check('OWL:single-binding:' + str(prop), any((n, OWL.onProperty, prop) in schema and
          int(schema.value(n, OWL.maxCardinality) or 0) == 1 for n in schema.objects(cls, RDFS.subClassOf)))
check('no-new-ontology-modules', len(json.loads((ROOT / 'ontology-registry.json').read_text())) == 40)
check('no-framework-dependency', B + 'cram-action-bridge.owl' not in imports and 'urn:gpt-cram-ontologies:aicor-unified' not in imports)
report = {'release': '0.22.0', 'checks': checks, 'passed': sum(c['passed'] for c in checks),
          'scope': 'Four existing action families; bounded declarative constraints and synthetic input tests. No new module, world grounding, accepted observation, physical solver or execution.',
          'input_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted((ROOT / 'domains').glob('*.ttl'))}}
(ROOT / 'reports/primary-action-validation.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'checks': len(checks), 'passed': report['passed'], 'failures': [c for c in checks if not c['passed']]}))
assert all(c['passed'] for c in checks)
