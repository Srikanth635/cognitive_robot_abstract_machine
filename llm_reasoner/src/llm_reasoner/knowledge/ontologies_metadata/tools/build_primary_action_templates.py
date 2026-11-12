"""Extract symbolic generation aids from existing four-action examples.

No scalar defaults, world objects, new ontology modules, transport or LLM calls.
The output is a structural aid rather than a complete output validation schema.
"""
import hashlib
import json
from rdflib import Graph, Namespace, RDF
from ontology_imports import ROOT, REGISTRY, closure
from aspect_runtime import A, C, one
from condition_signatures import signature
from object_manipulation import N as OM
from mobility_navigation import N as MN
from tool_material_transfer import MT, TC

B = 'http://aicor.knowledge/ontologies/'
schema, imports = closure(B + 'all-action-knowledge.owl')
specs = [
    ('pickup', 'object-manipulation-profiles.ttl', 'object-manipulation-examples', 'pickup',
     {'manipulator': OM.manipulator, 'object': OM.object, 'initial_support': OM.initialSupport}, {}),
    ('place', 'object-manipulation-profiles.ttl', 'object-manipulation-examples', 'placement',
     {'manipulator': OM.manipulator, 'object': OM.object, 'destination_support': OM.destinationSupport, 'target_region': OM.targetRegion}, {}),
    ('navigate', 'mobility-navigation-profiles.ttl', 'mobility-examples', 'navigation',
     {'robot': MN.robot, 'target_region': MN.targetRegion, 'footprint': MN.footprint, 'frame': MN.referenceFrame},
     {'position_error': MN.positionErrorParameter, 'position_tolerance': MN.positionToleranceParameter,
      'heading_error': MN.headingErrorParameter, 'heading_tolerance': MN.headingToleranceParameter,
      'clearance': MN.clearanceParameter}),
    ('pour', 'tool-material-transfer-profiles.ttl', 'tool-transfer-examples', 'pouringTransfer', {}, {}),
]
templates = []
for name, file, ns, local, entity_fields, parameter_fields in specs:
    source = Graph().parse(ROOT / 'examples' / file)
    g = schema + source
    E = Namespace(B + ns + '.owl#'); root = E[local]; record = E[local + 'Record']
    entities = {role: one(g, root, prop) for role, prop in entity_fields.items()}
    parameters = {role: one(g, root, prop) for role, prop in parameter_fields.items()}
    if name == 'pour':
        contract = one(g, root, TC.hasTransferContract)
        entities = {role: one(g, contract, prop) for role, prop in {
            'source': TC.source, 'destination': TC.destination, 'material': TC.substance,
            'transferred_portion': TC.transferredPortion}.items()}
        parameters = {role: one(g, contract, prop) for role, prop in {
            'delivered_amount': TC.deliveredAmount, 'target_amount': TC.targetAmount,
            'spill_amount': TC.spillAmount, 'spill_tolerance': TC.spillTolerance}.items()}
    symbols = {iri: 'entity:' + key for key, iri in entities.items()}
    symbols.update({iri: 'parameter:' + key for key, iri in parameters.items()})
    def symbol(term):
        if term not in symbols:
            raise ValueError('Unmapped fixture-specific argument: ' + str(term))
        return symbols[term]
    conditions = {}
    for key, prop in [('precondition', A.hasPrecondition), ('goal', A.hasGoal),
                      ('success', A.hasSuccessCriterion), ('fault', A.hasFailureCriterion)]:
        expression = one(g, one(g, root, prop), C.hasExpression)
        atoms, comparisons = signature(g, expression)
        conditions[key] = {'combination': 'all-clauses', 'root_expression_type': str(one(g, expression, RDF.type)),
            'atoms': sorted([{'scope': str(scope), 'relation': str(rel), 'arguments': [symbol(x) for x in args],
                             'negated': neg} for scope, rel, args, neg in atoms], key=lambda x: json.dumps(x, sort_keys=True)),
            'comparisons': sorted([{'operator': str(op), 'left': {'scope': str(scope), 'parameter': symbol(left)},
                                   'right': {'scope': str(rscope), 'parameter': symbol(right)}}
                                  for op, scope, left, rscope, right in comparisons], key=lambda x: json.dumps(x, sort_keys=True))}
    templates.append({'action': name, 'source_fixture': 'examples/' + file, 'source_record': str(record),
        'source_record_version': str(one(g, record, A.recordVersion)), 'source_description': str(root),
        'action_kind': str(one(g, root, A.hasActionKind)),
        'entity_roles': sorted(entities),
        'parameters': {role: {'unit_iri': str(one(g, iri, A.unitIRI)), 'scalar_default': None} for role, iri in parameters.items()},
        'conditions': conditions})
out = {'release': '0.22.0', 'artifact_kind': 'symbolic generation aid', 'complete_output_schema': False,
       'new_record_identity_policy': 'Use a fresh record and mutable annotation/context identities; reuse immutable descriptions/specifications only through an explicit versioned sharing policy.',
       'unknown_value_policy': 'Leave physical values unspecified; scalar_default null is template metadata, never an RDF numeric literal.',
       'epistemic_policy': 'Generated claims remain Proposed; no observations, world bindings or accepted beliefs are produced.',
       'templates': templates, 'schema_imports': imports,
       'ontology_sha256': {REGISTRY[iri]: hashlib.sha256((ROOT / REGISTRY[iri]).read_bytes()).hexdigest() for iri in imports},
       'input_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted({ROOT / 'examples' / s[1] for s in specs})}}
target = ROOT / 'generation/primary-action-templates.json'
target.parent.mkdir(exist_ok=True); target.write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps({'templates': len(templates), 'physical_scalar_defaults': 0}))
