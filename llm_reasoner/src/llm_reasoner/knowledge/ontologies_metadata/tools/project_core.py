"""Demonstrate two read-only projections of the same record, retaining claim provenance.

This is a semantic view demo, not an execution adapter or a Minsky frame implementation.
"""
from pathlib import Path
import json
from rdflib import Graph, Namespace, RDF
R=Path(__file__).resolve().parents[1]; A=Namespace('http://aicor.knowledge/ontologies/action-description-core.owl#'); E=Namespace('http://aicor.knowledge/ontologies/action-examples.owl#')
g=Graph().parse(R/'examples/stereotypical-actions.ttl')
views=Graph();views.bind('ac',A);views.bind('ex',E)
def project(record,component_type,key):
 output={'record':str(record),'version':str(g.value(record,A.recordVersion)),'projection':key,'claims':[],'missing':[]}
 artifact=E['pour-'+key];views.add((artifact,RDF.type,A.ProjectionArtifact));views.add((artifact,A.derivedFromRecord,record));views.add((artifact,A.conformsToSpecification,E.goalView if key=='goal-view' else E.phaseView))
 for claim in sorted(g.objects(record,A.hasClaim)):
  if not any((c,RDF.type,component_type) in g for c in g.objects(claim,A.aboutComponent)):continue
  statement=g.value(claim,A.hasStatement)
  output['claims'].append({'id':str(claim),'status':str(g.value(claim,A.hasStatus)),'subject':str(g.value(statement,A.statementSubject)), 'predicate':str(g.value(statement,A.statementPredicateIRI)), 'object':str(g.value(statement,A.statementObject) or g.value(statement,A.statementLiteral)), 'evidence':[str(e) for e in g.objects(claim,A.hasEvidence)]})
  views.add((artifact,A.includesClaim,claim))
 if not output['claims']:output['missing'].append('No matching claims in source record.')
 return output
out=[project(E.pourRecordV1,A.GoalSpecification,'goal-view'),project(E.pourRecordV1,A.PhaseSpecification,'phase-view')]
(R/'examples/projection-results.json').write_text(json.dumps(out,indent=2)+'\n'); views.serialize(R/'examples/projection-artifacts.ttl',format='turtle')
print('Wrote two projections from pourRecordV1, preserving claim status and evidence references.')
