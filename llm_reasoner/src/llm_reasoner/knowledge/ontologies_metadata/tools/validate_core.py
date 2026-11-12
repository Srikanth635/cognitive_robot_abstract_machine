"""Validate the core and synthetic examples; exercise representative failure cases."""
from pathlib import Path
import json
from rdflib import Graph, Namespace, RDF, Literal, XSD, OWL
from pyshacl import validate
R=Path(__file__).resolve().parents[1]; A=Namespace('http://aicor.knowledge/ontologies/action-description-core.owl#'); E=Namespace('http://aicor.knowledge/ontologies/action-examples.owl#')
# SHACL uses the local core hierarchy; DUL logical compatibility is checked by reason_core.py.
schema=Graph().parse(R/'core/action-description-core.ttl')
shapes=Graph().parse(R/'core/action-core-shapes.ttl')
data=Graph().parse(R/'examples/stereotypical-actions.ttl')+Graph().parse(R/'examples/projection-artifacts.ttl')
def check(g):
 return validate(g,shacl_graph=shapes,ont_graph=schema,inference='rdfs',advanced=True)
results=[]
ok,_,txt=check(data);results.append({'case':'valid-two-action-examples','passed':bool(ok)})
(R/'reports/core-shacl.txt').write_text(txt)
mutations={
 'foreign-annotation-claim':lambda g:g.add((E.pourAnnotation,A.contributesClaim,E.graspClaim)),
 'missing-status':lambda g:g.remove((E.pourGoalClaim,A.hasStatus,None)),
 'supported-without-evidence':lambda g:(g.set((E.pourGoalClaim,A.hasStatus,A.Supported)),g.remove((E.pourGoalClaim,A.hasEvidence,None))),
 'ambiguous-statement-object':lambda g:g.add((E.pourGoalStatement,A.statementLiteral,Literal('duplicate object'))),
 'numeric-parameter-without-unit':lambda g:g.add((E.targetAmount,A.numericValue,Literal('0.3',datatype=XSD.decimal))),
 'reversed-parameter-bounds':lambda g:(g.add((E.targetAmount,A.lowerBound,Literal('2',datatype=XSD.decimal))),g.add((E.targetAmount,A.upperBound,Literal('1',datatype=XSD.decimal))),g.add((E.targetAmount,A.unitIRI,Literal('urn:unit:litre',datatype=XSD.anyURI)))),
 'unscoped-binding':lambda g:(g.add((E.binding,RDF.type,A.BindingRecord)),g.add((E.binding,A.bindsSpecification,E.sourceContainer)),g.add((E.binding,A.bindingTargetIRI,Literal('urn:world:object',datatype=XSD.anyURI)))),
 'foreign-projection-claim':lambda g:(g.add((E.badView,RDF.type,A.ProjectionArtifact)),g.add((E.badView,A.derivedFromRecord,E.pourRecordV1)),g.add((E.badView,A.conformsToSpecification,E.goalView)),g.add((E.badView,A.includesClaim,E.graspClaim))),
}
for name,mutate in mutations.items():
 g=Graph()+data;mutate(g);valid,_,_=check(g);results.append({'case':name,'passed':not bool(valid)})
# A reified proposal must not assert its corresponding predicate in the graph.
results.append({'case':'proposal-not-materialized','passed':(E.transferGoal,E.recipientSpecification,E.receivingContainer) not in data})
results.append({'case':'core-imports-only-DUL','passed':set(Graph().parse(R/'core/action-description-core.ttl').objects(None,OWL.imports))=={__import__('rdflib').URIRef('http://www.ontologydesignpatterns.org/ont/dul/DUL.owl')}})
(R/'reports/core-validation.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2)); assert all(r['passed'] for r in results)
