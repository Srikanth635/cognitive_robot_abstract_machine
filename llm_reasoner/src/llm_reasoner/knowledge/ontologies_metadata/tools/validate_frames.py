from pathlib import Path
import json
from rdflib import Graph,Namespace,RDF,Literal,XSD,BNode
from pyshacl import validate
from ontology_imports import ROOT,closure
from frame_runtime import A,F,effective_slots,project_frame,artifact_graph
E=Namespace('http://aicor.knowledge/ontologies/action-examples.owl#');data=Graph();shapes=Graph();schema,_=closure('http://aicor.knowledge/ontologies/all-action-aspects.owl');schema-=Graph().parse(ROOT/'DUL.owl')
for n in ['stereotypical-actions','aspect-actions','phase-actions','causal-physical-actions','control-actions','strategy-failure-actions','frame-contracts']:data+=Graph().parse(ROOT/'examples'/(n+'.ttl'))
for p in [ROOT/'core/action-core-shapes.ttl',ROOT/'aspects/validation/frame-shapes.ttl']:shapes+=Graph().parse(p)
results=[]
def record(n,ok):results.append({'case':n,'passed':bool(ok)})
outputs=[];artifacts=Graph();before=set(data)
for source,contract in [(E.pourRecordV6,E.PouringFrameContract),(E.graspRecordV6,E.GraspingFrameContract)]:
 output=project_frame(data,source,contract);outputs.append(output);artifacts+=artifact_graph(output)
ok,_,txt=validate(data+artifacts,shacl_graph=shapes,ont_graph=schema,inference='rdfs',advanced=True);record('valid-contracts-and-artifacts',ok);(ROOT/'reports/frame-shacl.txt').write_text(txt)
record('source-graph-unchanged',set(data)==before)
def slot(out,key):return next(s for s in out['slots'] if s['key']==key)
record('inherited-goals-reuse-source-iris',slot(outputs[0],'goals')['values'][0]['term']['value']==str(E.pourAmountGoal))
record('pouring-suppresses-inherited-default',not slot(outputs[0],'exploration-policy')['values'])
record('grasping-retains-marked-default',slot(outputs[1],'exploration-policy')['values'][0]['assessment']=='default-unverified')
record('unsupported-roles-marked-unassessed',all(v['assessment']=='unassessed-template-content' for v in slot(outputs[0],'participants')['values']))
record('proposed-claims-retained',any(v['support'] and v['support'][0]['status']==str(A.Proposed) for v in slot(outputs[0],'goals')['values']))
record('claims-belong-to-source-record',all((__import__('rdflib').URIRef(o['sourceRecord']),A.hasClaim,__import__('rdflib').URIRef(c)) in data for o in outputs for c in o['includedClaims']))
mutations={
'cyclic-inheritance':lambda g:g.add((E.GenericActionFrame,F.parentFrame,E.PouringFrame)),
'multiple-parents':lambda g:g.add((E.PouringFrame,F.parentFrame,E.GraspingFrame)),
'duplicate-slot-key':lambda g:g.set((E.roleSlot,F.slotKey,Literal('goals'))),
'foreign-override':lambda g:g.set((E.pourDefaultException,F.overridesSlot,E.notInherited)),
'contradictory-override':lambda g:g.add((E.pourDefaultException,F.replacementLiteral,Literal('other'))),
'noncontiguous-path':lambda g:g.set((E.goalPath,F.position,Literal(2,datatype=XSD.positiveInteger))),
'reversed-cardinality':lambda g:g.set((E.goalSlot,F.maximumCount,Literal(0,datatype=XSD.nonNegativeInteger))),
'relative-predicate':lambda g:g.set((E.goalPath,F.predicateIRI,Literal('hasGoal',datatype=XSD.anyURI))),
'multiple-defaults':lambda g:g.add((E.diagnosticPolicySlot,F.defaultEntity,E.pourAmountGoal)),
'unstable-anonymous-filler':lambda g:g.set((E.pourActionV6,A.hasGoal,BNode())),
'strict-without-status-policy':lambda g:g.set((E.PouringFrameContract,F.evidencePolicy,Literal('require-allowed-claim'))),
}
for n,mutate in mutations.items():
 g=Graph()+data;mutate(g)
 try:project_frame(g,E.pourRecordV6,E.PouringFrameContract);passed=False
 except ValueError:passed=True
 record('reject-'+n,passed)
strict=Graph()+data;strict.set((E.PouringFrameContract,F.evidencePolicy,Literal('require-allowed-claim')));strict.add((E.PouringFrameContract,F.allowedStatus,A.Proposed))
out=project_frame(strict,E.pourRecordV6,E.PouringFrameContract)
record('strict-retains-proposed-when-explicitly-allowed',bool(slot(out,'goals')['values']))
record('strict-excludes-unassessed-content',not slot(out,'participants')['values'])
strict.set((E.pourRecordV2Claim1,A.hasStatus,A.Rejected));out=project_frame(strict,E.pourRecordV6,E.PouringFrameContract);record('rejected-claim-vetoes-strict-value',not slot(out,'goals')['values'])
missing=Graph()+data;missing.remove((E.pourActionV6,A.hasGoal,None));out=project_frame(missing,E.pourRecordV6,E.PouringFrameContract);record('missing-required-slot-reported',any(i['issue']=='minimum-cardinality' for i in out['issues']))
missing.set((E.PouringFrameContract,A.missingInformationPolicy,Literal('retain-unknown')));out=project_frame(missing,E.pourRecordV6,E.PouringFrameContract);record('unknown-retained-without-invented-value',slot(out,'goals')['unknown'] and not slot(out,'goals')['values'])
missing.set((E.PouringFrameContract,A.missingInformationPolicy,Literal('reject')))
try:project_frame(missing,E.pourRecordV6,E.PouringFrameContract);passed=False
except ValueError:passed=True
record('reject-policy-enforced',passed)
# A second path step traverses role specifications to shared entity specifications.
multi=Graph()+data;multi.add((E.roleMapping,F.hasPathStep,E.entityPath));multi.add((E.entityPath,F.position,Literal(2,datatype=XSD.positiveInteger)));multi.add((E.entityPath,F.predicateIRI,Literal(str(A.hasEntitySpecification),datatype=XSD.anyURI)))
out=project_frame(multi,E.pourRecordV6,E.PouringFrameContract);record('multistep-path-follows-shared-entities',{v['term']['value'] for v in slot(out,'participants')['values']}=={str(E.sourceContainer),str(E.receivingContainer),str(E.transferableSubstance)})
replacement=Graph()+data;replacement.remove((E.pourDefaultException,F.suppressDefault,None));replacement.add((E.pourDefaultException,F.replacementLiteral,Literal('inspect-evidence')))
record('explicit-default-replacement',slot(project_frame(replacement,E.pourRecordV6,E.PouringFrameContract),'exploration-policy')['values'][0]['term']['value']=='inspect-evidence')
strictDefault=Graph()+data;strictDefault.set((E.GraspingFrameContract,F.evidencePolicy,Literal('require-allowed-claim')));strictDefault.add((E.GraspingFrameContract,F.allowedStatus,A.Proposed))
record('strict-excludes-unverified-default',not slot(project_frame(strictDefault,E.graspRecordV6,E.GraspingFrameContract),'exploration-policy')['values'])
overflow=Graph()+data;overflow.add((E.pourActionV6,A.hasActionKind,E.Grasping))
record('maximum-cardinality-reported',any(i['issue']=='maximum-cardinality' for i in project_frame(overflow,E.pourRecordV6,E.PouringFrameContract)['issues']))
record('output-does-not-assess-execution',all(not o['executionReadinessAssessed'] for o in outputs))
(ROOT/'examples/frame-projections.json').write_text(json.dumps(outputs,indent=2)+'\n');artifacts.serialize(ROOT/'examples/frame-projection-artifacts.ttl',format='turtle')
(ROOT/'reports/frame-validation.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps({'checks':len(results),'passed':sum(r['passed'] for r in results),'failures':[r for r in results if not r['passed']]},indent=2));assert all(r['passed'] for r in results)
