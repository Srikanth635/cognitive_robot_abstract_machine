"""Context/provenance fixtures are synthetic and contain no world instances."""
import json
from hashlib import sha256
from rdflib import Graph,Namespace,RDF,Literal,XSD
from pyshacl import validate
from ontology_imports import ROOT,closure
from context_review import K,A,inspect,verify_bytes,instant
B='http://aicor.knowledge/ontologies/';E=Namespace(B+'context-review-examples.owl#')
g=Graph();g.bind('kc',K);g.bind('ac',A);g.bind('ex',E)
def add(s,p,o):g.add((s,p,o))
def text(v):return Literal(v,datatype=XSD.string)
def dt(v):return Literal('2026-11-11T'+v+'Z',datatype=XSD.dateTime)
add(E.record,A.recordVersion,text('1'));add(E.action,A.hasActionKind,E.GenericAction);add(E.GenericAction,RDF.type,A.ActionKind);add(E.targetAmount,RDF.type,A.ParameterSpecification);add(E.targetAmount,A.parameterKey,text('target-amount'));add(E.action,A.hasParameter,E.targetAmount);add(E.action,RDF.type,A.ActionDescription);add(E.record,RDF.type,A.KnowledgeRecord);add(E.record,A.expressesDescription,E.action)
for ctx in [E.contextA,E.contextB]:
 add(ctx,RDF.type,K.KnowledgeContext);add(ctx,K.contextKey,text(str(ctx).split('#')[-1]));add(ctx,K.scopedDescription,E.action)
for name,value,ctx,status,observed in [('first','0.3',E.contextA,A.Supported,'10:00:00'),('alternative','0.4',E.contextA,A.Proposed,'09:00:00'),('otherInstance','0.8',E.contextB,A.Supported,'10:00:00')]:
 claim=E[name+'Claim'];stmt=E[name+'Statement'];assessment=E[name+'Assessment'];evidence=E[name+'Evidence']
 add(E.record,A.hasClaim,claim);add(claim,RDF.type,A.KnowledgeClaim);add(claim,A.hasStatement,stmt);add(claim,A.hasStatus,A.Proposed);add(claim,A.aboutComponent,E.targetAmount)
 add(stmt,RDF.type,A.Statement);add(stmt,A.statementSubject,E.targetAmount);add(stmt,A.statementPredicateIRI,Literal(str(A.numericValue),datatype=XSD.anyURI));add(stmt,A.statementLiteral,Literal(value,datatype=XSD.decimal))
 for p,o in [(RDF.type,K.ClaimAssessment),(K.inRecord,E.record),(K.assessesClaim,claim),(K.inContext,ctx),(K.assessedStatus,status),(K.assessedAt,dt('10:00:00')),(K.validFrom,dt('10:00:00')),(K.validUntil,dt('11:00:00')),(K.usesEvidence,evidence)]:add(assessment,p,o)
 for p,o in [(RDF.type,K.ObservationEvidence),(K.observedAt,dt(observed)),(A.evidenceKind,A.CuratedExample),(A.evidenceText,text('Synthetic observation fixture; no sensor acquisition.')),(K.sourceSnapshot,E.source)]:add(evidence,p,o)
for p,o in [(RDF.type,K.SourceSnapshot),(K.sourceIRI,Literal('urn:test:synthetic-source',datatype=XSD.anyURI)),(K.sourceVersion,text('fixture-v1')),(K.selector,text('example rows 1-3')),(K.exactExcerpt,text('Synthetic targets: 0.3, 0.4, 0.8'))]:add(E.source,p,o)
payload=b'synthetic immutable record payload v1\n'
for p,o in [(RDF.type,K.RecordSnapshot),(K.snapshotOf,E.record),(K.sha256,text(sha256(payload).hexdigest())),(K.serializationProfile,text('test exact UTF-8 bytes v1'))]:add(E.snapshot,p,o)
g.serialize(ROOT/'review/fixtures/context-provenance.ttl',format='turtle')
schema,_=closure(B+'knowledge-context.owl');schema-=Graph().parse(ROOT/'DUL.owl');shapes=Graph().parse(ROOT/'core/knowledge-context-shapes.ttl')+Graph().parse(ROOT/'core/action-core-shapes.ttl')
def valid(data):return bool(validate(data,shacl_graph=shapes,ont_graph=schema,inference='rdfs',advanced=True)[0])
tests=[]
def check(n,ok):tests.append({'case':n,'passed':bool(ok)})
check('valid-contextual-fixture',valid(g))
a=inspect(g,E.record,E.contextA,'2026-11-11T10:00:30Z',60);b=inspect(g,E.record,E.contextB,'2026-11-11T10:00:30Z',60)
check('two-source-assessments-in-context-A',len(a['assessments'])==2)
check('context-B-isolated',len(b['assessments'])==1 and b['assessments'][0]['value'].startswith('"0.8"'))
check('alternative-values-visible',len(a['potentialAlternativeValues'])==1)
check('no-conflict-in-other-context',not b['potentialAlternativeValues'])
check('fresh-observation-reported',a['assessments'][1]['evidenceFreshness']==['fresh'])
check('stale-observation-reported',a['assessments'][0]['evidenceFreshness']==['stale'])
check('no-automatic-acceptance',all(not x['accepted'] for x in a['assessments']))
check('base-claim-status-preserved',all(g.value(c,A.hasStatus)==A.Proposed for c in g.objects(E.record,A.hasClaim)))
check('value-not-materialized',not list(g.objects(E.targetAmount,A.numericValue)))
check('end-exclusive',not inspect(g,E.record,E.contextA,'2026-11-11T11:00:00Z',60)['assessments'])
check('not-yet-valid',not inspect(g,E.record,E.contextA,'2026-11-11T09:59:59Z',60)['assessments'])
check('digest-valid',verify_bytes(payload,g.value(E.snapshot,K.sha256)))
check('digest-detects-mutated-bytes',not verify_bytes(payload+b'edited',g.value(E.snapshot,K.sha256)))
for name,s,p,o in [('missing-record',E.firstAssessment,K.inRecord,None),('missing-context',E.firstAssessment,K.inContext,None),('missing-source-version',E.source,K.sourceVersion,None),('missing-selector',E.source,K.selector,None),('missing-observed-time',E.firstEvidence,K.observedAt,None),('supported-without-evidence',E.firstAssessment,K.usesEvidence,None)]:
 h=Graph()+g;h.remove((s,p,o));check('reject-'+name,not valid(h))
for name,s,p,o in [('foreign-claim',E.firstAssessment,K.assessesClaim,E.foreignClaim),('wrong-description',E.contextA,K.scopedDescription,E.otherAction),('reversed-interval',E.firstAssessment,K.validUntil,dt('09:00:00')),('bad-digest',E.snapshot,K.sha256,text('abc'))]:
 h=Graph()+g;h.set((s,p,o));check('reject-'+name,not valid(h))
try:instant('2026-11-11T10:00:00');check('reject-naive-time',False)
except ValueError:check('reject-naive-time',True)
h=Graph()+g;h.set((E.firstEvidence,K.observedAt,dt('10:01:00')))
check('future-observation-not-fresh',inspect(h,E.record,E.contextA,'2026-11-11T10:00:30Z',60)['assessments'][1]['evidenceFreshness']==['future'])
(ROOT/'reports/context-validation.json').write_text(json.dumps({'release':'0.11.0','checks':tests,'answers':{'contextA':a,'contextB':b},'limitation':'Potential alternatives are not contradictions unless the predicate is single valued in the domain; no automatic truth acceptance.'},indent=2)+'\n')
print(json.dumps({'checks':len(tests),'passed':sum(x['passed'] for x in tests),'failed':[x for x in tests if not x['passed']]}));assert all(x['passed'] for x in tests)
