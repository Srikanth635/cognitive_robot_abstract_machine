import json
from rdflib import Graph,Namespace,RDF,RDFS,URIRef,Literal,XSD
from pyshacl import validate
from ontology_imports import ROOT,closure
from aspect_runtime import A,C,P,one,structural_errors
from temporal_runtime import temporal_errors,assess_monitor
from perception_attention import N,perception_errors,inspect
from context_review import K
B='http://aicor.knowledge/ontologies/';E=Namespace(B+'perception-examples.owl#');T=Namespace(B+'perception-test-inputs.owl#')
schema,imports=closure(B+'perception-attention.owl');schema-=Graph().parse(ROOT/'DUL.owl');data=Graph().parse(ROOT/'examples/perception-attention-profiles.ttl');g=data+schema;shapes=Graph()
for p in [ROOT/'core/action-core-shapes.ttl',ROOT/'core/knowledge-context-shapes.ttl',ROOT/'aspects/validation/aspect-shapes.ttl',ROOT/'aspects/validation/temporal-shapes.ttl',ROOT/'domains/perception-attention-shapes.ttl']:shapes.parse(p)
def valid(graph):
 ok,_,text=validate(graph,shacl_graph=shapes,ont_graph=schema,inference='rdfs',advanced=True)
 if graph is data:(ROOT/'reports/perception-attention-shacl.txt').write_text(text)
 return bool(ok)
tests=[]
def check(n,ok):tests.append({'case':n,'passed':bool(ok)})
def rejects(n,fn):
 try:fn();check(n,False)
 except ValueError:check(n,True)
AT='2026-11-11T10:00:30Z';WHEN='2026-11-11T10:00:00Z'
def result(name,tag='one',matches=(),outcome=N.Completed,attention=None,visibility=None,view=None,when=WHEN,kind=A.CuratedExample):
 graph=Graph()+g;node=T[name+tag]
 triples=[(node,RDF.type,N.PerceptionResultRecord),(node,N.inRecord,E[name+'Record']),(node,N.inContext,E[name+'Context']),(node,N.answersQuery,E[name+'Query']),(node,N.resultFrame,E[name+'Frame']),(node,N.outcome,outcome),(node,K.observedAt,Literal(when,datatype=XSD.dateTime)),(node,A.evidenceKind,kind),(node,A.sourceIRI,Literal('urn:test:synthetic-perception-source',datatype=XSD.anyURI)),(node,A.evidenceText,Literal('Synthetic test input, not acquired sensor data.',datatype=XSD.string))]
 if attention is not None:triples.append((node,N.attentionState,attention))
 if visibility is not None:triples.append((node,N.visibilityState,visibility))
 if view is not None:triples.append((node,N.forView,view))
 for i,match in enumerate(matches):
  candidate=T[name+tag+'Candidate'+str(i)];entity=T[name+tag+'Entity'+str(i)];triples.extend([(node,N.hasCandidate,candidate),(candidate,RDF.type,N.CandidateHypothesis),(candidate,N.candidateEntity,entity),(candidate,N.matchAssessment,match),(entity,RDF.type,A.EntitySpecification)])
 for t in triples:graph.add(t)
 return graph,node
def assess(graph,name,ids):return inspect(graph,E[name+'Record'],ids,AT,60)
check('five-profile-shapes',valid(data));check('domain-contracts',not perception_errors(g));check('relation-signatures',not structural_errors(g));check('phase-monitor-integrity',not temporal_errors(g))
for name in ['attention','visibility','detection','identification','search']:
 check(name+':no-result-remains-unknown',assess(g,name,[])['goalSatisfied'] is None)
 check(name+':unknown-guard-requests-evidence',assess_monitor(g,E[name+'FinishMonitor'])['declared_response']==str(Namespace(B+'monitoring-requirements.owl#').RequestEvidence))
for state,expected in [(N.Directed,True),(N.NotDirected,False),(N.AttentionUnknown,None)]:
 h,r=result('attention',attention=state);check('attention-state:'+str(state).split('#')[-1],assess(h,'attention',[r])['goalSatisfied'] is expected)
for state,expected in [(N.Visible,True),(N.Occluded,True),(N.VisibilityUnknown,None)]:
 h,r=result('visibility',visibility=state);check('visibility-assessment:'+str(state).split('#')[-1],assess(h,'visibility',[r])['goalSatisfied'] is expected)
h,r=result('attention',matches=[N.Matches],attention=N.AttentionUnknown);check('candidate-does-not-prove-attention',assess(h,'attention',[r])['goalSatisfied'] is None)
h,r=result('detection',attention=N.Directed);check('attention-does-not-prove-detection',assess(h,'detection',[r])['goalSatisfied'] is False)
for matches,status,goal in [([], 'not-detected-within-scope',False),([N.Matches],'detected',True),([N.Matches,N.Matches],'detected',True),([N.DoesNotMatch],'not-detected-within-scope',False),([N.MatchUnknown],'matching-unresolved',None),([N.Matches,N.MatchUnknown],'detected',True)]:
 h,r=result('detection',matches=matches);out=assess(h,'detection',[r]);check('detection:'+','.join(str(m).split('#')[-1] for m in matches),out['status']==status and out['goalSatisfied'] is goal);check('no-global-absence:'+str(matches),not out['worldAbsenceInferred'])
for matches,status,goal in [([N.Matches],'identified',True),([N.Matches,N.DoesNotMatch],'identified',True),([N.Matches,N.Matches],'ambiguous',None),([N.Matches,N.MatchUnknown],'identity-unresolved',None),([],'not-identified',False)]:
 h,r=result('identification',matches=matches);out=assess(h,'identification',[r]);check('identification:'+str(matches),out['status']==status and out['goalSatisfied'] is goal)
h,r=result('detection',outcome=N.Failed,matches=[N.Matches]);out=assess(h,'detection',[r]);check('failed-operation-not-successful-detection',out['goalSatisfied'] is None and out['operationCompleted'] is False)
h,r=result('detection',outcome=N.Undetermined);check('undetermined-operation-stays-unknown',assess(h,'detection',[r])['goalSatisfied'] is None)
for when,status in [('2026-11-11T09:00:00Z','stale-evidence'),('2026-11-11T10:01:00Z','future-evidence')]:
 h,r=result('detection',when=when,matches=[N.Matches]);check(status+'-does-not-establish-success',assess(h,'detection',[r])['status']==status)
for kind in [A.LLMGeneration,A.DocumentSource]:
 h,r=result('detection',kind=kind,matches=[N.Matches]);check('unsupported-kind:'+str(kind).split('#')[-1],assess(h,'detection',[r])['status']=='unsupported-evidence-kind')
h,r=result('detection',matches=[N.Matches]);before=set(h);out=assess(h,'detection',[r]);check('synthetic-input-explicit',out['syntheticInput']);check('no-belief-publication',out['acceptedBeliefs']==0 and not out['worldUpdated']);check('inspection-does-not-mutate-graph',set(h)==before)
check('result-record-validates-against-core-and-context-shapes',valid(h))
for label,p,value in [('foreign-record',N.inRecord,E.attentionRecord),('foreign-context',N.inContext,E.attentionContext),('foreign-query',N.answersQuery,E.attentionQuery),('wrong-frame',N.resultFrame,E.attentionFrame)]:
 bad=Graph()+h;bad.set((r,p,value));rejects(label,lambda:assess(bad,'detection',[r]))
rejects('duplicate-result-selection',lambda:assess(h,'detection',[r,r]));rejects('negative-freshness-limit',lambda:inspect(h,E.detectionRecord,[r],AT,-1));rejects('nonfinite-freshness-limit',lambda:inspect(h,E.detectionRecord,[r],AT,float('inf')))
bad=Graph()+h;bad.set((r,K.observedAt,Literal('2026-11-11T10:00:00',datatype=XSD.dateTime)));rejects('timezone-required',lambda:assess(bad,'detection',[r]))
# Multiple fresh records require selection, never an implicit latest/first choice.
h2,r2=result('detection','two',matches=[N.Matches]);both=h+h2;check('multiple-results-require-policy',assess(both,'detection',[r,r2])['status']=='multiple-results-require-selection')
candidate=next(h.objects(r,N.hasCandidate));bad=Graph()+h;bad.add((T.foreignResult,N.hasCandidate,candidate));rejects('candidate-hypothesis-result-scope',lambda:assess(bad,'detection',[r]))
# Search has a declared view sequence and explicit budget.
a,ra=result('search','one',view=E.searchView1);b,rb=result('search','two',view=E.searchView2);search=a+b
check('first-empty-view-is-incomplete',assess(search,'search',[ra])['status']=='search-incomplete')
check('all-views-exhausted-is-local',assess(search,'search',[ra,rb])['status']=='declared-views-exhausted' and not assess(search,'search',[ra,rb])['worldAbsenceInferred'])
check('out-of-order-view-not-silently-accepted',assess(search,'search',[rb])['status']=='out-of-order-view-evidence')
bad=Graph()+search;bad.set((E.searchBudget,A.numericValue,Literal(1,datatype=XSD.decimal)));check('budget-exhaustion-distinct-from-view-exhaustion',assess(bad,'search',[ra])['status']=='attempt-budget-exhausted');check('budget-overrun-reported',assess(bad,'search',[ra,rb])['status']=='attempt-budget-exceeded')
found,rf=result('search','found',view=E.searchView1,matches=[N.Matches]);check('search-can-stop-after-a-match',assess(found,'search',[rf])['status']=='search-found');check('results-after-stop-require-policy',assess(found+b,'search',[rf,rb])['status']=='results-beyond-stop-condition')
failed,rr=result('search','failed',view=E.searchView1,outcome=N.Failed);check('search-fault-not-absence',assess(failed,'search',[rr])['status']=='search-operation-failed')
unknown,rr=result('search','unknown',view=E.searchView1,matches=[N.MatchUnknown]);check('unknown-match-does-not-exhaust-search',assess(unknown,'search',[rr])['status']=='search-match-unresolved')
bad=Graph()+search;bad.set((rb,N.forView,E.searchView1));check('multiple-results-for-view-require-policy',assess(bad,'search',[ra,rb])['status']=='multiple-results-for-view-require-selection')
bad=Graph()+search;bad.set((rb,N.forView,T.foreignView));rejects('foreign-search-view',lambda:assess(bad,'search',[ra,rb]))
# Estimated positions are optional hypotheses, not desired targets or published poses.
h,r=result('detection',matches=[N.Matches]);candidate=next(h.objects(r,N.hasCandidate));pose=T.estimate
for triple in [(candidate,N.poseEstimate,pose),(pose,RDF.type,N.PoseEstimate),(pose,N.estimateFrame,E.detectionFrame)]:h.add(triple)
for axis in ['x','y','z']:
 p=T['estimate'+axis]
 for triple in [(pose,N[axis+'Parameter'],p),(p,RDF.type,A.ParameterSpecification),(p,A.parameterKey,Literal('estimate-'+axis,datatype=XSD.string)),(p,A.numericValue,Literal('0.1',datatype=XSD.decimal)),(p,A.unitIRI,Literal('http://qudt.org/vocab/unit/M',datatype=XSD.anyURI))]:h.add(triple)
check('estimate-remains-a-hypothesis',assess(h,'detection',[r])['worldUpdated'] is False)
check('estimate-shape-valid',valid(h))
bad=Graph()+h;bad.set((pose,N.estimateFrame,E.attentionFrame));rejects('foreign-estimate-frame',lambda:assess(bad,'detection',[r]))
bad=Graph()+h;bad.remove((pose,N.xParameter,None));check('incomplete-estimate-shape-rejected',not valid(bad))
# Proposal scope, without an acceptance or publication operation.
h,r=result('detection',matches=[N.Matches]);claim=next(data.objects(E.detectionRecord,A.hasClaim));proposal=T.proposal
for t in [(proposal,RDF.type,N.BeliefUpdateProposal),(proposal,N.basedOnResult,r),(proposal,N.proposedClaim,claim),(proposal,N.proposalContext,E.detectionContext),(proposal,N.proposalRecord,E.detectionRecord)]:h.add(t)
check('proposal-scope-valid',not perception_errors(h));check('proposal-does-not-publish',assess(h,'detection',[r])['worldUpdated'] is False)
bad=Graph()+h;bad.set((proposal,N.proposalContext,E.attentionContext));check('foreign-proposal-context-rejected',bool(perception_errors(bad)))
# Shape and semantic integrity mutations.
for label,subject,p in [('missing-observer',E.detection,N.observer),('missing-query-frame',E.detectionQuery,N.queryFrame),('missing-viewpoint',E.searchView1,N.viewpoint)]:
 bad=Graph()+data;bad.remove((subject,p,None));check('reject-'+label,not valid(bad))
for label,subject,p,value in [('wrong-target',E.detectionQuery,N.queryTarget,E.attentionTarget),('wrong-view-frame',E.searchView1,N.viewFrame,E.attentionFrame),('wrong-view-region',E.searchView1,N.viewRegion,E.attentionRegion),('duplicate-view-index',E.searchView2,N.positionIndex,Literal(1,datatype=XSD.positiveInteger)),('zero-budget',E.searchBudget,A.numericValue,Literal(0,datatype=XSD.decimal)),('fractional-budget',E.searchBudget,A.numericValue,Literal('1.5',datatype=XSD.decimal))]:
 bad=Graph()+g;bad.set((subject,p,value));check('reject-'+label,bool(perception_errors(bad)))
for name in ['attention','visibility','detection','identification','search']:
 covered=set()
 for c in data.objects(E[name+'Record'],A.hasClaim):
  st=one(data,c,A.hasStatement);objs=list(data.objects(st,A.statementObject))+list(data.objects(st,A.statementLiteral));covered.add((one(data,st,A.statementSubject),str(one(data,st,A.statementPredicateIRI)),objs[0]))
 todo=[E[name]];seen=set()
 while todo:
  node=todo.pop()
  if node in seen:continue
  seen.add(node);todo.extend(o for o in data.objects(node) if isinstance(o,URIRef) and str(o).startswith(str(E)))
 expected={(s,str(p),o) for s in seen for p,o in data.predicate_objects(s) if p not in [RDF.type,RDFS.label,RDFS.comment]};check(name+':exact-provenance',bool(expected) and expected<=covered)
check('no-result-observation-in-profile-data',not list(data.subjects(RDF.type,N.PerceptionResultRecord)))
check('no-world-bindings',not list(data.triples((None,A.hasBinding,None))))
check('no-execution-framework-or-mobility-domain-import',not any('aicor-unified' in i or 'cram-action-bridge' in i or 'mobility-navigation.owl' in i for i in imports))
report={'release':'0.16.0','checks':tests,'imports':imports,'scope':'Five hypothetical profiles; result records are synthetic in-memory fixtures only. No sensor acquisition or accepted belief update.'};(ROOT/'reports/perception-attention-validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'checks':len(tests),'passed':sum(t['passed'] for t in tests),'failures':[t for t in tests if not t['passed']]}));assert all(t['passed'] for t in tests)
