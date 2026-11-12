import json
from rdflib import Graph,Namespace,RDF,RDFS,Literal,URIRef,XSD
from pyshacl import validate
from ontology_imports import ROOT,closure
from aspect_runtime import A,C,P,one,structural_errors
from temporal_runtime import temporal_errors,assess_monitor,assess_outgoing
from mobility_navigation import N,RM,signature,required_success,mobility_errors,assess
B='http://aicor.knowledge/ontologies/';E=Namespace(B+'mobility-examples.owl#');K=Namespace(B+'knowledge-context.owl#');PT=Namespace(B+'phases-transitions.owl#');MR=Namespace(B+'monitoring-requirements.owl#')
schema,imports=closure(B+'mobility-navigation.owl');schema-=Graph().parse(ROOT/'DUL.owl');data=Graph().parse(ROOT/'examples/mobility-navigation-profiles.ttl');g=data+schema;shapes=Graph()
for p in [ROOT/'core/action-core-shapes.ttl',ROOT/'core/knowledge-context-shapes.ttl',ROOT/'aspects/validation/aspect-shapes.ttl',ROOT/'aspects/validation/temporal-shapes.ttl',ROOT/'domains/robot-motion-shapes.ttl',ROOT/'domains/mobility-navigation-shapes.ttl']:shapes.parse(p)
def valid(graph):
 ok,_,text=validate(graph,shacl_graph=shapes,ont_graph=schema,inference='rdfs',advanced=True)
 if graph is data:(ROOT/'reports/mobility-navigation-shacl.txt').write_text(text)
 return bool(ok)
tests=[]
def check(n,ok):tests.append({'case':n,'passed':bool(ok)})
check('five-profile-shapes',valid(data));check('condition-and-relation-integrity',not structural_errors(g));check('phase-and-monitor-dependencies',not temporal_errors(g));check('mobility-contracts',not mobility_errors(g))
profiles=['navigation','routeNavigation','facing','levelNavigation','elevatorNavigation']
def interval(name):return {'context':str(E[name+'Context']),'start':'2026-11-11T10:00:00Z','end':'2026-11-11T10:01:00Z'}
def supplied(name):
 atoms,pairs=required_success(g,E[name]);facts={key:True for key in atoms};quantities={}
 for error,tol in pairs:
  unit=str(one(g,error,A.unitIRI));quantities[(C.PostState,error)]=('0.001',unit);quantities[(C.TaskContext,tol)]=('0.01',unit)
 return facts,quantities
for name in profiles:
 f,q=supplied(name);i=interval(name) if name in ['routeNavigation','facing'] else None
 check(name+':unknown-without-evidence',assess(g,E[name])['success'] is None)
 check(name+':required-state-satisfies',assess(g,E[name],f,q,i)['success'] is True)
 check(name+':preconditions-not-assumed',assess(g,E[name],f,q,i)['precondition'] is None)
 check(name+':fault-not-assumed',assess(g,E[name],f,q,i)['failure'] is None)
 check(name+':not-execution-approval',assess(g,E[name],f,q,i)['executionReadinessAssessed'] is False)
 key=next(k for k in q if k[0]==C.PostState);bad=dict(q);bad[key]=('0.2',q[key][1]);check(name+':out-of-tolerance-fails',assess(g,E[name],f,bad,i)['success'] is False)
 bad=dict(q);bad.pop(key);check(name+':missing-metric-is-unknown',assess(g,E[name],f,bad,i)['success'] is None)
 bad=dict(q);bad[key]=('-1',q[key][1])
 try:assess(g,E[name],f,bad,i);check(name+':negative-error-rejected',False)
 except ValueError:check(name+':negative-error-rejected',True)
 bad=dict(q);bad[key]=('0.001','urn:test:wrong-unit')
 try:assess(g,E[name],f,bad,i);check(name+':wrong-unit-rejected',False)
 except ValueError:check(name+':wrong-unit-rejected',True)
# Route and facing evidence must describe the whole selected interval.
for name in ['routeNavigation','facing']:
 f,q=supplied(name);check(name+':no-interval-no-aggregate-success',assess(g,E[name],f,q)['success'] is None)
 for label,iv in [('wrong-context',{**interval(name),'context':'urn:test:foreign'}),('empty-interval',{**interval(name),'end':interval(name)['start']}),('naive-time',{**interval(name),'start':'2026-11-11T10:00:00'})]:
  try:assess(g,E[name],f,q,iv);check(name+':reject-'+label,False)
  except ValueError:check(name+':reject-'+label,True)
f,q=supplied('routeNavigation');f.pop((C.PostState,N.RouteFollowed,(E.routeNavigationRobot,E.routeNavigationRoute)));check('endpoint-alone-not-route-completion',assess(g,E.routeNavigation,f,q,interval('routeNavigation'))['success'] is None)
f,q=supplied('facing');q[(C.PostState,E.facingDriftError)]=('0.2','http://qudt.org/vocab/unit/M');check('correct-heading-with-excess-drift-fails',assess(g,E.facing,f,q,interval('facing'))['success'] is False)
for name in ['levelNavigation','elevatorNavigation']:
 f,q=supplied(name);key=(C.PostState,N.OnLevel,(E[name+'Robot'],E[name+'DestinationLevel']));f.pop(key);f[(C.PostState,N.OnLevel,(E[name+'Robot'],E[name+'SourceLevel']))]=True;check(name+':wrong-level-not-arrival',assess(g,E[name],f,q)['success'] is None)
f,q=supplied('elevatorNavigation');f[(C.PostState,N.OutsideCabin,(E.elevatorNavigationRobot,E.elevatorNavigationCabin))]=False;check('still-in-cabin-is-not-complete',assess(g,E.elevatorNavigation,f,q)['success'] is False)
# Guard semantics: observed closed is distinct from absent open; no edge is executed.
for phase,guard in [('Wait','EntryReady'),('Enter','RideReady'),('Ride','ExitReady')]:
 expression=one(g,E['elevatorNavigation'+guard],C.hasExpression);atoms,_=signature(g,expression);facts={a:True for a in atoms}
 report=assess_outgoing(g,E.elevatorNavigationPhases,E['elevatorNavigation'+phase],facts)
 check(phase+':required-guard-reported-true',any(r['guard_value'] is True for r in report['transitions']))
 check(phase+':no-automatic-transition',report['selected_transition'] is None)
 facts.pop(next(iter(facts)));check(phase+':missing-guard-evidence-remains-incomplete',assess_outgoing(g,E.elevatorNavigationPhases,E['elevatorNavigation'+phase],facts)['assessment']=='incomplete')
facts={(C.CurrentState,N.InsideCabin,(E.elevatorNavigationRobot,E.elevatorNavigationCabin)):True,(C.CurrentState,N.DoorOpen,(E.elevatorNavigationSourceDoor,)):False}
check('not-open-does-not-prove-closed',not any(r['guard_value'] is True for r in assess_outgoing(g,E.elevatorNavigationPhases,E.elevatorNavigationEnter,facts)['transitions']))
# SHACL failures focus on structural completeness.
for label,subject,prop in [('missing-target-pose',E.navigationTarget,N.poseSpecification),('missing-robot',E.navigation,N.robot),('missing-footprint',E.navigation,N.footprint),('missing-context',E.navigation,N.evaluationContext),('missing-segment-destination',E.routeNavigationSegment1,N.toRegion),('missing-map-version',E.routeNavigationMapSnapshot,K.sourceVersion),('missing-elevator-cabin',E.elevatorNavigation,N.cabin)]:
 h=Graph()+data;h.remove((subject,prop,None));check('reject-'+label,not valid(h))
# Semantic mutations remain well-typed but break a domain contract.
for label,subject,prop,obj in [('wrong-target-frame',E.navigationPose,RM.referenceFrame,E.facingFrame),('duplicate-route-index',E.routeNavigationSegment2,N.positionIndex,Literal(1,datatype=XSD.positiveInteger)),('disconnected-route',E.routeNavigationSegment2,N.fromRegion,E.navigationTarget),('wrong-route-destination',E.routeNavigationRoute,N.destinationRegion,E.navigationTarget),('wrong-route-frame',E.routeNavigationRoute,N.routeFrame,E.facingFrame),('wrong-level-binding',E.levelNavigationTarget,N.regionLevel,E.levelNavigationSourceLevel),('collapsed-levels',E.levelNavigation,N.destinationLevel,E.levelNavigationSourceLevel),('foreign-context',E.navigationContext,K.scopedDescription,E.facing),('foreign-cabin-facility',E.elevatorNavigationCabin,N.partOfFacility,E.levelNavigationFacility),('wrong-heading-unit',E.navigationHeadingError,A.unitIRI,Literal('http://qudt.org/vocab/unit/M',datatype=XSD.anyURI)),('wrong-exit-door',E.elevatorNavigationDestinationDoorOpenBinding1,P.bindsEntity,E.elevatorNavigationSourceDoor)]:
 h=Graph()+g;h.set((subject,prop,obj));check('reject-'+label,bool(mobility_errors(h)))
h=Graph()+g;h.set((E.navigationRequiredClearance,A.numericValue,Literal('-1',datatype=XSD.decimal)));check('reject-negative-clearance',bool(mobility_errors(h)))
h=Graph()+g;h.set((E.navigationRequiredClearance,A.unitIRI,Literal('http://qudt.org/vocab/unit/RAD',datatype=XSD.anyURI)));check('reject-angular-clearance-unit',bool(mobility_errors(h)))
h=Graph()+g;h.add((E.foreignRoute,N.hasSegment,E.routeNavigationSegment1));check('reject-shared-indexed-segment-occurrence',bool(mobility_errors(h)))
h=Graph()+g;h.add((E.elevatorNavigation,RDF.type,N.RouteNavigationDescription))
try:mobility_errors(h);check('mixed-family-contract-not-silently-ignored',False)
except ValueError:check('mixed-family-contract-not-silently-ignored',True)
# Renaming a phase must not change its semantic role checks.
h=Graph()
for t in g:h.add(tuple(E.renamedWaitingPhase if n==E.elevatorNavigationWait else n for n in t))
check('stage-checks-independent-of-phase-IRI',not mobility_errors(h))
f,q=supplied('navigation');f[(C.CurrentState,N.NavigationFault,(E.navigationRobot,E.navigationTarget))]=True;check('simultaneous-success-and-fault-reported',assess(g,E.navigation,f,q)['successAndFaultReported'])
for name in profiles:
 claims=set(data.objects(E[name+'Record'],A.hasClaim));covered=set()
 for claim in claims:
  st=one(data,claim,A.hasStatement);objs=list(data.objects(st,A.statementObject))+list(data.objects(st,A.statementLiteral));covered.add((one(data,st,A.statementSubject),str(one(data,st,A.statementPredicateIRI)),objs[0]))
 seen=set();todo=[E[name]]
 while todo:
  node=todo.pop()
  if node in seen:continue
  seen.add(node);todo.extend(o for o in data.objects(node) if isinstance(o,URIRef) and str(o).startswith(str(E)))
 expected={(s,str(p),o) for s in seen for p,o in data.predicate_objects(s) if p not in [RDF.type,RDFS.label,RDFS.comment]};check(name+':exact-provenance-coverage',bool(expected) and expected<=covered)
check('proposed-example-claims-only',all(data.value(c,A.hasStatus)==A.Proposed for c in data.subjects(RDF.type,A.KnowledgeClaim)))
check('no-world-binding',not list(data.triples((None,A.hasBinding,None))))
check('independent-of-object-manipulation-and-execution-framework',not any('object-manipulation.owl' in i or 'aicor-unified' in i or 'cram-action-bridge' in i for i in imports))
report={'release':'0.15.0','checks':tests,'imports':imports,'scope':'Five hypothetical mobility profiles; externally supplied evidence/metrics, no planner, localization, collision checker, elevator controller or execution.'}
(ROOT/'reports/mobility-navigation-validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'checks':len(tests),'passed':sum(t['passed'] for t in tests),'failures':[t for t in tests if not t['passed']]}));assert all(t['passed'] for t in tests)
