import json
from rdflib import Graph,Namespace,RDF,URIRef,RDFS
from pyshacl import validate
from ontology_imports import ROOT,closure
from aspect_runtime import A,C,P,one,structural_errors
from temporal_runtime import temporal_errors,assess_monitor,assess_outgoing
from object_manipulation import N,required_success,manipulation_errors,assess
B='http://aicor.knowledge/ontologies/';E=Namespace(B+'object-manipulation-examples.owl#');PT=Namespace(B+'phases-transitions.owl#');MR=Namespace(B+'monitoring-requirements.owl#')
schema,imports=closure(B+'object-manipulation.owl');schema-=Graph().parse(ROOT/'DUL.owl');data=Graph().parse(ROOT/'examples/object-manipulation-profiles.ttl');g=data+schema
shapes=Graph()
for p in [ROOT/'core/action-core-shapes.ttl',ROOT/'core/knowledge-context-shapes.ttl',ROOT/'aspects/validation/aspect-shapes.ttl',ROOT/'aspects/validation/temporal-shapes.ttl',ROOT/'domains/robot-motion-shapes.ttl',ROOT/'domains/object-manipulation-shapes.ttl']:shapes.parse(p)
def valid(graph):
 ok,_,text=validate(graph,shacl_graph=shapes,ont_graph=schema,inference='rdfs',advanced=True)
 if graph is data:(ROOT/'reports/object-manipulation-shacl.txt').write_text(text)
 return bool(ok)
tests=[]
def check(name,ok):tests.append({'case':name,'passed':bool(ok)})
check('seven-profiles-conform',valid(data));check('relation-and-expression-signatures',not structural_errors(g));check('phase-and-monitor-integrity',not temporal_errors(g));check('domain-contract-integrity',not manipulation_errors(g))
profiles=['reaching','grasping','holding','pickup','placement','release','fixation']
interval=lambda name:{'context':str(E[name+'Context']),'start':'2026-11-11T10:00:00Z','end':'2026-11-11T10:01:00Z'}
def satisfying(name):return {(scope,relation,args):not negative for scope,relation,args,negative in required_success(g,E[name])}
for name in profiles:
 action=E[name];facts=satisfying(name);i=interval(name) if name in ['holding','fixation'] else None
 check(name+':missing-evidence-is-unknown',assess(g,action)['success'] is None)
 check(name+':bound-required-state-succeeds',assess(g,action,facts,i)['success'] is True)
 check(name+':success-is-not-execution-approval',assess(g,action,facts,i)['executionReadinessAssessed'] is False)
 key=next(iter(facts));incomplete=dict(facts);incomplete.pop(key);check(name+':missing-required-fact-is-unknown',assess(g,action,incomplete,i)['success'] is None)
 false=dict(facts);false[key]=not facts[key];check(name+':violated-required-state-fails',assess(g,action,false,i)['success'] is False)
 h=Graph()+data;h.remove((action,N.manipulator,None));check(name+':requires-manipulator',not valid(h))
 check(name+':fault-unknown-without-evidence',assess(g,action,facts,i)['failure'] is None)
 monitor=assess_monitor(g,E[name+'Monitor']);check(name+':unknown-fault-requests-evidence',monitor['declared_response']==str(MR.RequestEvidence))
# Semantic distinctions, rather than just shape presence.
check('gripper-closed-is-not-a-grasp',assess(g,E.grasping,{(C.PostState,N.GripperClosed,(E.graspingManipulator,)):True})['success'] is None)
check('contact-alone-is-not-a-grasp',assess(g,E.grasping,{(C.PostState,P.InContactWith,(E.graspingManipulator,E.graspingObject)):True})['success'] is None)
check('hold-alone-does-not-establish-pickup',assess(g,E.pickup,{(C.PostState,N.Holds,(E.pickupManipulator,E.pickupObject)):True})['success'] is None)
f=satisfying('pickup');f[(C.PostState,P.Supports,(E.pickupSupport,E.pickupObject))]=True;check('still-supported-is-not-pickup',assess(g,E.pickup,f)['success'] is False)
f=satisfying('placement');f.pop((C.PostState,N.StableOnSupport,(E.placementObject,E.placementSupport)));check('support-without-stability-is-not-placement',assess(g,E.placement,f)['success'] is None)
f=satisfying('placement');f[(C.PostState,N.Holds,(E.placementManipulator,E.placementObject))]=True;check('held-object-is-not-completed-placement',assess(g,E.placement,f)['success'] is False)
check('release-does-not-require-placement-support',assess(g,E.release,satisfying('release'))['success'] is True)
for name in ['holding','fixation']:
 check(name+':endpoint-alone-not-sustained',assess(g,E[name],{(C.PostState,N.Holds,(E[name+'Manipulator'],E[name+'Object'])):True},interval(name))['success'] is None)
 check(name+':aggregate-without-interval-rejected-as-evidence',assess(g,E[name],satisfying(name))['success'] is None)
 try:assess(g,E[name],satisfying(name),interval('other'));check(name+':wrong-context-rejected',False)
 except ValueError:check(name+':wrong-context-rejected',True)
 try:
  bad=interval(name);bad['end']=bad['start'];assess(g,E[name],satisfying(name),bad);check(name+':zero-interval-rejected',False)
 except ValueError:check(name+':zero-interval-rejected',True)
f=satisfying('fixation');f[(C.PostState,N.PoseMaintained,(E.fixationObject,E.fixationAnchor))]=False;check('holding-does-not-establish-fixation',assess(g,E.fixation,f,interval('fixation'))['success'] is False)
f=satisfying('holding');f[(C.CurrentState,N.HoldLost,(E.holdingManipulator,E.holdingObject))]=True;check('conflicting-endpoint-and-fault-flagged',assess(g,E.holding,f,interval('holding'))['successAndFaultReported'])
check('pickup-hold-monitor-only-after-acquisition',(E.pickupExecute,MR.hasMonitoringRequirement,E.pickupHoldMonitor) in g and (E.pickupAcquire,MR.hasMonitoringRequirement,E.pickupHoldMonitor) not in g)
check('unknown-acquisition-selects-no-transition',assess_outgoing(g,E.pickupPhases,E.pickupAcquire)['selected_transition'] is None)
f={(C.CurrentState,N.Holds,(E.pickupManipulator,E.pickupObject)):True,(C.CurrentState,N.GraspFailed,(E.pickupManipulator,E.pickupObject)):True};check('ambiguous-acquisition-is-reported',assess_outgoing(g,E.pickupPhases,E.pickupAcquire,f)['assessment']=='ambiguous')
# Context and scope boundaries are substantive conditions of the profile.
for name in ['holding','fixation']:
 h=Graph()+data;h.remove((E[name],N.evaluationContext,None));check(name+':missing-context-fails-shapes',not valid(h))
h=Graph()+g
from context_review import K
h.set((E.holdingContext,K.scopedDescription,E.fixation));check('foreign-interval-description-rejected',bool(manipulation_errors(h)))
f=satisfying('grasping');key=(C.PostState,N.Holds,(E.graspingManipulator,E.graspingObject));f.pop(key);f[(C.CurrentState,N.Holds,(E.graspingManipulator,E.graspingObject))]=True;check('current-hold-does-not-substitute-post-state',assess(g,E.grasping,f)['success'] is None)
f=satisfying('placement');key=(C.PostState,N.AtTarget,(E.placementObject,E.placementTargetRegion));f.pop(key);f[(C.PostState,N.AtTarget,(E.placementObject,E.reachingTargetRegion))]=True;check('wrong-destination-cannot-substitute',assess(g,E.placement,f)['success'] is None)
f=satisfying('release');f[(C.PreState,N.Holds,(E.releaseManipulator,E.releaseObject))]=False;check('release-precondition-reported-separately',assess(g,E.release,f)['precondition'] is False)
for label,interval0 in [('missing-end',{'context':str(E.holdingContext),'start':'2026-11-11T10:00:00Z'}),('naive-time',{'context':str(E.holdingContext),'start':'2026-11-11T10:00:00','end':'2026-11-11T10:01:00Z'})]:
 try:assess(g,E.holding,satisfying('holding'),interval0);check('reject-'+label,False)
 except ValueError:check('reject-'+label,True)
# Negative semantic mutations and provenance coverage.
for name,action,prop,obj in [('foreign-manipulator',E.grasping,N.manipulator,E.pickupManipulator),('wrong-placement-support',E.placement,N.destinationSupport,E.pickupSupport),('collapsed-pickup-support',E.pickup,N.initialSupport,E.pickupObject),('wrong-motion-target',E.reachingMotion,N.motionTarget,E.placementTargetRegion)]:
 h=Graph()+g;h.set((action,prop,obj));check('reject-'+name,bool(manipulation_errors(h)))
for name in profiles:
 record=E[name+'Record'];covered=set()
 for claim in data.objects(record,A.hasClaim):
  st=one(data,claim,A.hasStatement);objects=list(data.objects(st,A.statementObject))+list(data.objects(st,A.statementLiteral));covered.add((one(data,st,A.statementSubject),str(one(data,st,A.statementPredicateIRI)),objects[0]))
 seen=set();todo=[E[name]]
 while todo:
  node=todo.pop()
  if node in seen:continue
  seen.add(node)
  for obj in data.objects(node):
   if isinstance(obj,URIRef) and str(obj).startswith(str(E)):todo.append(obj)
 expected={(subject,str(p),obj) for subject in seen for p,obj in data.predicate_objects(subject) if p not in [RDF.type,RDFS.label,RDFS.comment]}
 check(name+':exact-statement-coverage',bool(expected) and expected<=covered)
check('no-world-binding',not list(data.triples((None,A.hasBinding,None))))
check('no-execution-framework-import',not any('cram-action-bridge' in iri or 'aicor-unified' in iri for iri in imports))
check('all-claims-proposed',all(data.value(c,A.hasStatus)==A.Proposed for c in data.subjects(RDF.type,A.KnowledgeClaim)))
report={'release':'0.14.0','checks':tests,'imports':imports,'scope':'Seven synthetic domain profiles; externally supplied predicate evidence only, no physical validation or execution.'}
(ROOT/'reports/object-manipulation-validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'checks':len(tests),'passed':sum(t['passed'] for t in tests),'failures':[t for t in tests if not t['passed']]}));assert all(t['passed'] for t in tests)
