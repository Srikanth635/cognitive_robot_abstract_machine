"""Synthetic carrying composition checks, not a transport experiment."""
import json
from copy import deepcopy
from rdflib import Graph,Namespace,RDF,RDFS,URIRef,Literal,XSD
from pyshacl import validate
from ontology_imports import ROOT,closure
from aspect_runtime import A,C,one,structural_errors
from condition_signatures import signature
from temporal_runtime import T,M,temporal_errors,assess_monitor
from transport_carrying import N,OM,MN,RM,K,carrying_errors,assess

B='http://aicor.knowledge/ontologies/';E=Namespace(B+'transport-examples.owl#')
schema,imports=closure(B+'transport-carrying.owl');schema-=Graph().parse(ROOT/'DUL.owl')
data=Graph().parse(ROOT/'examples/transport-carrying-profiles.ttl');g=data+schema;shapes=Graph()
for p in [ROOT/'core/action-core-shapes.ttl',ROOT/'core/knowledge-context-shapes.ttl',ROOT/'aspects/validation/aspect-shapes.ttl',ROOT/'aspects/validation/temporal-shapes.ttl',ROOT/'domains/robot-motion-shapes.ttl',ROOT/'domains/object-manipulation-shapes.ttl',ROOT/'domains/mobility-navigation-shapes.ttl',ROOT/'compositions/transport-carrying-shapes.ttl']:shapes.parse(p)
checks=[]
def check(name,ok):checks.append({'case':name,'passed':bool(ok)})
def rejects(name,fn):
 try:fn();check(name,False)
 except ValueError:check(name,True)
ok,_,report=validate(data,shacl_graph=shapes,ont_graph=schema,inference='rdfs',advanced=True);(ROOT/'reports/transport-carrying-shacl.txt').write_text(report)
check('four-composed-profile-shapes',ok);check('composition-bindings',not carrying_errors(g));check('expression-signatures',not structural_errors(g));check('child-phases-and-composition-monitors',not temporal_errors(g))
names=['carrying','routeCarrying','orientationCarrying','routeOrientationCarrying']
START='2026-11-11T10:00:00Z';END='2026-11-11T10:01:00Z'
def window(context):return {'context':str(context),'start':START,'end':END}
def supplied(condition):
 atoms,pairs=signature(g,one(g,condition,C.hasExpression));facts={(scope,rel,args):not neg for scope,rel,args,neg in atoms};quantities={}
 for op,scope,left,rscope,right in pairs:
  quantities[(scope,left)]=('0.01',str(one(g,left,A.unitIRI)));quantities[(rscope,right)]=('0.02',str(one(g,right,A.unitIRI)))
 return facts,quantities
def inputs(name):
 action=E[name];components={}
 for prop,cprop in [(N.holdingRequirement,OM.evaluationContext),(N.mobilityRequirement,MN.evaluationContext)]:
  child=one(g,action,prop);context=one(g,child,cprop);facts,q=supplied(one(g,child,A.hasSuccessCriterion));components[child]={'context':str(context),'interval':window(context),'facts':facts,'quantities':q}
 facts,q=supplied(one(g,action,N.localSuccess))
 return {'components':components,'interval':window(E[name+'Context']),'facts':facts,'quantities':q}
def inspect(name,kwargs=None,graph=g):return assess(graph,E[name+'Record'],E[name+'Context'],**({} if kwargs is None else kwargs))
for name in names:
 action=E[name];hold=one(g,action,N.holdingRequirement);nav=one(g,action,N.mobilityRequirement);kw=inputs(name)
 check(name+':empty-input-is-unknown',inspect(name)['success'] is None)
 result=inspect(name,kw);check(name+':qualified-composition-succeeds',result['success'] is True and result['coextensiveIntervalBound'])
 check(name+':success-not-precondition',result['precondition'] is None)
 check(name+':no-placement-release-or-execution',not result['executed'] and not result['executionReadinessAssessed'] and not result['releasePerformed'] and not result['placementAssessed'] and result['acceptedObservations']==0)
 bad=deepcopy(kw);bad['components'].pop(hold);check(name+':arrival-without-holding-unknown',inspect(name,bad)['success'] is None)
 bad=deepcopy(kw);bad['components'].pop(nav);check(name+':holding-without-arrival-unknown',inspect(name,bad)['success'] is None)
 bad=deepcopy(kw);bad['interval']=None
 for child in bad['components'].values():child.pop('interval')
 out=inspect(name,bad);check(name+':raw-positive-aggregates-without-window-withheld',out['success'] is None and not out['coextensiveIntervalBound'])
 bad=deepcopy(kw);bad['components'][hold].pop('interval');check(name+':parent-window-not-inherited-by-child',inspect(name,bad)['success'] is None)
 bad=deepcopy(kw);bad['components'][nav].pop('interval');check(name+':coextensive-navigation-scope-required',inspect(name,bad)['success'] is None)
 bad=deepcopy(kw);key=(C.PostState,OM.HoldMaintained,(E[name+'Manipulator'],E[name+'Load']));bad['components'][hold]['facts'].pop(key);check(name+':endpoint-grasp-not-maintained-grasp',inspect(name,bad)['success'] is None)
 bad=deepcopy(kw);bad['components'][hold]['facts'][key]=False;check(name+':interrupted-hold-fails-despite-arrival',inspect(name,bad)['success'] is False)
 bad=deepcopy(kw);key=(C.PostState,MN.InRegion,(E[name+'Load'],one(g,action,N.destination)));bad['facts'][key]=False;check(name+':carrier-arrival-not-load-arrival',inspect(name,bad)['success'] is False)
 bad=deepcopy(kw);key=(C.PostState,N.LoadBorneThroughout,(E[name+'Carrier'],E[name+'Load']));bad['facts'][key]=False;check(name+':held-but-not-borne-load-fails-carrying',inspect(name,bad)['success'] is False)
 bad=deepcopy(kw);key=(C.PostState,N.LoadEnvelopeMaintained,(E[name+'Carrier'],E[name+'Load'],one(g,action,N.loadedFootprint)));bad['facts'].pop(key);check(name+':footprint-declaration-not-clearance-evidence',inspect(name,bad)['success'] is None)
 bad=deepcopy(kw);bad['components'][hold]['context']=str(E.foreignContext);rejects(name+':wrong-child-context',lambda:inspect(name,bad))
 bad=deepcopy(kw);bad['components'][hold]['interval']['end']='2026-11-11T10:00:59Z';rejects(name+':shorter-hold-interval',lambda:inspect(name,bad))
 bad=deepcopy(kw);bad['components'][nav]['interval']['start']='2026-11-11T10:00:01Z';rejects(name+':different-navigation-window',lambda:inspect(name,bad))
 bad=deepcopy(kw);bad['components'][hold]['interval']['start']='2026-11-11T11:00:00+01:00';bad['components'][hold]['interval']['end']='2026-11-11T11:01:00+01:00';check(name+':equal-instants-different-timezones',inspect(name,bad)['success'] is True)
 bad=deepcopy(kw);bad['interval']['context']=str(E.foreignContext);rejects(name+':foreign-parent-window',lambda:inspect(name,bad))
 bad=deepcopy(kw);bad['components'][hold].pop('context');rejects(name+':unqualified-component-input',lambda:inspect(name,bad))
 bad=deepcopy(kw);bad['components'][E.foreignChild]=bad['components'][hold];rejects(name+':foreign-component-input',lambda:inspect(name,bad))
 bad=deepcopy(kw);bad['interval']=None;rejects(name+':child-window-needs-parent-window',lambda:inspect(name,bad))
 bad=deepcopy(kw);bad['interval']['start']=END;rejects(name+':zero-duration-window',lambda:inspect(name,bad))
 bad=deepcopy(kw);bad['interval']['start']='2026-11-11T10:00:00';rejects(name+':timezone-required',lambda:inspect(name,bad))
 bad=deepcopy(kw);ff,_=supplied(one(g,hold,A.hasFailureCriterion));bad['components'][hold]['facts'].update(ff);check(name+':success-and-component-fault-exposed',inspect(name,bad)['successAndFaultReported'])
 bad=deepcopy(kw);ff,_=supplied(one(g,action,N.localFailure));bad['facts'].update(ff);check(name+':load-fault-exposed-independently',inspect(name,bad)['failure'] is True)
 check(name+':unknown-load-monitor-declares-suspension',assess_monitor(g,E[name+'LoadMonitor'])['declared_response']==str(M.RequestSuspension) and not assess_monitor(g,E[name+'LoadMonitor'])['response_executed'])
 # All preconditions are assessed independently, with explicit root qualification.
 good=inputs(name)
 for child in [hold,nav]:
  ff,qq=supplied(one(g,child,A.hasPrecondition));good['components'][child]['facts'].update(ff);good['components'][child]['quantities'].update(qq)
 ff,qq=supplied(one(g,action,N.localPrecondition));good['facts'].update(ff);good['quantities'].update(qq)
 check(name+':explicit-preconditions-satisfy',inspect(name,good)['precondition'] is True)
 for req in g.objects(action,N.orientationRequirement):
  error=one(g,req,N.maximumDeviation);bad=inputs(name);bad['quantities'][(C.PostState,error)]=('0.03','http://qudt.org/vocab/unit/RAD');out=inspect(name,bad);check(name+':maximum-deviation-not-endpoint-orientation',out['success'] is False and out['failure'] is None)
  bad=inputs(name);bad['quantities'].pop((C.PostState,error));bad['quantities'][(C.PostState,E.endpointOrientationError)]=('0','http://qudt.org/vocab/unit/RAD');check(name+':endpoint-metric-cannot-substitute',inspect(name,bad)['success'] is None)
  for label,value,unit in [('negative','-1','RAD'),('nonfinite','NaN','RAD'),('wrong-unit','0.01','M')]:
   bad=inputs(name);bad['quantities'][(C.PostState,error)]=(value,'http://qudt.org/vocab/unit/'+unit);rejects(name+':invalid-orientation-'+label,lambda:inspect(name,bad))
 if 'route' in name.lower():
  route=one(g,nav,MN.route);bad=inputs(name);bad['components'][nav]['facts'].pop((C.PostState,MN.RouteFollowed,(E[name+'Carrier'],route)));check(name+':endpoint-does-not-prove-requested-route',inspect(name,bad)['success'] is None)

def mutation(label,subject,predicate,value):
 h=Graph()+g;h.set((subject,predicate,value));check('reject-'+label,bool(carrying_errors(h)))
a=E.carrying;hold=one(g,a,N.holdingRequirement);nav=one(g,a,N.mobilityRequirement);foot=one(g,a,N.loadedFootprint)
for row in [('wrong-load',a,N.load,E.otherLoad),('wrong-carrier',a,N.carrier,E.otherCarrier),('wrong-manipulator-membership',E.carryingManipulator,N.manipulatorOf,E.otherCarrier),('wrong-destination',a,N.destination,E.otherRegion),('wrong-footprint',nav,MN.footprint,E.otherFootprint),('footprint-for-other-load',foot,N.footprintLoad,E.otherLoad),('foreign-holding-child',a,N.holdingRequirement,one(g,E.routeCarrying,N.holdingRequirement)),('wrong-kind',a,A.hasActionKind,N.RouteCarrying),('wrong-policy',a,N.intervalPolicy,Literal('sequential',datatype=XSD.string)),('wrong-local-success',a,N.localSuccess,one(g,E.routeCarrying,N.localSuccess)),('wrong-monitor-context',E.carryingLoadMonitor,N.monitorContext,E.foreignContext),('wrong-motion-phase',a,N.motionPhase,E.carryingNavnavigationComplete),('unsafe-unknown-response',E.carryingLoadMonitor,M.whenUnknown,M.ReportSatisfied),('wrong-orientation-load',E.orientationCarryingOrientation,N.orientedLoad,E.carryingLoad),('orientation-metric-omitted',E.orientationCarrying,N.localSuccess,one(g,E.carrying,N.localSuccess))]:mutation(*row)
h=Graph()+g;expr=one(h,one(h,a,A.hasSuccessCriterion),C.hasExpression);h.remove((expr,C.hasOperand,one(h,one(h,hold,A.hasSuccessCriterion),C.hasExpression)));check('reject-parent-omits-holding',bool(carrying_errors(h)))
h=Graph()+g;h.remove((expr,RDF.type,C.Conjunction));h.add((expr,RDF.type,C.Disjunction));check('reject-arrival-or-holding-instead-of-both',bool(carrying_errors(h)))
h=Graph()+g;h.add((a,A.hasSubActionDescription,E.placement));check('reject-extra-placement-child',bool(carrying_errors(h)))
mutation('holding-child-mislabeled-as-release',hold,A.hasActionKind,OM.Releasing)
mutation('mobility-child-mislabeled-as-route',nav,A.hasActionKind,MN.FollowingRoute)
req=E.orientationCarryingOrientation
for label,p,value in [('quaternion-not-normalized',E.orientationCarryingqx,'2'),('negative-tolerance',one(g,req,N.deviationTolerance),'-0.1')]:
 h=Graph()+g;h.set((p,A.numericValue,Literal(value,datatype=XSD.decimal)))
 if 'quaternion' in label:
  for c in ['qy','qz','qw']:h.set((E['orientationCarrying'+c],A.numericValue,Literal('0',datatype=XSD.decimal)))
 check('reject-'+label,bool(carrying_errors(h)))
# Exact statements, no orphan descriptions, no materialized world/evidence defaults.
reachable=set()
for name in names:
 seen=set();todo=[E[name]]
 while todo:
  n=todo.pop()
  if n in seen:continue
  seen.add(n);todo.extend(o for o in data.objects(n) if isinstance(o,URIRef) and str(o).startswith(str(E)))
 reachable.update(seen);expected={(s,str(p),o) for s in seen for p,o in data.predicate_objects(s) if p not in [RDF.type,RDFS.label,RDFS.comment]};covered=set()
 for claim in data.objects(E[name+'Record'],A.hasClaim):
  st=one(data,claim,A.hasStatement);obj=list(data.objects(st,A.statementObject))+list(data.objects(st,A.statementLiteral));covered.add((one(data,st,A.statementSubject),str(one(data,st,A.statementPredicateIRI)),obj[0]))
 check(name+':exact-statement-provenance',expected==covered and bool(expected))
meta={s for cls in [A.KnowledgeRecord,A.AnnotationRecord,A.KnowledgeClaim,A.Statement,A.EvidenceRecord] for s in data.subjects(RDF.type,cls)};meta.update(data.objects(None,A.aboutComponent))
check('no-orphan-description-nodes',not(set(data.subjects())-reachable-meta));check('all-claims-proposed',all(data.value(c,A.hasStatus)==A.Proposed for c in data.subjects(RDF.type,A.KnowledgeClaim)))
check('no-physical-values',not list(data.triples((None,A.numericValue,None))));check('no-world-bindings',not list(data.triples((None,A.hasBinding,None))))
check('independent-from-framework-and-other-domains',not any('cram-action-bridge' in x or 'aicor-unified' in x or 'articulated-mechanisms.owl' in x or 'tool-interaction.owl' in x for x in imports))
report={'release':'0.19.0','checks':checks,'imports':imports,'scope':'Four synthetic composition fixtures; interval metadata scopes supplied evidence and does not verify coverage or accept observations.'}
(ROOT/'reports/transport-carrying-validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'checks':len(checks),'passed':sum(c['passed'] for c in checks),'failures':[c for c in checks if not c['passed']]}));assert all(c['passed'] for c in checks)
