import json
from rdflib import Graph,Namespace,RDF,Literal,XSD
from pyshacl import validate
from ontology_imports import ROOT,closure
from aspect_runtime import A,C,structural_errors
from robot_motion import N,motion_errors,assess_completion
B='http://aicor.knowledge/ontologies/';E=Namespace(B+'robot-motion-examples.owl#')
schema,imports=closure(B+'robot-motion.owl');schema-=Graph().parse(ROOT/'DUL.owl');data=Graph().parse(ROOT/'examples/robot-motion-profiles.ttl')
shapes=Graph()
for p in [ROOT/'core/action-core-shapes.ttl',ROOT/'aspects/validation/aspect-shapes.ttl',ROOT/'domains/robot-motion-shapes.ttl']:shapes.parse(p)
def valid(g):
 ok,_,txt=validate(g,shacl_graph=shapes,ont_graph=schema,inference='rdfs',advanced=True)
 if g is data:(ROOT/'reports/robot-motion-shacl.txt').write_text(txt)
 return bool(ok)
g=data+schema;tests=[]
def check(name,ok):tests.append({'case':name,'passed':bool(ok)})
check('six-profile-shacl',valid(data));check('domain-integrity',not motion_errors(g));check('expression-integrity',not structural_errors(g))
profiles=['jointRepositioning','stowing','gripperConfigurationChange','gripperApertureChange','cartesianPositioning','posePathFollowing']
for name in profiles:
 check(name+':unknown-without-context',assess_completion(g,E[name])['conditionValue'] is None)
 check(name+':no-execution-approval',not assess_completion(g,E[name])['executionReadinessAssessed'])
 unit='http://qudt.org/vocab/unit/RAD' if name=='jointRepositioning' else 'http://qudt.org/vocab/unit/M'
 if name in profiles[:4]:values={(C.PostState,E[name+'Error']):('0.001',unit),(C.TaskContext,E[name+'Tolerance']):('0.002',unit)}
 else:values={(C.PostState,E[name+'EndpointError']):('0.001',unit),(C.TaskContext,E[name+'EndpointTolerance']):('0.002',unit)}
 if name=='posePathFollowing':
  check('endpoint-alone-not-path-success',assess_completion(g,E[name],values)['conditionValue'] is None)
  values[(C.PostState,E[name+'TrackingError'])]=('0.01',unit);values[(C.TaskContext,E[name+'TrackingTolerance'])]=('0.002',unit)
  check('bad-path-tracking-fails-despite-endpoint',assess_completion(g,E[name],values)['conditionValue'] is False)
  values[(C.PostState,E[name+'TrackingError'])]=('0.001',unit)
 check(name+':supplied-metrics-satisfy',assess_completion(g,E[name],values)['conditionValue'] is True)
 key=next(k for k in values if k[0]==C.PostState);values[key]=('0.1',unit)
 check(name+':out-of-tolerance-fails',assess_completion(g,E[name],values)['conditionValue'] is False)
 values[key]=('-0.1',unit)
 try:assess_completion(g,E[name],values);check(name+':reject-negative-error',False)
 except ValueError:check(name+':reject-negative-error',True)
for label,s,p in [('missing-controlled-part',E.jointRepositioning,N.controlledEntity),('missing-coordinate-target',E.jointRepositioningEntry,N.targetParameter),('missing-frame',E.cartesianPositioningTarget,N.referenceFrame),('missing-position-axis',E.cartesianPositioningTargetPosition,N.xParameter),('missing-waypoint-pose',E.posePathFollowingWaypoint1,N.waypointPose)]:
 h=Graph()+data;h.remove((s,p,None));check('reject-'+label,not valid(h))
for label,s,p,o in [('duplicate-waypoint-index',E.posePathFollowingWaypoint2,N.positionIndex,Literal(1,datatype=XSD.positiveInteger)),('mixed-frames',E.posePathFollowingPose2,N.referenceFrame,E.cartesianPositioningFrame),('mismatched-coordinate-units',E.jointRepositioningTolerance,A.unitIRI,Literal('urn:test:other-unit',datatype=XSD.anyURI)),('unknown-coordinate-kind',E.jointRepositioningCoordinate,N.coordinateKind,Literal('unknown',datatype=XSD.string)),('negative-speed-limit',E.jointRepositioningSpeedLimit,A.numericValue,Literal('-1',datatype=XSD.decimal))]:
 h=Graph()+g;h.set((s,p,o));check('reject-'+label,bool(motion_errors(h)))
h=Graph()+g;h.set((E.jointRepositioningCoordinate,N.coordinateKind,Literal('linear',datatype=XSD.string)));check('angular-unit-cannot-be-linear-coordinate',bool(motion_errors(h)))
h=Graph()+g;h.remove((E.posePathFollowing,N.trackingErrorParameter,None));check('path-cannot-drop-tracking-metric',bool(motion_errors(h)))
h=Graph()+g;h.set((E.posePathFollowing,N.trackingErrorParameter,E.posePathFollowingEndpointError));check('bound-path-metric-must-match-expression',bool(motion_errors(h)))
h=Graph()+g;h.add((E.cartesianPositioningTarget,N.orientation,E.requestedOrientation));check('orientation-cannot-silently-use-position-only-success',bool(motion_errors(h)))
check('no-world-or-cram-dependency' ,not any('aicor-unified' in i or 'cram-action-bridge' in i for i in imports))
check('no-world-bindings',not list(data.triples((None,A.hasBinding,None))))
check('all-example-claims-proposed',all(data.value(c,A.hasStatus)==A.Proposed for c in data.subjects(RDF.type,A.KnowledgeClaim)))
# Exact edge coverage within each record's reachable descriptive subgraph.
from rdflib import RDFS,URIRef
for name in profiles:
 record=E[name+'Record'];covered=set()
 for c in data.objects(record,A.hasClaim):
  st=data.value(c,A.hasStatement);o=data.value(st,A.statementObject) or data.value(st,A.statementLiteral);covered.add((data.value(st,A.statementSubject),str(data.value(st,A.statementPredicateIRI)),o))
 todo=[E[name]];seen=set()
 while todo:
  node=todo.pop()
  if node in seen:continue
  seen.add(node)
  for o in data.objects(node):
   if isinstance(o,URIRef) and str(o).startswith(str(E)):todo.append(o)
 expected={(s,str(p),o) for s in seen for p,o in data.predicate_objects(s) if p not in [RDF.type,RDFS.label,RDFS.comment]}
 check(name+':exact-statement-coverage',bool(expected) and expected<=covered)
report={'release':'0.13.0','checks':tests,'imports':imports,'scope':'Six synthetic descriptive profiles; supplied error metrics only, no FK/IK or motion execution.'}
(ROOT/'reports/robot-motion-validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'checks':len(tests),'passed':sum(t['passed'] for t in tests),'failures':[t for t in tests if not t['passed']]}));assert all(t['passed'] for t in tests)
