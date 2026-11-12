"""Domain authoring integrity and bounded completion checks; no FK/IK or controller."""
from decimal import Decimal
from rdflib import Namespace,RDF
from aspect_runtime import A,C,one,evaluate
N=Namespace('http://aicor.knowledge/ontologies/robot-motion.owl#')
def motion_errors(g):
 errors=[]
 for config in g.subjects(RDF.type,N.ConfigurationTarget):
  entries=list(g.objects(config,N.hasCoordinateTarget));coords=[one(g,e,N.coordinate) for e in entries];keys=[str(one(g,c,N.coordinateKey)) for c in coords]
  if len(coords)!=len(set(coords)) or len(keys)!=len(set(keys)):errors.append('duplicate coordinate target')
  for e in entries:
   kind=str(one(g,one(g,e,N.coordinate),N.coordinateKind))
   if kind not in ['linear','angular']:errors.append('unknown coordinate kind')
   params=[one(g,e,p) for p in [N.targetParameter,N.errorParameter,N.toleranceParameter]]
   units=[g.value(p,A.unitIRI) for p in params]
   if None in units or len(set(units))!=1:errors.append('coordinate target/error/tolerance unit mismatch')
   expected={'http://qudt.org/vocab/unit/M':'linear','http://qudt.org/vocab/unit/RAD':'angular'}.get(str(units[0]))
   if expected is not None and expected!=kind:errors.append('coordinate kind and canonical unit mismatch')
 for path in g.subjects(RDF.type,N.PathSpecification):
  points=list(g.objects(path,N.hasWaypoint));indices=[int(one(g,p,N.positionIndex)) for p in points]
  if sorted(indices)!=list(range(1,len(points)+1)):errors.append('noncontiguous or duplicate waypoint indices')
  frames={one(g,one(g,p,N.waypointPose),N.referenceFrame) for p in points}
  if len(frames)!=1:errors.append('mixed path frames require a transform contract')
 for o in g.subjects(RDF.type,N.OrientationTarget):
  if str(one(g,o,N.orientationConvention))!='quaternion-xyzw':errors.append('unsupported orientation convention')
 for p in g.subjects(RDF.type,N.MotionBounds):
  for prop in [N.speedLimit,N.accelerationLimit]:
   for parameter in g.objects(p,prop):
    value=g.value(parameter,A.numericValue)
    if value is not None and (not Decimal(str(value)).is_finite() or Decimal(str(value))<0):errors.append('negative or nonfinite motion bound')
 for action in g.subjects(RDF.type,N.CartesianMotionDescription):
  if not list(g.objects(action,N.endpointErrorParameter)) or not list(g.objects(action,N.endpointToleranceParameter)):errors.append('missing Cartesian endpoint metric')
 for action in g.subjects(RDF.type,N.PathFollowingDescription):
  if not list(g.objects(action,N.trackingErrorParameter)) or not list(g.objects(action,N.trackingToleranceParameter)):errors.append('path tracking metric required beyond endpoint')
 for action in set(g.subjects(N.poseTarget,None))|set(g.subjects(N.pathTarget,None)):
  poses=list(g.objects(action,N.poseTarget))
  for path in g.objects(action,N.pathTarget):poses.extend(g.value(w,N.waypointPose) for w in g.objects(path,N.hasWaypoint))
  if any(list(g.objects(p,N.orientation)) for p in poses):
   if not list(g.objects(action,N.orientationErrorParameter)) or not list(g.objects(action,N.orientationToleranceParameter)):errors.append('orientation target needs an angular completion metric')
 # The bounded authoring profile requires each bound metric pair as a positive
 # conjunct, rather than merely mentioning it somewhere inside an arbitrary formula.
 actions=set(g.subjects(N.configurationTarget,None))|set(g.subjects(N.poseTarget,None))|set(g.subjects(N.pathTarget,None))
 for action in actions:
  criterion=one(g,action,A.hasSuccessCriterion);todo=[one(g,criterion,C.hasExpression)];seen=set();pairs=set()
  while todo:
   node=todo.pop()
   if node in seen:continue
   seen.add(node)
   if (node,RDF.type,C.Conjunction) in g:todo.extend(g.objects(node,C.hasOperand));continue
   if (node,RDF.type,C.Comparison) not in g:errors.append('unsupported motion completion expression');continue
   left=one(g,node,C.leftOperand);right=one(g,node,C.rightOperand)
   if g.value(node,C.operator)!=C.LessOrEqual or g.value(left,C.inScope)!=C.PostState or g.value(right,C.inScope)!=C.TaskContext:errors.append('motion metric comparison must be post error <= context tolerance');continue
   pairs.add((one(g,left,C.parameter),one(g,right,C.parameter)))
  required=[]
  for target in g.objects(action,N.configurationTarget):
   required.extend((one(g,e,N.errorParameter),one(g,e,N.toleranceParameter)) for e in g.objects(target,N.hasCoordinateTarget))
  for ep,tp in [(N.endpointErrorParameter,N.endpointToleranceParameter),(N.trackingErrorParameter,N.trackingToleranceParameter),(N.orientationErrorParameter,N.orientationToleranceParameter)]:
   if list(g.objects(action,ep)):required.append((one(g,action,ep),one(g,action,tp)))
  if not set(required)<=pairs:errors.append('completion omits a required positive metric comparison')
 return errors

def assess_completion(g,action,quantities=None):
 quantities={} if quantities is None else quantities
 criterion=one(g,action,A.hasSuccessCriterion);expression=one(g,criterion,C.hasExpression)
 nodes=set();todo=[expression]
 while todo:
  n=todo.pop()
  if n in nodes:continue
  nodes.add(n)
  for prop in [C.hasOperand,C.leftOperand,C.rightOperand]:todo.extend(g.objects(n,prop))
 for n in nodes:
  if (n,RDF.type,C.ParameterReference) not in g:continue
  param=one(g,n,C.parameter);value=quantities.get((one(g,n,C.inScope),param))
  if value is None:continue
  scalar,unit=value;scalar=Decimal(str(scalar))
  if not scalar.is_finite() or scalar<0:raise ValueError('Completion errors and tolerances must be finite and nonnegative')
  declared=g.value(param,A.unitIRI)
  if declared is None or str(declared)!=str(unit):raise ValueError('Supplied completion unit does not match its parameter')
 return {'conditionValue':evaluate(g,expression,quantities=quantities),'executionReadinessAssessed':False,'motionExecuted':False}
