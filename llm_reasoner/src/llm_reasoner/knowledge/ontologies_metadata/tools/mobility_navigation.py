"""Bounded mobility authoring checks and supplied-evidence assessment, not navigation."""
from decimal import Decimal
from rdflib import Namespace,RDF
from aspect_runtime import A,C,P,one,evaluate
from context_review import K,instant
from condition_signatures import signature as condition_signature,criteria_assessment
N=Namespace('http://aicor.knowledge/ontologies/mobility-navigation.owl#');RM=Namespace('http://aicor.knowledge/ontologies/robot-motion.owl#')
KINDS=['ElevatorNavigationDescription','InterLevelNavigationDescription','RouteNavigationDescription','FacingDescription','NavigationDescription']
def family(g,action):
 kind=next((k for k in KINDS if (action,RDF.type,N[k]) in g),None)
 if kind is None:raise ValueError('Unknown mobility profile')
 allowed={'ElevatorNavigationDescription':{'ElevatorNavigationDescription','InterLevelNavigationDescription','NavigationDescription'},'InterLevelNavigationDescription':{'InterLevelNavigationDescription','NavigationDescription'},'RouteNavigationDescription':{'RouteNavigationDescription','NavigationDescription'},'FacingDescription':{'FacingDescription'},'NavigationDescription':{'NavigationDescription'}}[kind]
 if any((action,RDF.type,N[k]) in g for k in KINDS if k not in allowed):raise ValueError('Mixed concrete mobility families need an explicit composite contract')
 return kind

def signature(g,expression):
 atoms=set();metrics=set();seen=set();active=set()
 def visit(n):
  if n in active:raise ValueError('Cyclic expression')
  if n in seen:return
  active.add(n)
  if (n,RDF.type,C.Conjunction) in g:
   for child in g.objects(n,C.hasOperand):visit(child)
  elif (n,RDF.type,C.AtomicCondition) in g:
   pattern=one(g,n,C.hasPattern);bindings=[]
   for b in g.objects(pattern,P.hasArgumentBinding):bindings.append((int(one(g,one(g,b,P.bindsArgument),P.argumentPosition)),one(g,b,P.bindsEntity)))
   atoms.add((one(g,n,C.inScope),one(g,pattern,P.usesRelation),tuple(e for _,e in sorted(bindings))))
  elif (n,RDF.type,C.Comparison) in g:
   left=one(g,n,C.leftOperand);right=one(g,n,C.rightOperand)
   if one(g,n,C.operator)!=C.LessOrEqual or one(g,left,C.inScope)!=C.PostState or one(g,right,C.inScope)!=C.TaskContext:raise ValueError('Expected post-state error <= context tolerance')
   metrics.add((one(g,left,C.parameter),one(g,right,C.parameter)))
  else:raise ValueError('Unsupported mobility condition profile')
  active.remove(n);seen.add(n)
 visit(expression);return atoms,metrics

def required_success(g,action):
 kind=family(g,action);robot=one(g,action,N.robot);target=one(g,action,N.targetRegion)
 atoms=set();pairs={(one(g,action,N.headingErrorParameter),one(g,action,N.headingToleranceParameter))}
 if kind=='FacingDescription':pairs.add((one(g,action,N.translationDriftParameter),one(g,action,N.translationToleranceParameter)))
 else:
  atoms.add((C.PostState,N.InRegion,(robot,target)));pairs.add((one(g,action,N.positionErrorParameter),one(g,action,N.positionToleranceParameter)))
 if kind=='RouteNavigationDescription':atoms.add((C.PostState,N.RouteFollowed,(robot,one(g,action,N.route))))
 if kind in ['InterLevelNavigationDescription','ElevatorNavigationDescription']:atoms.add((C.PostState,N.OnLevel,(robot,one(g,action,N.destinationLevel))))
 if kind=='ElevatorNavigationDescription':atoms.add((C.PostState,N.OutsideCabin,(robot,one(g,action,N.cabin))))
 return atoms,pairs

def mobility_errors(g):
 from robot_motion import motion_errors
 errors=list(motion_errors(g));actions={a for k in KINDS for a in g.subjects(RDF.type,N[k])}
 for action in actions:
  kind=family(g,action);robot=one(g,action,N.robot);target=one(g,action,N.targetRegion);frame=one(g,action,N.referenceFrame);context=one(g,action,N.evaluationContext);footprint=one(g,action,N.footprint)
  if robot in [target,footprint] or target==footprint:errors.append('collapsed robot/target/footprint identity')
  if one(g,context,K.scopedDescription)!=action:errors.append('context describes another action')
  pose=one(g,target,N.poseSpecification)
  if one(g,pose,RM.referenceFrame)!=frame:errors.append('target frame disagrees with action frame')
  requiredAtoms,requiredPairs=required_success(g,action);atoms,pairs=signature(g,one(g,one(g,action,A.hasSuccessCriterion),C.hasExpression))
  if not requiredAtoms<=atoms or not requiredPairs<=pairs:errors.append('success omits required state or metric')
  goal_atoms,goal_pairs=signature(g,one(g,one(g,action,A.hasGoal),C.hasExpression))
  if not (goal_atoms or goal_pairs) or not goal_atoms<=atoms or not goal_pairs<=pairs:errors.append('success does not establish every declared navigation goal conjunct')
  participants={e for role in g.objects(action,A.hasRoleSpecification) for e in g.objects(role,A.hasEntitySpecification)}
  if not {robot,target,footprint}<=participants:errors.append('mobility participant roles disagree with direct robot/target/footprint bindings')
  fault,_=condition_signature(g,one(g,one(g,action,A.hasFailureCriterion),C.hasExpression))
  if (C.CurrentState,N.NavigationFault,(robot,target),False) not in fault or any(neg or scope!=C.CurrentState for scope,rel,args,neg in fault):errors.append('navigation fault must be explicit current evidence for intended robot/target')
  pre,_=signature(g,one(g,one(g,action,A.hasPrecondition),C.hasExpression))
  needed={(C.TaskContext,N.MotionCapabilityAvailable,(robot,)),(C.PreState,N.NavigableTo,(robot,target,footprint))}
  if kind=='RouteNavigationDescription':needed.add((C.PreState,N.RouteTraversableBy,(one(g,action,N.route),robot,footprint)))
  if kind in ['InterLevelNavigationDescription','ElevatorNavigationDescription']:
   source=one(g,action,N.sourceLevel);dest=one(g,action,N.destinationLevel);facility=one(g,action,N.transitionFacility)
   if source==dest:errors.append('inter-level source/destination collapsed')
   if one(g,target,N.regionLevel)!=dest:errors.append('target region level disagrees with destination')
   needed|={(C.PreState,N.OnLevel,(robot,source)),(C.TaskContext,N.FacilityServiceAvailable,(facility,source,dest))}
  if not needed<=pre:errors.append('precondition omits required scoped traversability/capability')
  clearance=one(g,action,N.clearanceParameter)
  if str(one(g,clearance,A.unitIRI))!='http://qudt.org/vocab/unit/M':errors.append('clearance must declare the canonical length unit')
  raw=g.value(clearance,A.numericValue)
  if raw is not None and (not Decimal(str(raw)).is_finite() or Decimal(str(raw))<0):errors.append('negative or nonfinite clearance')
  for error,tol in pairs:
   unit=g.value(error,A.unitIRI)
   if unit is None or unit!=g.value(tol,A.unitIRI):errors.append('metric/tolerance unit mismatch')
   expected='http://qudt.org/vocab/unit/RAD' if error==g.value(action,N.headingErrorParameter) else 'http://qudt.org/vocab/unit/M'
   if str(unit)!=expected:errors.append('wrong canonical mobility metric unit')
   for p in [error,tol]:
    raw=g.value(p,A.numericValue)
    if raw is not None and (not Decimal(str(raw)).is_finite() or Decimal(str(raw))<0):errors.append('negative or nonfinite metric/tolerance')
  if kind=='RouteNavigationDescription':
   route=one(g,action,N.route);segments=list(g.objects(route,N.hasSegment))
   if any(len(set(g.subjects(N.hasSegment,segment)))!=1 for segment in segments):errors.append('route segment occurrence must have one owner')
   ordered=sorted(segments,key=lambda s:int(one(g,s,N.positionIndex)))
   if [int(one(g,s,N.positionIndex)) for s in ordered]!=list(range(1,len(segments)+1)):errors.append('noncontiguous or duplicate route indices')
   if one(g,route,N.routeFrame)!=frame:errors.append('route frame mismatch')
   if one(g,route,N.destinationRegion)!=target:errors.append('route has another destination')
   if ordered and (one(g,ordered[0],N.fromRegion)!=one(g,route,N.sourceRegion) or one(g,ordered[-1],N.toRegion)!=target):errors.append('route endpoints disagree')
   if any(one(g,a,N.toRegion)!=one(g,b,N.fromRegion) for a,b in zip(ordered,ordered[1:])):errors.append('disconnected route segments')
  if kind=='ElevatorNavigationDescription':
   cabin=one(g,action,N.cabin)
   if one(g,cabin,N.partOfFacility)!=one(g,action,N.transitionFacility):errors.append('cabin belongs to another facility')
   # Guard requirements are attached by their actual source/destination phase roles.
   PT=Namespace('http://aicor.knowledge/ontologies/phases-transitions.owl#');structure=one(g,action,PT.hasPhaseStructure);edges=list(g.objects(structure,PT.hasTransition))
   def guarded(srcRole,dstRole,required):
    matching=[e for e in edges if (one(g,e,PT.fromPhase),N.stageRole,srcRole) in g and (one(g,e,PT.toPhase),N.stageRole,dstRole) in g]
    if len(matching)!=1:errors.append('missing or ambiguous elevator stage transition');return
    condition=one(g,matching[0],PT.guardCondition);actual,_=signature(g,one(g,condition,C.hasExpression))
    if not required<=actual:errors.append('elevator stage omits required safety observation')
   # Semantic roles keep this check independent of concrete phase identifiers.
   guarded(N.WaitForEntry,N.EnteringCabin,{(C.CurrentState,N.DoorOpen,(one(g,action,N.sourceDoor),)),(C.CurrentState,N.CabinAtLevel,(cabin,one(g,action,N.sourceLevel))),(C.CurrentState,N.EntryClear,(robot,cabin,footprint))})
   guarded(N.EnteringCabin,N.RidingCabin,{(C.CurrentState,N.InsideCabin,(robot,cabin)),(C.CurrentState,N.DoorClosed,(one(g,action,N.sourceDoor),))})
   guarded(N.RidingCabin,N.ExitingCabin,{(C.CurrentState,N.CabinAtLevel,(cabin,one(g,action,N.destinationLevel))),(C.CurrentState,N.DoorOpen,(one(g,action,N.destinationDoor),)),(C.CurrentState,N.ExitClear,(robot,target,footprint))})
 return errors

def assess(g,action,facts=None,quantities=None,interval=None):
 kind=family(g,action);facts={} if facts is None else dict(facts);quantities={} if quantities is None else dict(quantities);context=one(g,action,N.evaluationContext);intervalBound=False
 if kind in ['FacingDescription','RouteNavigationDescription']:
  if interval is not None:
   if not {'context','start','end'}<=set(interval):raise ValueError('Interval context/start/end required')
   if str(interval['context'])!=str(context):raise ValueError('Wrong evaluation context')
   if instant(interval['start'])>=instant(interval['end']):raise ValueError('Interval start must precede end')
   intervalBound=True
  else:
   facts={key:value for key,value in facts.items() if key[1]!=N.RouteFollowed}
   if kind=='FacingDescription':quantities.pop((C.PostState,one(g,action,N.translationDriftParameter)),None)
 _,pairs=required_success(g,action)
 for error,tol in pairs:
  for scope,p in [(C.PostState,error),(C.TaskContext,tol)]:
   if (scope,p) not in quantities:continue
   value,unit=quantities[(scope,p)];v=Decimal(str(value))
   if not v.is_finite() or v<0:raise ValueError('Errors and tolerances must be finite and nonnegative')
   if str(unit)!=str(one(g,p,A.unitIRI)):raise ValueError('Supplied metric unit differs from declaration')
 def truth(prop):return evaluate(g,one(g,one(g,action,prop),C.hasExpression),facts,quantities)
 pre=truth(A.hasPrecondition);success=truth(A.hasSuccessCriterion);fault=truth(A.hasFailureCriterion)
 return {'precondition':pre,'success':success,'failure':fault,'intervalBound':intervalBound,**criteria_assessment(pre,success,fault)}
