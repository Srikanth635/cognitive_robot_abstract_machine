"""Bounded sequential recipe inspection; no scheduling, fact promotion or execution."""
from rdflib import Namespace,RDF
from aspect_runtime import A,C,P,one,evaluate
from context_review import K,instant
from condition_signatures import signature
from object_manipulation import N as OM,assess as manipulation_assess,manipulation_errors
from tool_material_transfer import TI,assess as tool_assess,tool_errors
from transport_carrying import N as TC,assess_description as carrying_assess,carrying_errors,bounds,conjunction,disjunction
from specialized_tool_interactions import N as ST,PROFILES,specialized_errors,assess as specialized_assess
B='http://aicor.knowledge/ontologies/';N=Namespace(B+'action-recipes.owl#')
PCP_ROLES=[N.Pickup,N.Carry,N.Place]
TOOL_ROLES=[N.AcquireTool,N.PositionTool,N.EngageTool,N.UseTool,N.WithdrawTool,N.ReturnTool]

def ordered_steps(g,action):return sorted(g.objects(action,N.hasStep),key=lambda x:int(one(g,x,N.positionIndex)))
def atom(scope,rel,*args):return scope,rel,args,False
def recipe_errors(g):
 errors=[]
 for validate in [manipulation_errors,tool_errors,carrying_errors,specialized_errors]:
  try:errors.extend(validate(g))
  except ValueError as error:errors.append(str(error))
 actions={x for typ in [N.RecipeDescription,N.PickCarryPlaceDescription,N.ToolUseCycleDescription] for x in g.subjects(RDF.type,typ)}
 for root in actions:
  try:
   pcp=(root,RDF.type,N.PickCarryPlaceDescription) in g;tool=(root,RDF.type,N.ToolUseCycleDescription) in g
   if pcp==tool:raise ValueError('Select one concrete recipe family')
   if one(g,root,A.hasActionKind)!=(N.PickCarryPlace if pcp else N.AcquireUseReturnTool):raise ValueError('Recipe action kind mismatch')
   steps=ordered_steps(g,root);expected=PCP_ROLES if pcp else TOOL_ROLES
   if [int(one(g,x,N.positionIndex)) for x in steps]!=list(range(1,len(expected)+1)) or [one(g,x,N.stepRole) for x in steps]!=expected:raise ValueError('Recipe stages must have contiguous unique indices and the expected semantic order')
   if any(set(g.subjects(N.hasStep,x))!={root} for x in steps):raise ValueError('Stage occurrence must have one recipe owner')
   children=[one(g,x,N.stepAction) for x in steps]
   if len(set(children))!=len(children) or root in children or set(g.objects(root,A.hasSubActionDescription))!=set(children):raise ValueError('Distinct stage action occurrences must match the recipe subactions')
   context=one(g,root,N.evaluationContext);contexts=[one(g,x,N.stepContext) for x in steps]
   if len(set(contexts+[context]))!=len(steps)+1 or one(g,context,K.scopedDescription)!=root:raise ValueError('Recipe and stage contexts must be distinct')
   if any(one(g,c,K.scopedDescription)!=a for c,a in zip(contexts,children)):raise ValueError('Stage context scopes another action')
   operator=one(g,root,N.operator);payload=one(g,root,N.payload);source=one(g,root,N.sourceSupport);dest=one(g,root,N.destinationSupport);region=one(g,root,N.placementRegion)
   if operator==payload or source in [operator,payload] or dest in [operator,payload]:raise ValueError('Collapsed recipe participants/supports')
   for step,child,role,context in zip(steps,children,expected,contexts):
    if role in [N.Pickup,N.Place,N.AcquireTool,N.ReturnTool]:
     cls={N.Pickup:OM.PickupDescription,N.Place:OM.PlacementDescription,N.AcquireTool:OM.GraspAcquisitionDescription,N.ReturnTool:OM.PlacementDescription}[role]
     kind={N.Pickup:OM.PickingUp,N.Place:OM.Placing,N.AcquireTool:OM.GraspAcquisition,N.ReturnTool:OM.Placing}[role]
     if (child,RDF.type,cls) not in g or one(g,child,A.hasActionKind)!=kind:raise ValueError('Manipulation stage family/kind mismatch')
     if one(g,child,OM.manipulator)!=operator or one(g,child,OM.object)!=payload or one(g,child,OM.evaluationContext)!=context:raise ValueError('Manipulation stage participant/context mismatch')
    elif role==N.Carry:
     if not any((child,RDF.type,x) in g for x in [TC.CarryingDescription,TC.RouteCarryingDescription]):raise ValueError('Carry stage must retain the transport composition')
     if one(g,child,TC.load)!=payload or one(g,child,TC.manipulator)!=operator or one(g,child,TC.carrier)!=one(g,root,N.carrier) or one(g,child,TC.destination)!=one(g,root,N.destinationRegion) or one(g,child,TC.evaluationContext)!=context:raise ValueError('Carrying stage bindings mismatch')
    else:
     if one(g,child,TI.operator)!=operator or one(g,child,TI.tool)!=payload or one(g,child,TI.target)!=one(g,root,N.toolTarget) or one(g,child,TI.workingPart)!=one(g,root,N.workingPart) or one(g,child,TI.intendedEffect)!=one(g,root,N.intendedEffect) or one(g,child,TI.evaluationContext)!=context:raise ValueError('Tool stage participant/effect/context mismatch')
     if role==N.UseTool:
      if len([x for x in PROFILES if (child,RDF.type,x) in g])!=1:raise ValueError('Use stage must select a specialized technique')
     else:
      typ,kind={N.PositionTool:(TI.ToolPositioningDescription,TI.PositioningTool),N.EngageTool:(TI.ToolEngagementDescription,TI.EngagingTool),N.WithdrawTool:(TI.ToolDisengagementDescription,TI.DisengagingTool)}[role]
      if (child,RDF.type,typ) not in g or one(g,child,A.hasActionKind)!=kind:raise ValueError('Tool stage family/kind mismatch')
   if one(g,children[-1],OM.destinationSupport)!=dest or one(g,children[-1],OM.targetRegion)!=region:raise ValueError('Final placement support/region mismatch')
   if pcp:
    if one(g,children[0],OM.initialSupport)!=source or one(g,region,N.withinDestination)!=one(g,root,N.destinationRegion):raise ValueError('Pickup source or placement/carrier destination relation mismatch')
   elif source!=dest:raise ValueError('Return-tool recipe must return to its declared original support')
   if tool and len({one(g,a,TI.referenceFrame) for a in children[1:-1]})!=1:raise ValueError('Tool stages need a common declared frame or a future transform contract')
   handoffs=list(g.objects(root,N.hasHandoff));pairs={(one(g,h,N.fromStep),one(g,h,N.toStep)) for h in handoffs}
   if len(handoffs)!=len(steps)-1 or pairs!=set(zip(steps,steps[1:])):raise ValueError('Exactly one handoff per adjacent stage pair required')
   for h in handoffs:
    if set(g.subjects(N.hasHandoff,h))!={root}:raise ValueError('Handoff has multiple owners')
    actual,_=signature(g,one(g,one(g,h,N.handoffCondition),C.hasExpression));required={atom(C.PostState,N.ControlContinuity,operator,payload)}
    if pcp:required.add(atom(C.PostState,N.LoadSupportContinuity,one(g,root,N.carrier),payload))
    if actual!=required:raise ValueError('Handoff continuity condition has missing/foreign participants or scope')
   success=one(g,root,A.hasSuccessCriterion)
   if set(g.objects(success,N.requiresStepSuccess))!=set(steps) or set(g.objects(success,N.requiresStepPrecondition))!=set(steps) or set(g.objects(success,N.requiresHandoff))!=set(handoffs) or one(g,success,N.finalStep)!=steps[-1]:raise ValueError('Recipe success contract omits a stage, prerequisite, handoff or correct final stage')
   if list(g.objects(success,C.hasExpression)) or str(one(g,success,A.expressionLanguage))!='ordered-recipe-contract-v1':raise ValueError('Sequential outcomes must use the qualified recipe contract, not flattened PostState expressions')
   pre,_=signature(g,one(g,one(g,root,N.localPrecondition),C.hasExpression));fault,_=signature(g,one(g,one(g,root,N.localFailure),C.hasExpression))
   required={atom(C.TaskContext,N.RecipeResourcesReady,operator,payload)}
   if tool:required.add(atom(C.PreState,P.Supports,source,payload))
   if pre!=required or fault!={atom(C.CurrentState,N.RecipeFault,operator,payload)}:raise ValueError('Recipe-local prerequisites or fault contract mismatch')
  except ValueError as error:errors.append(str(root)+': '+str(error))
 return errors

def inspect_stage(g,step,value):
 action=one(g,step,N.stepAction);role=one(g,step,N.stepRole);context=one(g,step,N.stepContext)
 facts=value.get('facts');quantities=value.get('quantities');interval=value.get('interval')
 if role==N.Carry:out=carrying_assess(g,action,context,value.get('components'),interval,facts,quantities)
 elif role==N.UseTool:out=specialized_assess(g,action,context,facts,quantities,interval)
 elif role in [N.Pickup,N.Place,N.AcquireTool,N.ReturnTool]:out=manipulation_assess(g,action,facts)
 else:out=tool_assess(g,action,facts,quantities)
 return {'precondition':out['precondition'],'success':out['success'],'failure':out.get('failure',out.get('fault')),'detail':out}

def assess(g,record,context,stages=None,handoffs=None,interval=None,facts=None):
 """Inspect an ordered prefix of stage inputs and explicit gap/boundary evidence."""
 root=one(g,record,A.expressesDescription)
 if str(context)!=str(one(g,root,N.evaluationContext)) or one(g,one(g,root,N.evaluationContext),K.scopedDescription)!=root:raise ValueError('Recipe record/context mismatch')
 ordered=ordered_steps(g,root);stages={} if stages is None else stages;handoffs={} if handoffs is None else handoffs;facts={} if facts is None else facts
 if set(stages)-set(ordered):raise ValueError('Foreign stage input')
 provided=[s for s in ordered if s in stages]
 if provided!=ordered[:len(provided)]:raise ValueError('Stage input must be an ordered prefix; later outcomes cannot substitute for missing history')
 parent=None if interval is None else bounds(interval,context);windows={};results={}
 for step in ordered:
  value=stages.get(step,{})
  if step in stages:
   if not isinstance(value,dict) or 'context' not in value or str(value['context'])!=str(one(g,step,N.stepContext)):raise ValueError('Explicit stage context mismatch')
   if value.get('interval') is not None:
    if parent is None:raise ValueError('Stage windows require a parent window')
    win=bounds(value['interval'],one(g,step,N.stepContext))
    if not parent[0]<=win[0]<win[1]<=parent[1]:raise ValueError('Stage window outside recipe interval')
    windows[step]=win
  results[step]=inspect_stage(g,step,value)
 known=[windows[s] for s in ordered if s in windows]
 if any(a[1]>b[0] for a,b in zip(known,known[1:])):raise ValueError('Sequential stage windows overlap or are out of order')
 fully_bound=parent is not None and len(windows)==len(ordered)
 if ordered[0] in windows and windows[ordered[0]][0]!=parent[0]:raise ValueError('Recipe must start at the first stage boundary')
 if fully_bound and windows[ordered[-1]][1]!=parent[1]:raise ValueError('Recipe must end at the final stage boundary')
 specs=set(g.objects(root,N.hasHandoff))
 if set(handoffs)-specs:raise ValueError('Foreign handoff evidence')
 handoff_results={}
 for h in specs:
  value=handoffs.get(h)
  if value is None:handoff_results[h]=None;continue
  if not isinstance(value,dict) or not {'context','fromStep','toStep','start','end','facts'}<=set(value):raise ValueError('Handoff evidence needs explicit context, endpoints, bounds and facts')
  left=one(g,h,N.fromStep);right=one(g,h,N.toStep)
  if str(value['context'])!=str(context) or str(value['fromStep'])!=str(left) or str(value['toStep'])!=str(right):raise ValueError('Handoff context/stage mismatch')
  if left not in windows or right not in windows:raise ValueError('Handoff evidence requires both adjacent stage windows')
  if instant(value['start'])!=windows[left][1] or instant(value['end'])!=windows[right][0]:raise ValueError('Handoff must cover the exact stage gap or shared boundary')
  handoff_results[h]=evaluate(g,one(g,one(g,h,N.handoffCondition),C.hasExpression),value['facts'])
 local_pre=evaluate(g,one(g,one(g,root,N.localPrecondition),C.hasExpression),facts)
 local_fault=evaluate(g,one(g,one(g,root,N.localFailure),C.hasExpression),facts)
 success=conjunction([True if fully_bound else None,local_pre]+[r[k] for r in results.values() for k in ['precondition','success']]+list(handoff_results.values()))
 fault=disjunction([local_fault]+[r['failure'] for r in results.values()])
 return {'precondition':conjunction([local_pre,results[ordered[0]]['precondition']]),'success':success,'failure':fault,
         'stageResults':{str(k):v for k,v in results.items()},'handoffResults':{str(k):v for k,v in handoff_results.items()},'orderedIntervalBound':fully_bound,
         'successAndFaultReported':success is True and fault is True,'executionReadinessAssessed':False,'executed':False,'acceptedObservations':0,'selectedNextStage':None}
