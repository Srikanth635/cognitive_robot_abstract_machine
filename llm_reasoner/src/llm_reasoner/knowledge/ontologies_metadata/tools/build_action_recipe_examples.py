"""Build bounded synthetic ordered recipes from preserved domain/refinement fixtures."""
from rdflib import Graph,Namespace,RDF,Literal,XSD
from ontology_imports import ROOT
from aspect_runtime import A,C,P
from authoring_helpers import Fixture,clone_description
from context_review import K
from tool_material_transfer import TI
from transport_carrying import N as TC,OM
from specialized_tool_interactions import N as ST
B='http://aicor.knowledge/ontologies/';N=Namespace(B+'action-recipes.owl#');E=Namespace(B+'recipe-examples.owl#')
source=Graph()
for file in ['object-manipulation-profiles.ttl','tool-material-transfer-profiles.ttl','transport-carrying-profiles.ttl','specialized-tool-profiles.ttl']:source.parse(ROOT/'examples'/file)
OMEX=Namespace(B+'object-manipulation-examples.owl#');TIEX=Namespace(B+'tool-transfer-examples.owl#');TCEX=Namespace(B+'transport-examples.owl#');STEX=Namespace(B+'specialized-tool-examples.owl#')
g=Graph();g.bind('ar',N);g.bind('ex',E);f=Fixture(g,E)
for name,carry_source,technique in [('pickCarryPlace','carrying',None),('routeOrientedPickCarryPlace','routeOrientationCarrying',None),('cuttingCycle',None,'cutting'),('wipingCycle',None,'wiping')]:
 pcp=carry_source is not None;root=f.node(name,N.PickCarryPlaceDescription if pcp else N.ToolUseCycleDescription);operator=f.node(name+'Operator',A.EntitySpecification);payload=f.node(name+'Payload',A.EntitySpecification);source_support=f.node(name+'SourceSupport',A.EntitySpecification);destination_support=f.node(name+'DestinationSupport',A.EntitySpecification) if pcp else source_support
 region=E[name+'PlacementRegion'];context=f.node(name+'Context',K.KnowledgeContext);f.add(context,K.contextKey,Literal('synthetic-'+name,datatype=XSD.string));f.add(context,K.scopedDescription,root)
 for p,o in [(N.operator,operator),(N.payload,payload),(N.sourceSupport,source_support),(N.destinationSupport,destination_support),(N.placementRegion,region),(N.evaluationContext,context),(A.hasActionKind,N.PickCarryPlace if pcp else N.AcquireUseReturnTool)]:f.add(root,p,o)
 stages=[]
 def stage(role,old,overrides,context_property):
  index=len(stages)+1;action=E[name+'Stage'+str(index)+'Action'];clone_description(source,g,old,action,E,name+'Stage'+str(index),overrides)
  ctx=g.value(action,context_property)
  if ctx is None:
   ctx=f.node(name+'Stage'+str(index)+'Context',K.KnowledgeContext);f.add(ctx,K.contextKey,Literal(name+'-'+str(index),datatype=XSD.string));f.add(ctx,K.scopedDescription,action);f.add(action,context_property,ctx)
  step=f.node(name+'Step'+str(index),N.RecipeStep)
  for p,o in [(N.stepAction,action),(N.stepRole,role),(N.stepContext,ctx),(N.positionIndex,Literal(index,datatype=XSD.positiveInteger))]:f.add(step,p,o)
  f.add(root,N.hasStep,step);f.add(root,A.hasSubActionDescription,action);stages.append(step)
  for prop in [A.hasRoleSpecification,A.hasCapabilityRequirement]:
   for value in g.objects(action,prop):f.add(root,prop,value)
  return action
 if pcp:
  carrier=f.node(name+'Carrier',A.EntitySpecification);destination=f.node(name+'Destination',A.EntitySpecification);f.add(root,N.carrier,carrier);f.add(root,N.destinationRegion,destination);f.add(region,N.withinDestination,destination)
  stage(N.Pickup,OMEX.pickup,{source.value(OMEX.pickup,OM.manipulator):operator,source.value(OMEX.pickup,OM.object):payload,source.value(OMEX.pickup,OM.initialSupport):source_support},OM.evaluationContext)
  old=TCEX[carry_source];stage(N.Carry,old,{source.value(old,TC.manipulator):operator,source.value(old,TC.load):payload,source.value(old,TC.carrier):carrier,source.value(old,TC.destination):destination},TC.evaluationContext)
 else:
  target=f.node(name+'ToolTarget',A.EntitySpecification);part=f.node(name+'WorkingPart',A.EntitySpecification);effect=f.node(name+'Effect',A.EntitySpecification);f.add(root,N.toolTarget,target);f.add(root,N.workingPart,part);f.add(root,N.intendedEffect,effect)
  old=next(source.subjects(RDF.type,OM.GraspAcquisitionDescription));stage(N.AcquireTool,old,{source.value(old,OM.manipulator):operator,source.value(old,OM.object):payload},OM.evaluationContext)
  for role,old in [(N.PositionTool,TIEX.toolPositioning),(N.EngageTool,TIEX.toolEngagement),(N.UseTool,STEX[technique]),(N.WithdrawTool,TIEX.toolDisengagement)]:
   overrides={source.value(old,TI.operator):operator,source.value(old,TI.tool):payload,source.value(old,TI.workingPart):part,source.value(old,TI.target):target,source.value(old,TI.intendedEffect):effect,source.value(old,TI.referenceFrame):E[name+'ToolFrame']}
   stage(role,old,overrides,TI.evaluationContext)
 stage(N.Place if pcp else N.ReturnTool,OMEX.placement,{source.value(OMEX.placement,OM.manipulator):operator,source.value(OMEX.placement,OM.object):payload,source.value(OMEX.placement,OM.destinationSupport):destination_support,source.value(OMEX.placement,OM.targetRegion):region},OM.evaluationContext)
 handoffs=[]
 for i,(left,right) in enumerate(zip(stages,stages[1:]),1):
  handoff=f.node(name+'Handoff'+str(i),N.HandoffRequirement);f.add(root,N.hasHandoff,handoff);f.add(handoff,N.fromStep,left);f.add(handoff,N.toStep,right)
  terms=[f.atom(name+'Continuity'+str(i),N.ControlContinuity,[operator,payload],C.PostState)]
  if pcp:terms.append(f.atom(name+'LoadContinuity'+str(i),N.LoadSupportContinuity,[carrier,payload],C.PostState))
  cond=f.condition(name+'HandoffCondition'+str(i),A.ConditionSpecification,f.combine(name+'HandoffExpression'+str(i),C.Conjunction,terms));f.add(handoff,N.handoffCondition,cond);handoffs.append(handoff)
 terms=[f.atom(name+'ResourcesReady',N.RecipeResourcesReady,[operator,payload],C.TaskContext)]
 if not pcp:terms.append(f.atom(name+'ToolInitiallySupported',P.Supports,[source_support,payload],C.PreState))
 local_pre=f.condition(name+'LocalPre',A.ConditionSpecification,f.combine(name+'LocalPreExpression',C.Conjunction,terms));f.add(root,N.localPrecondition,local_pre)
 local_fail=f.condition(name+'LocalFault',A.ConditionSpecification,f.atom(name+'ExplicitFault',N.RecipeFault,[operator,payload],C.CurrentState));f.add(root,N.localFailure,local_fail)
 for prop,typ,label,text in [(A.hasPrecondition,A.PreconditionSpecification,'Precondition','First-stage prerequisites and recipe-local resources.'),(A.hasSuccessCriterion,N.RecipeSuccessContract,'Success','All qualified stage prerequisites/outcomes and handoffs; final intended outcome at the final stage.'),(A.hasFailureCriterion,A.FailureCriterion,'Failure','Any explicit stage or recipe fault.'),(A.hasGoal,A.GoalSpecification,'Goal','Final intended placement or supported return, preserving the requested intermediate effects.')]:
  cond=f.node(name+label,typ);f.add(cond,A.expressionLanguage,Literal('ordered-recipe-contract-v1',datatype=XSD.string));f.add(cond,A.lexicalExpression,Literal(text,datatype=XSD.string));f.add(root,prop,cond)
 success=g.value(root,A.hasSuccessCriterion)
 for step in stages:f.add(success,N.requiresStepSuccess,step);f.add(success,N.requiresStepPrecondition,step)
 for handoff in handoffs:f.add(success,N.requiresHandoff,handoff)
 f.add(success,N.finalStep,stages[-1]);f.snapshot(root,name)
g.serialize(ROOT/'examples/action-recipe-profiles.ttl',format='turtle');print({'recipes':4,'triples':len(g)})
