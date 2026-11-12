"""Bounded engineering profile checks, not tool execution or fluid simulation."""
from decimal import Decimal
from rdflib import Namespace,RDF
from aspect_runtime import A,C,P,one,evaluate
from context_review import K
TI=Namespace('http://aicor.knowledge/ontologies/tool-interaction.owl#');MT=Namespace('http://aicor.knowledge/ontologies/material-transfer.owl#');TC=Namespace('http://aicor.knowledge/ontologies/transfer-contracts.owl#')
TOOL_KINDS=['ToolPositioningDescription','ToolEngagementDescription','ToolUseDescription','ToolDisengagementDescription']
from condition_signatures import signature,criteria_assessment

def tool_errors(g):
 errors=[];actions={s for k in TOOL_KINDS for s in g.subjects(RDF.type,TI[k])}
 for action in actions:
  kinds=[k for k in TOOL_KINDS if (action,RDF.type,TI[k]) in g]
  if len(kinds)!=1:errors.append('select one concrete tool profile');continue
  kind=kinds[0];operator=one(g,action,TI.operator);tool=one(g,action,TI.tool);part=one(g,action,TI.workingPart);target=one(g,action,TI.target);effect=one(g,action,TI.intendedEffect)
  if tool==target or operator==tool:errors.append('collapsed tool/target/operator')
  if one(g,part,TI.workingPartOf)!=tool:errors.append('working part belongs to another tool')
  if one(g,one(g,action,TI.evaluationContext),K.scopedDescription)!=action:errors.append('tool context mismatch')
  actual,_=signature(g,one(g,one(g,action,A.hasSuccessCriterion),C.hasExpression))
  def atom(rel,args,negative=False):return (C.PostState,rel,tuple(args),negative)
  required=set()
  if kind=='ToolPositioningDescription':required.add(atom(TI.ToolAligned,[part,target]))
  if kind in ['ToolEngagementDescription','ToolUseDescription']:required.add(atom(TI.ToolHeld,[operator,tool]))
  if kind=='ToolUseDescription':required|={atom(TI.ToolFunctional,[tool]),atom(TI.EffectObserved,[target,effect])}
  if kind=='ToolDisengagementDescription':required.add(atom(TI.ToolClear,[tool,target]))
  policy=str(one(g,action,TI.contactPolicy))
  if policy not in ['required','forbidden','unspecified']:errors.append('unknown contact policy')
  if kind=='ToolEngagementDescription' and policy!='required':errors.append('engagement profile requires contact')
  if kind=='ToolDisengagementDescription' and policy!='forbidden':errors.append('disengagement profile requires separation')
  if policy!='unspecified':required.add(atom(P.InContactWith,[part,target],policy=='forbidden'))
  if not required<=actual:errors.append('tool success omits required scoped condition')
  for p in g.objects(action,TI.forceLimit):
   unit=g.value(p,A.unitIRI);v=g.value(p,A.numericValue)
   if str(unit)!='http://qudt.org/vocab/unit/N':errors.append('force limit must use canonical force units')
   if v is not None and (not Decimal(str(v)).is_finite() or Decimal(str(v))<0):errors.append('invalid force limit')
 return errors

def material_errors(g):
 errors=[];classes=[MT.PartialTransferDescription,MT.CompleteTransferDescription,MT.PouringDescription,MT.MaterialTransferDescription];actions={s for c in classes for s in g.subjects(RDF.type,c)}
 for action in actions:
  contract=one(g,action,TC.hasTransferContract);source=one(g,contract,TC.source);destination=one(g,contract,TC.destination);portion=one(g,contract,TC.transferredPortion);substance=one(g,contract,TC.substance)
  if source==destination or portion==substance:errors.append('collapsed transfer referents')
  criterion=one(g,action,A.hasSuccessCriterion)
  if one(g,contract,TC.successCriterion)!=criterion:errors.append('contract and action success disagree')
  context=one(g,action,MT.evaluationContext)
  if one(g,context,K.scopedDescription)!=action:errors.append('transfer context mismatch')
  mode=str(one(g,contract,TC.transferMode));complete=(action,RDF.type,MT.CompleteTransferDescription) in g
  if sum((action,RDF.type,c) in g for c in [MT.PartialTransferDescription,MT.CompleteTransferDescription,MT.PouringDescription])>1:errors.append('mixed concrete transfer families need an explicit composite contract')
  if mode not in ['partial','complete'] or (mode=='complete')!=complete:errors.append('transfer mode/profile mismatch')
  basis=str(one(g,action,MT.quantityBasis));unit={'volume':'http://qudt.org/vocab/unit/M3','mass':'http://qudt.org/vocab/unit/KiloGM'}.get(basis)
  if unit is None:errors.append('unsupported quantity basis')
  delivered=one(g,contract,TC.deliveredAmount);target=one(g,contract,TC.targetAmount);spill=one(g,contract,TC.spillAmount);tolerance=one(g,contract,TC.spillTolerance)
  actual,pairs=signature(g,one(g,criterion,C.hasExpression));required={(C.GreaterOrEqual,C.PostState,delivered,C.TaskContext,target),(C.LessOrEqual,C.PostState,spill,C.TaskContext,tolerance)}
  if (C.PostState,P.ContainedIn,(portion,destination),False) not in actual:errors.append('missing intended destination containment')
  parameters=[delivered,target,spill,tolerance]
  if complete:
   residual=one(g,action,MT.residualAmount);residual_tol=one(g,action,MT.residualTolerance);parameters.extend([residual,residual_tol]);required.add((C.LessOrEqual,C.PostState,residual,C.TaskContext,residual_tol))
  if not required<=pairs:errors.append('transfer success omits required quantity comparison')
  pre,_=signature(g,one(g,one(g,action,A.hasPrecondition),C.hasExpression))
  concrete_ready={(C.PreState,P.ContainedIn,(substance,source),False),(C.PreState,P.Accessible,(destination,),False)}
  if (C.PreState,MT.TransferReady,(source,destination,substance),False) not in pre and not concrete_ready<=pre:errors.append('transfer precondition omits readiness or scoped source-containment/destination-access prerequisites')
  goal,goal_pairs=signature(g,one(g,one(g,action,A.hasGoal),C.hasExpression))
  if not (goal or goal_pairs) or not goal<=actual or not goal_pairs<=pairs:errors.append('success does not establish every declared transfer goal conjunct')
  fault,fault_pairs=signature(g,one(g,one(g,action,A.hasFailureCriterion),C.hasExpression))
  spill_fault=(C.GreaterThan,C.PostState,spill,C.TaskContext,tolerance)
  if ((C.CurrentState,MT.TransferFault,(source,destination),False) not in fault and spill_fault not in fault_pairs) or any(neg or scope!=C.CurrentState for scope,rel,args,neg in fault):errors.append('transfer fault must identify intended endpoints or explicit spill/tolerance exceedance')
  for p in parameters:
   if str(one(g,p,A.unitIRI))!=unit:errors.append('transfer quantity units disagree with basis')
   v=g.value(p,A.numericValue)
   if v is not None and (not Decimal(str(v)).is_finite() or Decimal(str(v))<0):errors.append('invalid transfer amount/tolerance')
  if (action,RDF.type,MT.PouringDescription) in g:
   for interaction in g.objects(action,MT.usesToolInteraction):
    if one(g,interaction,TI.tool)!=source or one(g,interaction,TI.target)!=destination:errors.append('pouring/tool interaction referents disagree')
 return errors

def assess(g,action,facts=None,quantities=None):
 quantities={} if quantities is None else quantities
 _,pairs=signature(g,one(g,one(g,action,A.hasSuccessCriterion),C.hasExpression))
 for op,scope,left,rscope,right in pairs:
  for key in [(scope,left),(rscope,right)]:
   if key not in quantities:continue
   value,unit=quantities[key];v=Decimal(str(value))
   if not v.is_finite() or v<0:raise ValueError('Amounts/tolerances must be finite and nonnegative')
   if str(unit)!=str(one(g,key[1],A.unitIRI)):raise ValueError('Input unit differs from declared unit')
 def truth(prop):return evaluate(g,one(g,one(g,action,prop),C.hasExpression),facts,quantities)
 pre,success,fault=truth(A.hasPrecondition),truth(A.hasSuccessCriterion),truth(A.hasFailureCriterion)
 return {'precondition':pre,'success':success,'fault':fault,'failure':fault,**criteria_assessment(pre,success,fault)}
