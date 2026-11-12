"""Technique-specific supplied-evidence inspection; no physical tool model or execution."""
from decimal import Decimal,InvalidOperation
from rdflib import Namespace,RDF
from aspect_runtime import A,C,P,one
from context_review import K,instant
from condition_signatures import signature
from tool_material_transfer import TI,tool_errors,assess as generic_assess
B='http://aicor.knowledge/ontologies/';N=Namespace(B+'specialized-tool-interactions.owl#')
PROFILES={N.CuttingDescription:(N.Cutting,N.SpecifiedSeparationAchieved),N.WipingDescription:(N.Wiping,N.SpecifiedSurfaceEffectAchieved),N.PressingDescription:(N.Pressing,N.SpecifiedActuationAchieved),N.InsertionDescription:(N.Inserting,N.SpecifiedSeatingAchieved)}
def family(g,action):
 kinds=[k for k in PROFILES if (action,RDF.type,k) in g]
 if len(kinds)!=1:raise ValueError('Exactly one specialized technique required')
 return kinds[0]
def atom(scope,rel,*args):return scope,rel,args,False
def specialized_errors(g):
 errors=[]
 try:errors.extend(tool_errors(g))
 except ValueError as error:errors.append(str(error))
 for action in {s for cls in PROFILES for s in g.subjects(RDF.type,cls)}:
  try:
   kind=family(g,action);canonical,rel=PROFILES[kind]
   if one(g,action,A.hasActionKind)!=canonical:raise ValueError('Technique/action-kind mismatch')
   tool=one(g,action,TI.tool);target=one(g,action,TI.target);part=one(g,action,TI.workingPart);effect=one(g,action,TI.intendedEffect)
   pre,_=signature(g,one(g,one(g,action,A.hasPrecondition),C.hasExpression));actual,pairs=signature(g,one(g,one(g,action,A.hasSuccessCriterion),C.hasExpression))
   if atom(C.TaskContext,N.TechniqueReady,tool,target,effect) not in pre:raise ValueError('Technique readiness missing or wrong referents')
   required={atom(C.PostState,rel,*([tool,target,effect] if kind==N.InsertionDescription else [target,effect]))}
   if kind in [N.WipingDescription,N.PressingDescription] and str(one(g,action,TI.contactPolicy))!='required':raise ValueError('This bounded wiping/maintained-pressing profile requires terminal contact')
   if kind==N.WipingDescription:required.add(atom(C.PostState,N.SweepCoverageMaintained,tool,target))
   if kind==N.InsertionDescription:
    required.add(atom(C.PostState,N.AxisAlignmentSatisfied,part,target));depth=one(g,action,N.targetDepth);error=one(g,action,N.depthError);tol=one(g,action,N.depthTolerance)
    if len({depth,error,tol})!=3:raise ValueError('Insertion target/error/tolerance aliases')
    if one(g,error,N.depthErrorTarget)!=depth or one(g,error,N.depthErrorFrame)!=one(g,action,TI.referenceFrame):raise ValueError('Depth metric target/frame mismatch')
    for p in [depth,error,tol]:
     if str(one(g,p,A.unitIRI))!='http://qudt.org/vocab/unit/M':raise ValueError('Insertion quantities require canonical M units')
     value=g.value(p,A.numericValue)
     if value is not None and (not Decimal(str(value)).is_finite() or Decimal(str(value))<0):raise ValueError('Invalid depth/error/tolerance')
    if (C.LessOrEqual,C.PostState,error,C.TaskContext,tol) not in pairs:raise ValueError('Missing positive depth-error criterion')
   if not required<=actual:raise ValueError('Technique-specific outcome omitted or wrong scope/target')
  except (ValueError,InvalidOperation) as error:errors.append(str(action)+': '+str(error))
 return errors
def assess(g,action,context,facts=None,quantities=None,interval=None):
 if str(context)!=str(one(g,action,TI.evaluationContext)) or one(g,one(g,action,TI.evaluationContext),K.scopedDescription)!=action:raise ValueError('Technique context mismatch')
 kind=family(g,action);facts={} if facts is None else dict(facts);bound=False
 if interval is not None:
  if not {'context','start','end'}<=set(interval) or str(interval['context'])!=str(context):raise ValueError('Technique interval context/start/end required')
  if instant(interval['start'])>=instant(interval['end']):raise ValueError('Invalid technique interval')
  bound=True
 if kind==N.WipingDescription and not bound:facts={key:value for key,value in facts.items() if key[1]!=N.SweepCoverageMaintained}
 out=generic_assess(g,action,facts,quantities);out.update(intervalBound=bound,acceptedObservations=0)
 return out
