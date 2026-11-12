"""Bounded descriptive manipulation checks; no world lookup or robot execution."""
from rdflib import Namespace,RDF
from aspect_runtime import A,C,P,one,evaluate
from context_review import instant
N=Namespace('http://aicor.knowledge/ontologies/object-manipulation.owl#')
KINDS=['ReachingDescription','GraspAcquisitionDescription','HoldingDescription','PickupDescription','PlacementDescription','ReleaseDescription','FixationDescription']
from condition_atoms import atoms
from condition_signatures import criteria_assessment

def required_precondition(g,action):
 """Necessary prerequisites of the current bounded pickup/place profiles."""
 man=one(g,action,N.manipulator);obj=one(g,action,N.object)
 if (action,RDF.type,N.PickupDescription) in g:
  return {(C.PreState,P.Graspable,(obj,),False),
          (C.PreState,P.Supports,(one(g,action,N.initialSupport),obj),False)}
 if (action,RDF.type,N.PlacementDescription) in g:
  return {(C.PreState,N.Holds,(man,obj),False)}
 return set()

def required_success(g,action):
 kind=next((k for k in KINDS if (action,RDF.type,N[k]) in g),None)
 if kind is None:raise ValueError('Unknown manipulation profile')
 man=one(g,action,N.manipulator);obj=one(g,action,N.object)
 def atom(rel,args,negative=False):return (C.PostState,rel,tuple(args),negative)
 held=atom(N.Holds,[man,obj])
 if kind=='ReachingDescription':return {atom(N.AtTarget,[man,one(g,action,N.targetRegion)])}
 if kind=='GraspAcquisitionDescription':return {held,atom(P.InContactWith,[man,obj])}
 if kind=='PickupDescription':
  support=one(g,action,N.initialSupport);return {held,atom(P.Supports,[support,obj],True),atom(N.ClearOfSupport,[obj,support])}
 if kind=='PlacementDescription':
  support=one(g,action,N.destinationSupport);return {atom(P.Supports,[support,obj]),atom(N.StableOnSupport,[obj,support]),atom(N.AtTarget,[obj,one(g,action,N.targetRegion)]),atom(N.Holds,[man,obj],True),atom(N.ManipulatorClear,[man,obj])}
 if kind=='ReleaseDescription':return {atom(N.Holds,[man,obj],True)}
 result={held,atom(N.HoldMaintained,[man,obj])}
 if kind=='FixationDescription':result.add(atom(N.PoseMaintained,[obj,one(g,action,N.anchorFrame)]))
 return result

def manipulation_errors(g):
 errors=[];actions={s for k in KINDS for s in g.subjects(RDF.type,N[k])}
 for action in actions:
  kinds=[k for k in KINDS if (action,RDF.type,N[k]) in g]
  if len(kinds)!=1:errors.append('profile must select one concrete manipulation family');continue
  man=one(g,action,N.manipulator);obj=one(g,action,N.object)
  if man==obj:errors.append('manipulator and object collapsed')
  for prop in [N.initialSupport,N.destinationSupport]:
   for support in g.objects(action,prop):
    if support in [man,obj]:errors.append('support identity collapsed')
  roleEntities={e for role in g.objects(action,A.hasRoleSpecification) for e in g.objects(role,A.hasEntitySpecification)}
  participants={man,obj}|{entity for prop in [N.initialSupport,N.destinationSupport,N.targetRegion,N.anchorFrame] for entity in g.objects(action,prop)}
  if not participants<=roleEntities:errors.append('participant roles disagree with direct bindings')
  actual=atoms(g,one(g,one(g,action,A.hasSuccessCriterion),C.hasExpression))
  if not required_success(g,action)<=actual:errors.append('success omits required state/participant/scope')
  pre=atoms(g,one(g,one(g,action,A.hasPrecondition),C.hasExpression))
  if not required_precondition(g,action)<=pre:errors.append('precondition omits required initial support/graspability/holding referent or scope')
  if kinds[0] in ['PickupDescription','PlacementDescription']:
   goal=atoms(g,one(g,one(g,action,A.hasGoal),C.hasExpression))
   if not goal or not goal<=actual:errors.append('success does not establish every declared manipulation goal conjunct')
  for motion in g.objects(action,N.requiresMotion):
   if one(g,motion,N.motionControlledEntity)!=man:errors.append('motion requirement controls another entity')
   if g.value(action,N.targetRegion) is not None and g.value(motion,N.motionTarget)!=g.value(action,N.targetRegion):errors.append('motion target differs from action target')
  if list(g.objects(action,N.evaluationContext)) or kinds[0] in ['HoldingDescription','FixationDescription']:
   context=one(g,action,N.evaluationContext)
   from context_review import K
   if g.value(context,K.scopedDescription)!=action:errors.append('interval context describes another action')
  fault=atoms(g,one(g,one(g,action,A.hasFailureCriterion),C.hasExpression))
  if any(negative or scope!=C.CurrentState for scope,rel,args,negative in fault):errors.append('failure must use explicit current fault evidence')
  if kinds[0]=='PickupDescription' and (C.CurrentState,N.HoldLost,(man,obj),False) not in fault:errors.append('pickup fault must identify the intended hold')
  if kinds[0]=='PlacementDescription' and (C.CurrentState,N.SupportUnstable,(obj,one(g,action,N.destinationSupport)),False) not in fault:errors.append('placement fault must identify the intended support')
 return errors

def assess(g,action,facts=None,interval=None):
 facts={} if facts is None else dict(facts);temporal=(action,RDF.type,N.HoldingDescription) in g or (action,RDF.type,N.FixationDescription) in g
 interval_bound=False
 if temporal:
  if interval is not None:
   if not {'context','start','end'}<=set(interval):raise ValueError('Interval context, start and end required')
   if str(interval.get('context'))!=str(one(g,action,N.evaluationContext)):raise ValueError('Wrong interval context')
   if instant(interval['start'])>=instant(interval['end']):raise ValueError('Interval start must precede end')
   interval_bound=True
  else:
   facts={key:value for key,value in facts.items() if key[1] not in [N.HoldMaintained,N.PoseMaintained]}
 success=evaluate(g,one(g,one(g,action,A.hasSuccessCriterion),C.hasExpression),facts=facts)
 pre=evaluate(g,one(g,one(g,action,A.hasPrecondition),C.hasExpression),facts=facts)
 fault=evaluate(g,one(g,one(g,action,A.hasFailureCriterion),C.hasExpression),facts=facts)
 return {'precondition':pre,'success':success,'failure':fault,'intervalBound':interval_bound,**criteria_assessment(pre,success,fault)}
