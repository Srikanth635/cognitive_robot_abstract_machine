"""Strategy/profile structural inspection. No provider matching or recovery execution."""
from rdflib import Namespace,RDF
from aspect_runtime import A
B='http://aicor.knowledge/ontologies/';S=Namespace(B+'capabilities-strategies.owl#');F=Namespace(B+'failure-diagnosis.owl#')
def strategy_errors(g):
 errors=[];owners={}
 for strategy in g.subjects(RDF.type,S.ExecutionStrategy):
  members=set(g.objects(strategy,S.hasStep));done=set();active=set()
  def visit(n):
   if n in active:errors.append('cyclic strategy order');return
   if n in done:return
   active.add(n)
   for dst in g.objects(n,S.precedesStep):
    if dst not in members:errors.append('foreign ordered step')
    else:visit(dst)
   active.remove(n);done.add(n)
  for step in members:
   owners.setdefault(step,set()).add(strategy);visit(step)
 for step in g.subjects(RDF.type,S.StrategyStep):
  if len(owners.get(step,set()))!=1:errors.append('step must have one owning strategy')
 for action in g.subjects(S.hasStrategy,None):
  required=set()
  for profile in g.objects(action,S.hasCapabilityProfile):required.update(g.objects(profile,S.requiresCapability))
  for strategy in g.objects(action,S.hasStrategy):
   for step in g.objects(strategy,S.hasStep):
    for proc in g.objects(step,S.requiresProcess):
     if not set(g.objects(proc,S.fulfillsRequirement))<=required:errors.append('strategy requirement absent from action profile')
 for profile in g.subjects(RDF.type,F.FailureProfile):
  members=set(g.objects(profile,F.hasFailure))
  for hypothesis in g.objects(profile,F.hasHypothesis):
   if not set(g.objects(hypothesis,F.explainsFailure))<=members:errors.append('foreign diagnostic failure')
  for recovery in g.objects(profile,F.hasRecoveryOption):
   if not set(g.objects(recovery,F.addressesFailure))<=members:errors.append('foreign recovery failure')
 for recovery in g.subjects(RDF.type,F.RecoveryOption):
  for budget in g.objects(recovery,F.retryBudget):
   value=g.value(budget,A.numericValue)
   if value is not None:
    from decimal import Decimal
    v=Decimal(str(value))
    if not v.is_finite() or v<0 or v!=v.to_integral_value():errors.append('invalid retry budget')
 return errors

def inspect_strategy(g,strategy):
 steps=set(g.objects(strategy,S.hasStep));pending=set(steps);layers=[]
 while pending:
  ready=sorted(n for n in pending if not any((other,S.precedesStep,n) in g for other in pending))
  if not ready:raise ValueError('Cyclic step order')
  layers.append([str(n) for n in ready]);pending-=set(ready)
 return {'strategy':str(strategy),'precedenceLayers':layers,'parallelExecutionAuthorized':False,'providerMatchingPerformed':False,'executed':False}
