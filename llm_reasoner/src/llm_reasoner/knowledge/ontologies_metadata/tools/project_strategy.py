from pathlib import Path
import json
from rdflib import Graph,Namespace
from strategy_runtime import A,S,F,inspect_strategy
from ontology_imports import ROOT
E=Namespace('http://aicor.knowledge/ontologies/action-examples.owl#');g=Graph()
for n in ['stereotypical-actions','aspect-actions','phase-actions','causal-physical-actions','control-actions','strategy-failure-actions']:g+=Graph().parse(ROOT/'examples'/(n+'.ttl'))
def val(n,p):v=g.value(n,p);return str(v) if v is not None else None
def refs(n,p):return sorted(str(v) for v in g.objects(n,p))
views=[]
for record in [E.pourRecordV6,E.graspRecordV6]:
 action=g.value(record,A.expressesDescription);out={'record':str(record),'description':str(action),'requiredCapabilities':[],'strategies':[],'failures':[],'hypotheses':[],'recoveries':[],'newClaims':[]}
 for profile in g.objects(action,S.hasCapabilityProfile):out['requiredCapabilities']+=refs(profile,S.requiresCapability)
 for strategy in g.objects(action,S.hasStrategy):
  info=inspect_strategy(g,strategy);info['processes']=[{'id':str(proc),'interfaceKey':val(proc,S.interfaceKey),'requirements':refs(proc,S.fulfillsRequirement)} for step in g.objects(strategy,S.hasStep) for proc in g.objects(step,S.requiresProcess)];out['strategies'].append(info)
 for profile in g.objects(action,F.hasFailureProfile):
  for failure in g.objects(profile,F.hasFailure):out['failures'].append({'id':str(failure),'kind':val(failure,F.failureKind),'criteria':refs(failure,F.detectedBy)})
  for h in g.objects(profile,F.hasHypothesis):out['hypotheses'].append({'id':str(h),'candidateCause':val(h,F.candidateCause),'explains':refs(h,F.explainsFailure),'support':refs(h,F.supportingEvidence),'refutation':refs(h,F.refutingEvidence)})
  for recovery in g.objects(profile,F.hasRecoveryOption):
   strategy=g.value(recovery,F.usesStrategy);out['recoveries'].append({'id':str(recovery),'guard':val(recovery,F.guardedBy),'intendedCondition':val(recovery,F.restoresCondition),'budgetParameter':val(recovery,F.retryBudget),'budgetValue':val(g.value(recovery,F.retryBudget),A.numericValue),'strategy':inspect_strategy(g,strategy) if strategy else None})
 for c in g.objects(record,A.hasClaim):
  if 'StrategyClaim' in str(c):out['newClaims'].append({'id':str(c),'status':val(c,A.hasStatus),'evidence':refs(c,A.hasEvidence)})
 views.append(out)
assert len(views)==2 and sum(len(v['newClaims']) for v in views)==8
(ROOT/'examples/execution-recovery-projections.json').write_text(json.dumps(views,indent=2)+'\n');print('Two strategy/recovery views with eight proposed claims; no provider matching or execution.')
