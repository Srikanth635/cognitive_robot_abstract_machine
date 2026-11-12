from pathlib import Path
import json
from rdflib import Graph,Namespace,RDF,Literal,XSD
from pyshacl import validate
from ontology_imports import closure,ROOT
from strategy_runtime import A,S,F,strategy_errors,inspect_strategy
E=Namespace('http://aicor.knowledge/ontologies/action-examples.owl#')
schema,_=closure('http://aicor.knowledge/ontologies/all-action-aspects.owl');schema -= Graph().parse(ROOT/'DUL.owl');data=Graph();shapes=Graph()
for n in ['stereotypical-actions','aspect-actions','phase-actions','causal-physical-actions','control-actions','strategy-failure-actions']:data+=Graph().parse(ROOT/'examples'/(n+'.ttl'))
for p in [ROOT/'core/action-core-shapes.ttl']+list((ROOT/'aspects').rglob('*shapes.ttl')):shapes+=Graph().parse(p)
def check(g):
 ok,_,report=validate(g,shacl_graph=shapes,ont_graph=schema,inference='rdfs',advanced=True);errors=strategy_errors(g);return bool(ok) and not errors,report,errors
results=[]
def record(n,ok):results.append({'case':n,'passed':bool(ok)})
ok,txt,errors=check(data);record('valid-two-strategy-diagnosis-examples',ok);(ROOT/'reports/strategy-shacl.txt').write_text(txt+'\n'+str(errors))
def numeric(g,value):
 g.set((E.pourRetryBudget,A.numericValue,Literal(value,datatype=XSD.decimal)));g.set((E.pourRetryBudget,A.unitIRI,Literal('http://qudt.org/vocab/unit/UNITLESS',datatype=XSD.anyURI)))
mutations={
'cyclic-step-order':lambda g:g.add((E.pourVerifyStep,S.precedesStep,E.pourPerformStep)),
'foreign-step-order':lambda g:g.add((E.pourPerformStep,S.precedesStep,E.graspPerformStep)),
'shared-step-owner':lambda g:g.add((E.graspExecutionStrategy,S.hasStep,E.pourPerformStep)),
'missing-process':lambda g:g.remove((E.pourPerformStep,S.requiresProcess,None)),
'missing-action':lambda g:g.remove((E.pourPerformStep,S.describesAction,None)),
'missing-interface':lambda g:g.remove((E.pourPerformProcess,S.interfaceKey,None)),
'unlisted-capability':lambda g:g.remove((E.pourCapabilityProfile,S.requiresCapability,E.pourManipulationCapability)),
'unsupported-met-assessment':lambda g:g.set((E.pourCapabilityAssessment,S.assessmentOutcome,S.Met)),
'missing-assessment-context':lambda g:g.remove((E.pourCapabilityAssessment,S.assessmentContext,None)),
'foreign-diagnostic-failure':lambda g:g.set((E.pourDiagnosisHypothesis,F.explainsFailure,E.graspFailureDescription)),
'foreign-recovery-failure':lambda g:g.set((E.pourRecoveryOption,F.addressesFailure,E.graspFailureDescription)),
'missing-candidate-cause':lambda g:g.remove((E.pourDiagnosisHypothesis,F.candidateCause,None)),
'missing-detection-criterion':lambda g:g.remove((E.pourFailureDescription,F.detectedBy,None)),
'missing-recovery-guard':lambda g:g.remove((E.pourRecoveryOption,F.guardedBy,None)),
'missing-recovery-goal':lambda g:g.remove((E.pourRecoveryOption,F.restoresCondition,None)),
'negative-retry-budget':lambda g:numeric(g,'-1'),
'fractional-retry-budget':lambda g:numeric(g,'1.5'),
}
for n,mutate in mutations.items():
 g=Graph()+data;mutate(g);ok,_,_=check(g);record('reject-'+n,not ok)
g=Graph()+data;numeric(g,'0');ok,_,_=check(g);record('zero-retries-is-valid-explicit-policy',ok)
g=Graph()+data;g.add((E.pourDiagnosisHypothesis,F.supportingEvidence,E.exampleEvidence));g.add((E.pourDiagnosisHypothesis,F.refutingEvidence,E.exampleEvidence));ok,_,_=check(g);record('conflicting-evidence-retained-without-decision',ok)
view=inspect_strategy(data,E.pourExecutionStrategy);record('precedence-does-not-execute',len(view['precedenceLayers'])==2 and not view['executed'])
record('capability-remains-undetermined',data.value(E.pourCapabilityAssessment,S.assessmentOutcome)==S.Undetermined)
record('claims-remain-proposed',all(data.value(E[p+'StrategyClaim'+str(i)],A.hasStatus)==A.Proposed for p in ['pour','grasp'] for i in range(4)))
(ROOT/'reports/strategy-validation.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps({'checks':len(results),'passed':sum(r['passed'] for r in results),'failures':[r for r in results if not r['passed']]},indent=2));assert all(r['passed'] for r in results)
