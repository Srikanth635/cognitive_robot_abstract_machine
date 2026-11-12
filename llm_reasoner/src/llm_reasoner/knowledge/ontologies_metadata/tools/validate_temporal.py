"""Validate phase topology, monitoring dependencies and guarded preview semantics."""
from pathlib import Path
import json
from rdflib import Graph,Namespace,RDF
from pyshacl import validate
from aspect_runtime import A,P,C,structural_errors
from temporal_runtime import T,M,temporal_errors,assess_outgoing,assess_monitor
R=Path(__file__).resolve().parents[1];E=Namespace('http://aicor.knowledge/ontologies/action-examples.owl#')
SCHEMAS=['core/action-description-core.ttl','aspects/structure/participants-relations.ttl','aspects/structure/conditions-outcomes.ttl','aspects/structure/phases-transitions.ttl','aspects/observation/monitoring-requirements.ttl']
DATA=['examples/stereotypical-actions.ttl','examples/aspect-actions.ttl','examples/phase-actions.ttl']
schema=Graph();data=Graph();shapes=Graph()
for name in SCHEMAS:schema+=Graph().parse(R/name)
for name in DATA:data+=Graph().parse(R/name)
for name in ['core/action-core-shapes.ttl','aspects/validation/aspect-shapes.ttl','aspects/validation/temporal-shapes.ttl']:shapes+=Graph().parse(R/name)
results=[]
def record(name,ok):results.append({'case':name,'passed':bool(ok)})
def check(g):
 valid,_,report=validate(g,shacl_graph=shapes,ont_graph=schema,inference='rdfs',advanced=True)
 issues=structural_errors(g+schema)+temporal_errors(g+schema)
 return bool(valid) and not issues,report,issues
ok,report,issues=check(data);record('valid-pouring-and-grasping-with-recovery-cycle',ok)
(R/'reports/temporal-shacl.txt').write_text(report+'\n'+json.dumps(issues,indent=2))
def duplicate_observation(g):
 old=next(g.objects(E.pourSpillMonitor,M.requiresObservation));new=E.duplicateObservation
 for p,o in g.predicate_objects(old):g.add((new,p,o))
 g.add((E.pourSpillMonitor,M.requiresObservation,new))
mutations={
'foreign-initial-phase':lambda g:g.set((E.pourPhaseStructure,T.initialPhase,E.graspPrepare)),
'foreign-transition-target':lambda g:g.set((E.pourBegin,T.toPhase,E.graspEstablish)),
'outgoing-terminal-transition':lambda g:g.set((E.pourBegin,T.fromPhase,E.pourComplete)),
'unreachable-phase':lambda g:(g.add((E.orphanPhase,RDF.type,A.PhaseSpecification)),g.add((E.pourPhaseStructure,T.hasMemberPhase,E.orphanPhase))),
'missing-transition-target':lambda g:g.remove((E.pourVerified,T.toPhase,None)),
'nonterminal-dead-end':lambda g:g.set((E.pourComplete,RDF.type,A.PhaseSpecification)),
'no-terminal-phases':lambda g:(g.set((E.pourComplete,RDF.type,A.PhaseSpecification)),g.set((E.pourAbort,RDF.type,A.PhaseSpecification))),
'shared-transition-ownership':lambda g:g.add((E.graspPhaseStructure,T.hasTransition,E.pourBegin)),
'missing-guard':lambda g:g.remove((E.pourBegin,T.guardCondition,None)),
'guard-without-expression':lambda g:g.remove((E.pourReadyToVerify,C.hasExpression,None)),
'contradictory-terminal-types':lambda g:g.add((E.pourComplete,RDF.type,T.FailedTerminalPhase)),
'missing-observation':lambda g:g.remove((E.pourSpillMonitor,M.requiresObservation,next(g.objects(E.pourSpillMonitor,M.requiresObservation)))),
'wrong-observation-scope':lambda g:g.set((E.observe_quantity_spillAmount_CurrentState,M.inScope,C.PostState)),
'duplicate-observation':duplicate_observation,
'missing-unknown-response':lambda g:g.remove((E.pourSpillMonitor,M.whenUnknown,None)),
'unknown-treated-as-satisfied':lambda g:g.set((E.pourSpillMonitor,M.whenUnknown,M.ReportSatisfied)),
'wrong-monitor-attachment':lambda g:g.set((E.pourSpillMonitor,M.atPoint,M.TransitionCheck)),
'unattached-monitor':lambda g:g.remove((None,M.hasMonitoringRequirement,E.pourSpillMonitor)),
}
for name,mutate in mutations.items():
 g=Graph()+data;mutate(g);valid,_,_=check(g);record('reject-'+name,not valid)
g=data+schema
# All numeric inputs are synthetic fixtures; no real execution or measurement is represented.
unit='http://qudt.org/vocab/unit/L'
safe={(C.CurrentState,E.deliveredAmount):('0.3',unit),(C.TaskContext,E.targetAmount):('0.2',unit),(C.CurrentState,E.spillAmount):('0.005',unit),(C.TaskContext,E.spillTolerance):('0.01',unit)}
unsafe={**safe,(C.CurrentState,E.spillAmount):('0.02',unit)}
preview=assess_outgoing(g,E.pourPhaseStructure,E.pourTransfer,quantities=safe)
record('one-true-guard-is-not-execution',preview['assessment']=='one_guard_true' and preview['selected_transition'] is None)
record('sufficient-transfer-guard',next(t['guard_value'] for t in preview['transitions'] if t['transition']==str(E.pourAdvanceToVerify)) is True)
preview=assess_outgoing(g,E.pourPhaseStructure,E.pourTransfer,quantities=unsafe)
record('spill-failure-guard',next(t['guard_value'] for t in preview['transitions'] if t['transition']==str(E.pourSpillAbort)) is True)
notready={**safe,(C.CurrentState,E.deliveredAmount):('0.1',unit)}
record('false-guards-do-not-advance',assess_outgoing(g,E.pourPhaseStructure,E.pourTransfer,quantities=notready)['assessment']=='no_guard_true')
record('readiness-false-is-not-unknown',assess_monitor(g,E.pourReadinessMonitor,quantities=notready)['declared_response']==str(M.ReportUnsatisfied))
record('missing-data-does-not-select',assess_outgoing(g,E.pourPhaseStructure,E.pourTransfer)['assessment']=='incomplete')
record('terminal-has-no-next-transition',assess_outgoing(g,E.pourPhaseStructure,E.pourComplete)['assessment']=='terminal')
ambiguous=Graph()+g;ambiguous.set((E.pourSpillAbort,T.guardCondition,E.pourReadyToVerify))
record('multiple-true-guards-ambiguous',assess_outgoing(ambiguous,E.pourPhaseStructure,E.pourTransfer,quantities=safe)['assessment']=='ambiguous')
partial=Graph()+g;partial.set((E.pourSpillAbort,T.guardCondition,E.pourAmountSuccess))
record('true-plus-unknown-is-incomplete',assess_outgoing(partial,E.pourPhaseStructure,E.pourTransfer,quantities=safe)['assessment']=='incomplete')
unknown=assess_monitor(g,E.pourSpillMonitor)
record('unknown-requests-evidence',unknown['condition_value'] is None and unknown['declared_response']==str(M.RequestEvidence))
record('monitor-does-not-execute-response',unknown['response_executed'] is False)
record('spill-violation-requests-suspension',assess_monitor(g,E.pourSpillMonitor,quantities=unsafe)['declared_response']==str(M.RequestSuspension))
record('safe-spill-reports-satisfied',assess_monitor(g,E.pourSpillMonitor,quantities=safe)['declared_response']==str(M.ReportSatisfied))
facts={(C.CurrentState,P.Graspable,(E.graspableObject,)):True}
recovery=assess_outgoing(g,E.graspPhaseStructure,E.graspRecover,facts=facts)
record('recovery-loop-described-not-executed',recovery['assessment']=='one_guard_true' and recovery['selected_transition'] is None)
facts={(C.CurrentState,P.Graspable,(E.graspableObject,)):False}
record('explicit-negative-can-abort-recovery',next(t['guard_value'] for t in assess_outgoing(g,E.graspPhaseStructure,E.graspRecover,facts=facts)['transitions'] if t['transition']==str(E.graspRecoveryFailed)) is True)
(R/'reports/temporal-validation.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps({'checks':len(results),'passed':sum(r['passed'] for r in results),'failures':[r for r in results if not r['passed']]},indent=2))
assert all(r['passed'] for r in results)
