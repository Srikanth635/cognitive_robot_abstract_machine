"""Positive, negative and unknown-state fixtures; no physical measurements."""
import json
from rdflib import Graph, Namespace, RDF, RDFS, URIRef, Literal, XSD
from pyshacl import validate
from ontology_imports import ROOT, closure
from aspect_runtime import A, C, P, one, structural_errors
from temporal_runtime import T, M, temporal_errors, assess_outgoing, assess_monitor
from articulated_mechanisms import N, RM, OM, profile, mechanism_errors, assess, expression_signature, CONFIGURATION_TYPES

B = 'http://aicor.knowledge/ontologies/'
E = Namespace(B+'articulated-examples.owl#')
schema, imports = closure(B+'articulated-mechanisms.owl')
schema -= Graph().parse(ROOT/'DUL.owl')
data = Graph().parse(ROOT/'examples/articulated-mechanism-profiles.ttl')
g = data + schema
shapes = Graph()
for path in [ROOT/'core/action-core-shapes.ttl', ROOT/'core/knowledge-context-shapes.ttl',
             ROOT/'aspects/validation/aspect-shapes.ttl', ROOT/'aspects/validation/temporal-shapes.ttl',
             ROOT/'domains/robot-motion-shapes.ttl', ROOT/'domains/articulated-mechanisms-shapes.ttl']:
    shapes.parse(path)
checks = []
def check(name, passed):
    checks.append({'case': name, 'passed': bool(passed)})
def rejects(name, call):
    try:
        call()
        check(name, False)
    except ValueError:
        check(name, True)
def shape_valid(graph):
    ok, _, text = validate(graph, shacl_graph=shapes, ont_graph=schema, inference='rdfs', advanced=True)
    if graph is data:
        (ROOT/'reports/articulated-mechanisms-shacl.txt').write_text(text)
    return bool(ok)
check('eight-profile-shapes', shape_valid(data))
check('relation-expression-integrity', not structural_errors(g))
check('phase-monitor-integrity', not temporal_errors(g))
check('mechanism-profile-integrity', not mechanism_errors(g))
names = ['revoluteConfiguration','prismaticConfiguration','revoluteOpening','prismaticOpening',
         'revoluteClosing','prismaticClosing','releaseLatch','secureLatch']
def supplied(action, prop=A.hasSuccessCriterion):
    atoms, pairs = expression_signature(g, one(g, action, prop))
    facts = {(scope, rel, args): not neg for scope, rel, args, neg in atoms}
    quantities = {}
    for op, scope, left, rscope, right in pairs:
        quantities[(scope,left)] = ('0.01',str(one(g,left,A.unitIRI)))
        quantities[(rscope,right)] = ('0.02',str(one(g,right,A.unitIRI)))
    return facts, quantities
def inspect(graph, name, facts=None, quantities=None):
    return assess(graph, E[name], E[name+'Context'], facts, quantities)
for name in names:
    action = E[name]
    facts, quantities = supplied(action)
    check(name+':unknown-without-input', inspect(g,name)['success'] is None)
    result = inspect(g,name,facts,quantities)
    check(name+':supplied-success', result['success'] is True)
    check(name+':post-success-not-precondition', result['precondition'] is None)
    check(name+':no-fault-from-missing-success', inspect(g,name)['fault'] is None)
    check(name+':never-executes-or-accepts', not result['executed'] and not result['executionReadinessAssessed'] and result['acceptedObservations']==0)
    if facts:
        key = next(iter(facts))
        missing = dict(facts); missing.pop(key)
        wrong = dict(facts); wrong[key] = False
        check(name+':missing-required-atom-unknown', inspect(g,name,missing,quantities)['success'] is None)
        check(name+':false-required-atom-fails', inspect(g,name,wrong,quantities)['success'] is False)
    else:
        missing = dict(quantities); missing.pop(next(iter(missing)))
        check(name+':missing-metric-unknown', inspect(g,name,facts,missing)['success'] is None)
    if quantities:
        error = one(g,action,N.errorParameter); bad = dict(quantities); bad[(C.PostState,error)] = ('0.03',str(one(g,error,A.unitIRI)))
        check(name+':excess-error-fails', inspect(g,name,facts,bad)['success'] is False)
        bad[(C.PostState,error)] = ('-0.1',str(one(g,error,A.unitIRI)))
        rejects(name+':negative-error-rejected',lambda:inspect(g,name,facts,bad))
        bad[(C.PostState,error)] = ('NaN',str(one(g,error,A.unitIRI)))
        rejects(name+':nonfinite-error-rejected',lambda:inspect(g,name,facts,bad))
        bad[(C.PostState,error)] = ('0.01','http://qudt.org/vocab/unit/SEC')
        rejects(name+':wrong-input-unit-rejected',lambda:inspect(g,name,facts,bad))
    rejects(name+':foreign-context',lambda:assess(g,action,E.foreignContext,facts,quantities))
    pf,pq = supplied(action,A.hasPrecondition)
    check(name+':explicit-preconditions-satisfy',inspect(g,name,pf,pq)['precondition'] is True)
    ff,fq = supplied(action,A.hasFailureCriterion)
    out = inspect(g,name,{**facts,**ff},quantities)
    check(name+':concurrent-success-and-fault-exposed',out['successAndFaultReported'])
    check(name+':unknown-monitor-requests-evidence',assess_monitor(g,E[name+'VerificationMonitor'])['declared_response']==str(M.RequestEvidence))

for name in ['revoluteClosing','prismaticClosing']:
    facts,quantities=supplied(E[name])
    check(name+':closure-does-not-require-or-infer-latching',inspect(g,name,facts,quantities)['success'] is True and not any(rel==N.LatchSecured for _,rel,_ in facts))
    pre,_=supplied(E[name],A.hasPrecondition)
    check(name+':latch-presence-not-automatic-release-precondition',not any(rel==N.LatchReleased for _,rel,_ in pre))
    guard=one(g,E[name+'Transition1'],T.guardCondition)
    atoms,_=expression_signature(g,guard)
    check(name+':declared-nonblocking-latch-policy',not any(rel==N.LatchReleased for _,rel,_,_ in atoms))
facts,quantities=supplied(E.releaseLatch)
check('latch-release-does-not-establish-opening',not any(rel==N.OpenConfiguration for _,rel,_ in facts))
facts,quantities=supplied(E.secureLatch)
key=(C.PostState,N.ClosedConfiguration,(E.secureLatchAssembly,E.secureLatchMovingPart))
facts.pop(key)
check('securing-without-closure-remains-unknown',inspect(g,'secureLatch',facts,quantities)['success'] is None)
facts,quantities=supplied(E.revoluteOpening)
foreign={(scope,rel,tuple(E.foreignPart if x==E.revoluteOpeningMovingPart else x for x in args)):v for (scope,rel,args),v in facts.items()}
check('another-part-open-does-not-prove-target-open',inspect(g,'revoluteOpening',foreign,quantities)['success'] is None)
facts[(C.PostState,N.ClosedConfiguration,(E.revoluteOpeningAssembly,E.revoluteOpeningMovingPart))]=True
check('incompatible-open-closed-reports-exposed',bool(inspect(g,'revoluteOpening',facts,quantities)['incompatibleStateReports']))
facts={(C.CurrentState,N.LatchReleased,(E.revoluteOpeningLatch,)):True,(C.CurrentState,N.LatchSecured,(E.revoluteOpeningLatch,)):True}
check('incompatible-latch-reports-exposed',bool(inspect(g,'revoluteOpening',facts)['incompatibleStateReports']))
# The entry guard needs actual current actuation engagement, not just a held handle.
structure=E.revoluteOpeningPhases
out=assess_outgoing(g,structure,E.revoluteOpeningEngage)
check('unknown-engagement-does-not-select-transition',out['selected_transition'] is None and out['assessment']=='incomplete')
guard=one(g,E.revoluteOpeningTransition1,T.guardCondition)
atoms,_=expression_signature(g,guard)
facts={(sc,rel,args):not neg for sc,rel,args,neg in atoms}
facts.pop((C.CurrentState,N.LatchReleased,(E.revoluteOpeningLatch,)))
facts[(C.CurrentState,N.LatchSecured,(E.revoluteOpeningLatch,))]=True
out=assess_outgoing(g,structure,E.revoluteOpeningEngage,facts)
check('secured-is-not-released',next(r for r in out['transitions'] if r['transition']==str(E.revoluteOpeningTransition1))['guard_value'] is None)
facts,_=supplied(E.revoluteOpening,A.hasPrecondition)
out=assess_outgoing(g,structure,E.revoluteOpeningEngage,facts)
check('prestate-release-is-not-current-release',next(r for r in out['transitions'] if r['transition']==str(E.revoluteOpeningTransition1))['guard_value'] is None)
facts={(sc,rel,args):not neg for sc,rel,args,neg in atoms}
faults,_=supplied(E.revoluteOpening,A.hasFailureCriterion);facts.update(faults)
out=assess_outgoing(g,structure,E.revoluteOpeningEngage,facts)
check('engagement-and-fault-ambiguity-not-arbitrated',out['assessment']=='ambiguous' and out['selected_transition'] is None)

def mutation(name,subject,predicate,value):
    graph=Graph()+g;graph.set((subject,predicate,value));check('reject-'+name,bool(mechanism_errors(graph)))
for row in [
 ('wrong-action-kind',E.revoluteOpening,A.hasActionKind,N.ClosingMechanism),
 ('collapsed-fixed-moving',E.revoluteOpeningMechanism,N.fixedPart,E.revoluteOpeningMovingPart),
 ('foreign-handle',E.revoluteOpeningHandle,N.handleOf,E.prismaticOpeningMovingPart),
 ('foreign-latch',E.revoluteOpeningLatch,N.latchOf,E.prismaticOpeningMechanism),
 ('foreign-axis-frame',E.revoluteOpeningAxis,N.axisFrame,E.prismaticOpeningFrame),
 ('unsupported-joint-kind',E.revoluteOpeningJoint,N.jointKind,N.Continuous),
 ('wrong-coordinate-kind',E.revoluteOpeningCoordinate,RM.coordinateKind,Literal('linear',datatype=XSD.string)),
 ('wrong-unit',E.revoluteOpeningTolerance,A.unitIRI,Literal('http://qudt.org/vocab/unit/M',datatype=XSD.anyURI)),
 ('foreign-metric-coordinate',E.revoluteOpeningError,N.errorCoordinate,E.prismaticOpeningCoordinate),
 ('foreign-metric-target',E.revoluteOpeningError,N.errorTarget,E.prismaticOpeningOpenCoordinate),
 ('aliased-error-tolerance',E.revoluteOpening,N.toleranceParameter,E.revoluteOpeningError),
 ('wrong-opening-target',E.revoluteOpening,N.targetParameter,E.revoluteOpeningClosedCoordinate),
 ('bad-opening-direction',E.revoluteOpeningJoint,N.openingDirection,Literal(0,datatype=XSD.integer)),
 ('ambiguous-latch-policy',E.revoluteOpening,N.latchReleaseRequired,Literal('unknown',datatype=XSD.string)),
 ('release-required-closing-without-release-guard',E.revoluteClosing,N.latchReleaseRequired,Literal(True,datatype=XSD.boolean)),
 ('negative-tolerance',E.revoluteOpeningTolerance,A.numericValue,Literal('-0.1',datatype=XSD.decimal)),
 ('missing-success-condition',E.revoluteOpening,A.hasSuccessCriterion,E.prismaticOpeningSuccess),
 ('wrong-precondition-scope',E.revoluteOpeningReleasedBefore,C.inScope,C.PostState),
 ('wrong-completion-scope',E.revoluteOpeningPostErrorWithinErrorRef,C.inScope,C.CurrentState),
 ('wrong-guard',E.revoluteOpeningTransition1,T.guardCondition,E.revoluteOpeningPrecondition),
 ('skip-verification',E.revoluteOpeningTransition2,T.toPhase,E.revoluteOpeningComplete),
 ('wrong-final-criterion',E.revoluteOpeningTransition3,T.guardCondition,E.revoluteOpeningCurrentTarget),
 ('foreign-phase',E.revoluteOpening,N.actuationPhase,E.prismaticOpeningMove),
 ('wrong-move-guard',E.revoluteOpeningTransition2,T.guardCondition,E.revoluteOpeningPrecondition),
]:mutation(*row)
# A known model is only a synthetic in-memory test of bounds and sign conventions.
def known(name):
    h=Graph()+g;direction=int(one(h,E[name+'Joint'],N.openingDirection))
    values={'LowerLimit':'-2','UpperLimit':'2','OpenCoordinate':str(direction),'ClosedCoordinate':'0'}
    for suffix,value in values.items():h.set((E[name+suffix],A.numericValue,Literal(value,datatype=XSD.decimal)))
    for c,value in zip('xyz',['0','0','1']):h.set((E[name+'Axis'+c],A.numericValue,Literal(value,datatype=XSD.decimal)))
    # Negative origin coordinates are legitimate.
    h.set((E[name+'Originx'],A.numericValue,Literal('-0.4',datatype=XSD.decimal)))
    return h
check('negative-opening-coordinate-supported',not mechanism_errors(known('prismaticOpening')))
check('positive-opening-coordinate-supported',not mechanism_errors(known('revoluteOpening')))
for label,suffix,value in [('out-of-range-open','OpenCoordinate','3'),('reversed-limits','LowerLimit','3'),('degenerate-limits','LowerLimit','2'),('reversed-opening-convention','OpenCoordinate','-1'),('zero-axis','Axisz','0'),('nonnormal-axis','Axisz','2')]:
    h=known('revoluteOpening');h.set((E['revoluteOpening'+suffix],A.numericValue,Literal(value,datatype=XSD.decimal)));check('reject-'+label,bool(mechanism_errors(h)))
h=known('revoluteConfiguration');h.set((E.revoluteConfigurationTarget,A.numericValue,Literal('3',datatype=XSD.decimal)));check('reject-out-of-range-request',bool(mechanism_errors(h)))
h=Graph()+g;h.add((E.revoluteOpening,RDF.type,N.ClosingDescription));check('reject-mixed-opening-closing-profile',bool(mechanism_errors(h)))
h=Graph()+g;h.add((E.revoluteOpening,RDF.type,N.ConfigurationChangeDescription));check('materialized-inheritance-is-accepted',not mechanism_errors(h))
h=Graph()+g;h.remove((E.revoluteOpeningMechanism,N.latch,None));check('reject-required-but-unidentified-latch',bool(mechanism_errors(h)))
h=Graph()+g;h.remove((E.revoluteOpeningMove,M.hasMonitoringRequirement,E.revoluteOpeningEngagementMonitor));check('reject-unmonitored-actuation',bool(mechanism_errors(h)))
h=Graph()+g;h.remove((E.revoluteOpeningPhases,T.hasTransition,E.revoluteOpeningMoveFaultTransition));check('reject-missing-fault-exit',bool(mechanism_errors(h)))
h=Graph()+g;child=E.optionalEngagement
for t in [(E.revoluteOpening,N.engagementDescription,child),(child,RDF.type,OM.ObjectManipulationDescription),(child,OM.manipulator,E.revoluteOpeningOperator),(child,OM.object,E.revoluteOpeningHandle)]:h.add(t)
check('optional-engagement-reference-binding',not mechanism_errors(h))
h.set((child,OM.object,E.prismaticOpeningHandle));check('reject-foreign-engagement-handle',bool(mechanism_errors(h)))
# SHACL cardinality checks use small metadata-free descriptive fixtures.
description=Graph()
for s,p,o in data:
    if not any((s,RDF.type,cls) in data for cls in [A.KnowledgeRecord,A.KnowledgeClaim,A.Statement,A.AnnotationRecord]):description.add((s,p,o))
for label,subject,predicate in [('missing-axis',E.revoluteOpeningJoint,N.axis),('missing-context',E.revoluteOpening,N.evaluationContext),('missing-actuation-phase',E.revoluteOpening,N.actuationPhase)]:
    h=Graph()+description;h.remove((subject,predicate,None));check('shape-reject-'+label,not shape_valid(h))
# Exact statement coverage for each new description; historical data stays untouched.
all_reachable=set()
for name in names:
    covered=set()
    for claim in data.objects(E[name+'Record'],A.hasClaim):
        st=one(data,claim,A.hasStatement);objects=list(data.objects(st,A.statementObject))+list(data.objects(st,A.statementLiteral))
        covered.add((one(data,st,A.statementSubject),str(one(data,st,A.statementPredicateIRI)),objects[0]))
    seen=set();todo=[E[name]]
    while todo:
        n=todo.pop()
        if n in seen:continue
        seen.add(n);todo.extend(o for o in data.objects(n) if isinstance(o,URIRef) and str(o).startswith(str(E)))
    expected={(s,str(p),o) for s in seen for p,o in data.predicate_objects(s) if p not in [RDF.type,RDFS.label,RDFS.comment]}
    all_reachable.update(seen)
    check(name+':exact-statement-coverage',bool(expected) and expected==covered)
metadata={s for cls in [A.KnowledgeRecord,A.AnnotationRecord,A.KnowledgeClaim,A.Statement,A.EvidenceRecord]
          for s in data.subjects(RDF.type,cls)}
metadata.update(data.objects(None,A.aboutComponent))
check('no-unrecorded-orphan-description-nodes',not (set(data.subjects())-all_reachable-metadata))
check('all-claims-proposed',all(data.value(c,A.hasStatus)==A.Proposed for c in data.subjects(RDF.type,A.KnowledgeClaim)))
check('no-numeric-physical-defaults',not list(data.triples((None,A.numericValue,None))))
check('no-world-bindings',not list(data.triples((None,A.hasBinding,None))))
check('no-framework-tool-or-transfer-import',not any('aicor-unified' in i or 'cram-action-bridge' in i or 'tool-interaction.owl' in i or 'material-transfer.owl' in i for i in imports))
report={'release':'0.18.0','checks':checks,'imports':imports,'scope':'Eight synthetic mechanism profiles; supplied-state tests do not establish physical validity or execution readiness.'}
(ROOT/'reports/articulated-mechanisms-validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'checks':len(checks),'passed':sum(c['passed'] for c in checks),'failures':[c for c in checks if not c['passed']]}))
assert all(c['passed'] for c in checks)
