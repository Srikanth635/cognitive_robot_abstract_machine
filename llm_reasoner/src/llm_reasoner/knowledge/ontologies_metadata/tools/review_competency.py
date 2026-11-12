"""Reproducible ontology competency review; descriptive queries, not projections.

Run from any directory with the package dependencies installed. Results retain RDF
term identity, language and datatype. Tests use immutable V7 example records.
"""
import argparse
import hashlib
import json
from pathlib import Path
from rdflib import Graph, Namespace, RDF, RDFS, OWL, URIRef, Literal, XSD
from ontology_imports import ROOT, closure
from aspect_runtime import A, C, P, evaluate
from temporal_runtime import dependencies, assess_outgoing
from control_runtime import inspect_controller

B = 'http://aicor.knowledge/ontologies/'
E = Namespace(B + 'action-examples.owl#')
CS = Namespace(B + 'capabilities-strategies.owl#')
PM = Namespace(B + 'physical-models.owl#')
FD = Namespace(B + 'failure-diagnosis.owl#')
CATALOG = json.loads((ROOT / 'review/competency-questions.json').read_text())


def load_graph():
    schema, imports = closure(B + 'all-action-aspects.owl')
    schema -= Graph().parse(ROOT / 'DUL.owl')
    data = Graph()
    for path in sorted((ROOT / 'examples').glob('*.ttl')):
        data.parse(path)
    return schema + data, data, imports


def query(g, identifier, record):
    if (record, RDF.type, A.KnowledgeRecord) not in g:
        raise ValueError('An explicit, existing KnowledgeRecord is required')
    if len(list(g.objects(record, A.expressesDescription))) != 1:
        raise ValueError('Record must express exactly one description')
    cq = next(x for x in CATALOG if x['id'] == identifier)
    return [row.asdict() for row in g.query((ROOT / cq['query']).read_text(), initBindings={'record': record})]


def term(value):
    return {'value': str(value), 'type': 'literal' if isinstance(value, Literal) else 'iri',
            **({'datatype': str(value.datatype)} if isinstance(value, Literal) and value.datatype else {}),
            **({'language': value.language} if isinstance(value, Literal) and value.language else {})}


def serialize(rows):
    return sorted([{str(k): term(v) for k, v in row.items()} for row in rows], key=lambda row: json.dumps(row, sort_keys=True))


def review():
    g, data, imports = load_graph()
    tests, answers = [], {}
    def check(name, ok):
        tests.append({'case': name, 'passed': bool(ok)})
    def values(rows, key):
        return {r[key] for r in rows if key in r}
    # Expected inventories are pinned to the reviewed fixtures, not learned from query output.
    counts = {
        'pour': [3,4,10,4,6,14,1,7,0,2,3,12,2,1,1,0,1,5,1,1,1,31,37,7],
        'grasp': [2,4,6,0,7,5,1,3,0,2,3,11,2,1,1,0,1,2,1,1,1,31,33,2],
    }
    for name in ['pour', 'grasp']:
        record = E[name + 'RecordV7']; action = E[name + 'ActionV7']; record_results = {}
        for index, cq in enumerate(CATALOG):
            rows = query(g, cq['id'], record)
            record_results[cq['id']] = rows
            check(f'{name}:{cq["id"]}:expected-inventory', len(rows) == counts[name][index])
        answers[str(record)] = {identifier: serialize(rows) for identifier, rows in record_results.items()}
        check(name + ':role-entities', values(record_results['CQ01'], 'entity') == (
            {E.sourceContainer,E.receivingContainer,E.transferableSubstance} if name == 'pour'
            else {E.graspableObject,E.gripperSpecification}))
        check(name + ':declared-condition-kinds', values(record_results['CQ02'], 'relation') == {
            A.hasGoal,A.hasPrecondition,A.hasSuccessCriterion,A.hasFailureCriterion})
        check(name + ':only-owned-transitions', values(record_results['CQ05'], 'transition') == set(data.objects(E[name+'PhaseStructure'], Namespace(B+'phases-transitions.owl#').hasTransition)))
        check(name + ':capability-coverage', values(record_results['CQ12'], 'requirement') == {
            E[name+'ManipulationCapability'], E[name+'VerificationCapability'],
            E.quantityEstimationCapability if name == 'pour' else E.relationAssessmentCapability})
        check(name + ':scoped-diagnosis', values(record_results['CQ14'], 'cause') == {E[name+'SuspectedCause']})
        check(name + ':exact-record-claims', values(record_results['CQ22'], 'claim') == set(data.objects(record,A.hasClaim)))
        check(name + ':proposals-remain-proposals', values(record_results['CQ22'], 'status') == {A.Proposed})
        check(name + ':synthetic-evidence-only', values(record_results['CQ22'], 'kind') == {A.CuratedExample})
        check(name + ':explicit-unknown-force-strength', values(record_results['CQ17'], 'strength') == {Namespace(B+'force-dynamics.owl#').UnknownStrength})
        check(name + ':unresolved-retry-budget', not values(record_results['CQ15'], 'value'))
        check(name + ':unresolved-controller', bool(inspect_controller(data,E[name+'Controller'])['unresolved']))
        check(name + ':no-foreign-root-in-provenance-audit', len(record_results['CQ23']) == len([t for t in data.triples((action,None,None)) if t[1] != RDF.type]))
    # Contamination tests: evidence for a shared model is NOT evidence included by this record.
    other = Graph() + g
    for triple in [
        (E.reviewForeignRecord,RDF.type,A.KnowledgeRecord),
        (E.reviewForeignRecord,A.expressesDescription,E.reviewForeignDescription),
        (E.reviewForeignRecord,A.hasClaim,E.reviewForeignClaim),
        (E.reviewForeignClaim,A.hasStatus,A.Supported),
        (E.reviewForeignClaim,A.hasStatement,E.reviewForeignStatement),
        (E.reviewForeignClaim,A.hasEvidence,E.reviewForeignValidation),
        (E.reviewForeignStatement,A.statementSubject,E.pourActionV7),
        (E.reviewForeignStatement,A.statementPredicateIRI,Literal(str(A.hasActionKind),datatype=XSD.anyURI)),
        (E.reviewForeignStatement,A.statementObject,E.Pouring),
        (E.reviewForeignValidation,PM.validatesModel,E.pourPhysicalModel),
        (E.reviewForeignValidation,PM.validationArtifactIRI,Literal('urn:test:foreign-run',datatype=XSD.anyURI)),
        (E.reviewForeignValidation,PM.validationScope,E.pourModelScope),
        (E.reviewForeignDescription,CS.hasCapabilityProfile,E.reviewForeignProfile),
        (E.reviewForeignProfile,CS.requiresCapability,E.reviewForeignRequirement),
    ]: other.add(triple)
    for cq in ['CQ09','CQ12','CQ22','CQ23']:
        check('foreign-record-does-not-leak:'+cq, serialize(query(other,cq,E.pourRecordV7)) == answers[str(E.pourRecordV7)][cq])
    other.add((E.pourRecordV7,A.hasClaim,E.reviewForeignClaim))
    check('evidence-visible-only-after-explicit-membership', len(query(other,'CQ09',E.pourRecordV7)) == 1)
    check('exact-provenance-match-reduces-one-missing-link', len(query(other,'CQ23',E.pourRecordV7)) == 36)
    check('historical-frame-not-returned-as-current', len(query(g,'CQ16',E.pourRecordV6)) > 0 and not query(g,'CQ16',E.pourRecordV7))
    old = query(g,'CQ19',E.pourRecordV6)
    check('later-entity-facet-not-attached-to-old-record', old == [])
    try:
        query(g,'CQ12',E.noSuchRecord)
        check('unknown-record-rejected',False)
    except ValueError: check('unknown-record-rejected',True)
    # Requirement satisfaction is deliberately outside the new shared hierarchy.
    check('fulfillment-is-not-requirement', not list(g.query('SELECT ?x WHERE {<'+str(CS.fulfillsRequirement)+'> <'+str(RDFS.subPropertyOf)+'>+ <'+str(A.requiresCapability)+'> BIND(1 AS ?x)}')))
    # RDFS-only regression: specializing requirements must not classify profiles as ActionDescription.
    from owlrl import DeductiveClosure,RDFS_Semantics
    small = Graph().parse(ROOT/'core/action-description-core.ttl') + Graph().parse(ROOT/'aspects/planning/capabilities-strategies.ttl')
    small.add((E.reviewProfile,CS.requiresCapability,E.reviewRequirement))
    DeductiveClosure(RDFS_Semantics).expand(small)
    check('specialized-requirement-entails-generic-requirement',(E.reviewProfile,A.requiresCapability,E.reviewRequirement) in small)
    check('profile-not-mistyped-as-action',(E.reviewProfile,RDF.type,A.ActionDescription) not in small)
    # CQ25: bounded three-valued evaluation; these synthetic numeric inputs are test-only.
    check('CQ25:absent-quantities-stay-unknown',evaluate(data,E.pourSuccessExpression) is None)
    quantities={(C.PostState,E.deliveredAmount):('0.3','urn:test:volume'),(C.TaskContext,E.targetAmount):('0.3','urn:test:volume'),(C.PostState,E.spillAmount):('0.0','urn:test:volume'),(C.TaskContext,E.spillTolerance):('0.01','urn:test:volume')}
    check('CQ25:injected-criteria-satisfied',evaluate(data,E.pourSuccessExpression,quantities=quantities) is True)
    quantities[(C.PostState,E.spillAmount)]=('0.02','urn:test:volume')
    check('CQ25:excess-spill-fails',evaluate(data,E.pourSuccessExpression,quantities=quantities) is False)
    quantities[(C.PostState,E.spillAmount)]=('0.0','urn:test:incompatible-unit')
    check('CQ25:unit-mismatch-stays-unknown',evaluate(data,E.pourSuccessExpression,quantities=quantities) is None)
    check('CQ25:unknown-guards-do-not-select-transition',assess_outgoing(data,E.pourPhaseStructure,E.pourTransfer)['selected_transition'] is None)
    # Gap checks deliberately pin the LIMITATIONS, not assert stronger capabilities.
    leaves=dependencies(data,E.pourSuccessExpression)
    check('CQ26:destination-check-currently-missing',not any(k=='relation' for k,_,_ in leaves))
    check('CQ27:diagnostic-cause-not-linked-to-causal-antecedent',data.value(E.pourDiagnosisHypothesis,FD.candidateCause) != data.value(E.pourCausalLink,Namespace(B+'causal-mechanisms.owl#').antecedent))
    check('CQ30:generic-assembly-excludes-cram',B+'cram-action-bridge.owl' not in imports and 'urn:gpt-cram-ontologies:aicor-unified' not in imports)
    check('CQ30:no-example-world-binding',not list(data.triples((None,A.hasBinding,None))))
    report={'release':'0.10.0','scope':'24 descriptive SPARQL questions, two synthetic V7 records, plus explicit boundary checks; not scientific validation or world truth',
            'queries':len(CATALOG),'checks':len(tests),'passed':sum(t['passed'] for t in tests),'tests':tests,
            'answers':answers,
            'input_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.rglob('*.ttl'))},
            'query_sha256':{q['query']:hashlib.sha256((ROOT/q['query']).read_bytes()).hexdigest() for q in CATALOG}}
    (ROOT/'reports/competency-review.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['queries','checks','passed']}))
    failures=[t for t in tests if not t['passed']]
    if failures: raise AssertionError(failures)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record',help='Explicit KnowledgeRecord IRI for one descriptive query')
    parser.add_argument('--question',choices=[q['id'] for q in CATALOG])
    args=parser.parse_args()
    if args.record or args.question:
        if not (args.record and args.question): parser.error('--record and --question must be supplied together')
        graph,_,_=load_graph()
        print(json.dumps(serialize(query(graph,args.question,URIRef(args.record))),indent=2))
    else:review()
