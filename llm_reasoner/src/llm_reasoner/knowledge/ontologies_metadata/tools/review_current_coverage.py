"""Current-fixture coverage regressions; descriptive retrieval, never world truth.

Run with requirements.txt installed. Historical V7 checks remain unchanged.
The two new queries cover action descendants and their direct/profile requirements;
they intentionally do not replace every aspect-specific dependency query.
"""
import hashlib
import json
from pathlib import Path
from rdflib import Graph, Namespace, RDF, RDFS, OWL, URIRef, Literal, XSD
from ontology_imports import ROOT, closure
from aspect_runtime import A

B = 'http://aicor.knowledge/ontologies/'
AR = Namespace(B + 'action-recipes.owl#')
CS = Namespace(B + 'capabilities-strategies.owl#')
C = Namespace(B + 'conditions-outcomes.owl#')
E = Namespace(B + 'recipe-examples.owl#')
S = Namespace(B + 'specialized-tool-examples.owl#')


def retrieve(graph, name, record):
    if (record, RDF.type, A.KnowledgeRecord) not in graph:
        raise ValueError('An explicit existing KnowledgeRecord is required')
    if len(set(graph.objects(record, A.expressesDescription))) != 1:
        raise ValueError('Exactly one root description is required')
    root = next(graph.objects(record, A.expressesDescription))
    return list(graph.query((ROOT / 'queries' / (name + '.rq')).read_text(),
                            initBindings={'record': record, 'root': root}))


def main():
    schema, imports = closure(B + 'all-action-knowledge.owl')
    schema -= Graph().parse(ROOT / 'DUL.owl')
    data = Graph()
    for p in sorted((ROOT / 'examples').glob('*.ttl')):
        data.parse(p)
    # These queries never inspect reified claims/statements. Keep their types
    # and links in `data` for the separate provenance audit, and query the
    # descriptive view to avoid irrelevant reification joins in RDFLib.
    metadata = set(data.subjects(RDF.type, A.KnowledgeClaim)) | set(data.subjects(RDF.type, A.Statement))
    g = Graph() + schema
    for s, p, o in data:
        if s not in metadata and p != A.hasClaim:
            g.add((s, p, o))
    checks, snapshots = [], []
    def check(name, result):
        checks.append({'case': name, 'passed': bool(result)})
    reviewed_properties = {A.requiresCapability, A.hasCapabilityRequirement, CS.requiresCapability,
        URIRef(B + 'entities-materials-affordances.owl#requiresCapability'),
        URIRef(B + 'monitoring-requirements.owl#requiresCapability'),
        URIRef(B + 'sensorimotor-events.owl#observationRequirement'),
        URIRef(B + 'sensorimotor-events.owl#requiresCapability'),
        URIRef(B + 'failure-diagnosis.owl#requiresCapability')}
    declared_properties = {p for p in schema.subjects(RDF.type, OWL.ObjectProperty)
                           if A.requiresCapability in set(schema.transitive_objects(p, RDFS.subPropertyOf))}
    check('reviewed-requirement-vocabulary-is-current', declared_properties == reviewed_properties)
    for ns, names in [(E, ['pickCarryPlace', 'routeOrientedPickCarryPlace', 'cuttingCycle', 'wipingCycle']),
                      (S, ['cutting', 'wiping', 'pressing', 'insertion'])]:
        for name in names:
            record, root = ns[name + 'Record'], ns[name]
            caps = retrieve(g, 'current-action-capabilities', record)
            conditions = retrieve(g, 'current-action-conditions', record)
            check(name + ':selected-root-only', bool(conditions) and all(r.root == root for r in conditions + caps))
            claims = set(data.objects(record, A.hasClaim))
            statuses = {o for claim in claims for o in data.objects(claim, A.hasStatus)}
            kinds = {o for claim in claims for ev in data.objects(claim, A.hasEvidence) for o in data.objects(ev, A.evidenceKind)}
            check(name + ':proposed-only', statuses == {A.Proposed})
            check(name + ':synthetic-evidence-only', kinds == {A.CuratedExample})
            # Same bounded local-namespace closure as fixture authorship. This is
            # provenance coverage, not a claim that reachable statements are true.
            seen, todo = set(), [root]
            while todo:
                node = todo.pop()
                if node in seen:
                    continue
                seen.add(node)
                todo.extend(o for o in data.objects(node) if isinstance(o, URIRef) and str(o).startswith(str(ns)))
            descriptive = {(s, p, o) for s in seen for p, o in data.predicate_objects(s) if p not in [RDF.type, RDFS.label, RDFS.comment]}
            claimed = set()
            for claim in claims:
                for st in data.objects(claim, A.hasStatement):
                    subject = data.value(st, A.statementSubject)
                    pred = data.value(st, A.statementPredicateIRI)
                    objects = list(data.objects(st, A.statementObject)) + list(data.objects(st, A.statementLiteral))
                    if subject and pred and len(objects) == 1:
                        claimed.add((subject, URIRef(str(pred)), objects[0]))
            missing = descriptive - claimed
            check(name + ':reachable-exact-statement-coverage', not missing)
            if ns == E:
                expected = 6 if name in ['cuttingCycle', 'wipingCycle'] else 3
                children = {r.action for r in conditions if r.action != root}
                check(name + ':nested-stage-descriptions-visible', len(children) >= expected)
                check(name + ':root-contract-not-mistaken-for-expression',
                      any(r.action == root and r.relation == A.hasSuccessCriterion and
                          str(r.language) == 'ordered-recipe-contract-v1' and r.expression is None for r in conditions))
                check(name + ':nested-requirements-visible', any(r.action != root for r in caps))
            snapshots.append({'record': str(record), 'root': str(root), 'claims': len(claims),
                'descriptive_links_in_local_namespace_closure': len(descriptive),
                'links_missing_exact_statement_claims': len(missing),
                'condition_rows': len(conditions), 'action_profile_capability_rows': len(caps)})
            print(json.dumps({'reviewed_record': str(record), 'condition_rows': len(conditions), 'capability_rows': len(caps)}), flush=True)
    # Unrelated records and unrelated requirement owners must never leak.
    base = {n: {tuple(r) for r in retrieve(g, n, E.pickCarryPlaceRecord)}
            for n in ['current-action-capabilities', 'current-action-conditions']}
    other = Graph() + g
    for t in [(E.foreignRecord, RDF.type, A.KnowledgeRecord),
              (E.foreignRecord, A.expressesDescription, E.foreignRoot),
              (E.foreignRoot, CS.hasCapabilityProfile, E.foreignProfile),
              (E.foreignProfile, CS.requiresCapability, E.foreignRequirement),
              (E.foreignRoot, A.hasSuccessCriterion, E.foreignCondition),
              (E.foreignCondition, A.lexicalExpression, Literal('foreign', datatype=XSD.string))]:
        other.add(t)
    for name in base:
        check(name + ':no-unrelated-owner-leakage', {tuple(r) for r in retrieve(other, name, E.pickCarryPlaceRecord)} == base[name])
        try:
            retrieve(other, name, E.missingRecord)
        except ValueError:
            rejected = True
        else:
            rejected = False
        check(name + ':unknown-record-rejected', rejected)
    other.add((E.pickCarryPlace, A.hasSubActionDescription, E.pickCarryPlace))
    check('cycle-does-not-duplicate-capabilities', {tuple(r) for r in retrieve(other, 'current-action-capabilities', E.pickCarryPlaceRecord)} == base['current-action-capabilities'])
    other.add((E.pickCarryPlaceRecord, A.expressesDescription, E.foreignRoot))
    try:
        retrieve(other, 'current-action-conditions', E.pickCarryPlaceRecord)
    except ValueError:
        rejected = True
    else:
        rejected = False
    check('ambiguous-root-rejected', rejected)
    D = ROOT / 'review/coverage-0.21.0'
    rows = json.loads((D / 'coraplex-action-coverage.json').read_text())
    check('pinned-static-action-count', sum(r['kind'] == 'actions' for r in rows) == 28)
    check('pinned-static-motion-count', sum(r['kind'] == 'motions' for r in rows) == 13)
    check('mixing-gap-explicit', next(r for r in rows if r['class'] == 'MixingAction')['coverage_status'] == 'specific-refinement-gap')
    check('commented-search-not-counted', not any(r['class'] == 'SearchAction' for r in rows))
    check('30-local-question-rows', len(json.loads((D / 'local-competency-coverage.json').read_text())) == 30)
    check('65-distinct-scientific-questions', len({r['id'] for r in json.loads((D / 'repository-scientific-competencies.json').read_text())}) == 65)
    check('all-40-modules-reviewed', len(json.loads((D / 'module-consolidation.json').read_text())) == 40)
    check('generic-imports-exclude-optional-cram-aicor', B + 'cram-action-bridge.owl' not in imports and 'urn:gpt-cram-ontologies:aicor-unified' not in imports)
    report = {'release': '0.21.0', 'checks': checks, 'passed': sum(c['passed'] for c in checks),
        'query_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted((ROOT / 'queries').glob('current-action-*.rq'))},
        'snapshot_audits': snapshots, 'scope': 'Eight current synthetic tool/recipe records; descriptive queries and coverage artifact checks. Not 65 automated scientific answers, simulation, provider matching or robot execution.'}
    (ROOT / 'reports/current-coverage-review.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'checks': len(checks), 'passed': report['passed'], 'failures': [c for c in checks if not c['passed']]}))
    assert all(c['passed'] for c in checks)


if __name__ == '__main__':
    main()
