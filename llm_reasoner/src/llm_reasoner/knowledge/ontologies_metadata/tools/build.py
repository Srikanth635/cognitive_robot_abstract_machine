"""Rebuild the scoped AICOR comparison and DUL-integration candidate offline.

Run from any directory with Python 3.11+ and rdflib 7.6.0 installed.
This compares OWL syntax structurally, not by parser-generated blank-node IDs.
It does not claim logical equivalence or full OWL consistency.
"""
from pathlib import Path
from functools import lru_cache
from hashlib import sha256
import csv
import json
from rdflib import Graph, Namespace, URIRef, BNode, Literal, RDF, RDFS, OWL

ROOT = Path(__file__).resolve().parents[1]
N = Namespace('http://aicor.knowledge/l2-ontology.owl#')
D = Namespace('http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#')
OLD = URIRef('http://aicor.knowledge/l2-ontology.owl')
NEW = URIRef('urn:gpt-cram-ontologies:aicor-unified')
DUL = URIRef(str(D)[:-1])

def signature_function(graph):
    @lru_cache(None)
    def signature(node):
        if not isinstance(node, BNode):
            return node.n3()
        return '[' + '|'.join(sorted(p.n3() + ' ' + signature(o)
                                    for p, o in graph.predicate_objects(node))) + ']'
    return signature

def normalized(graph):
    """Canonicalize anonymous OWL expressions, never anonymous individuals.

    Inputs in this task use blank nodes only for OWL expressions, lists and axiom
    containers. Fail if an unrecognized anonymous structure appears.
    """
    signature = signature_function(graph)
    structural = {OWL.Restriction, OWL.Class, OWL.AllDisjointClasses,
                  OWL.AllDifferent, OWL.Axiom}
    result = Graph()
    for node in set(graph.all_nodes()):
        if isinstance(node, BNode):
            assert set(graph.objects(node, RDF.type)) & structural or (node, RDF.first, None) in graph, node
    def term(node):
        return BNode('s' + sha256(signature(node).encode()).hexdigest()) if isinstance(node, BNode) else node
    for triple in graph:
        result.add(tuple(term(n) for n in triple))
    return result

def stats(graph):
    return {'triples': len(graph), 'NamedClass': len({s for s in graph.subjects(RDF.type, OWL.Class) if isinstance(s, URIRef)}), **{str(t).split('#')[-1]: len(set(graph.subjects(RDF.type, t)))
            for t in [OWL.Class, OWL.ObjectProperty, OWL.DatatypeProperty,
                      OWL.AnnotationProperty, OWL.NamedIndividual, OWL.Restriction]}}

def serialize(graph, path):
    for prefix, ns in [('aicor', N), ('dul', D), ('owl', OWL), ('rdfs', RDFS)]:
        graph.bind(prefix, ns)
    graph.serialize(path, format='turtle')
    assert set(normalized(Graph().parse(path))) == set(normalized(graph))

def hierarchy_conflicts(graph):
    """Sound named-parent/disjointness screen; deliberately not a DL reasoner."""
    classes = {s for s in graph.subjects(RDF.type, OWL.Class) if isinstance(s, URIRef)}
    parents = {s: {o for o in graph.objects(s, RDFS.subClassOf) if isinstance(o, URIRef)} for s in classes}
    for s, o in graph.subject_objects(OWL.equivalentClass):
        if isinstance(s, URIRef) and isinstance(o, URIRef):
            parents.setdefault(s, set()).add(o)
            parents.setdefault(o, set()).add(s)
    def ancestors(s):
        seen = {s}; todo = [s]
        while todo:
            for p in parents.get(todo.pop(), set()) - seen:
                seen.add(p); todo.append(p)
        return seen
    pairs = {(s, o) for s, o in graph.subject_objects(OWL.disjointWith)
             if isinstance(s, URIRef) and isinstance(o, URIRef)}
    for ax in graph.subjects(RDF.type, OWL.AllDisjointClasses):
        members = list(graph.items(graph.value(ax, OWL.members)))
        pairs.update((a, b) for i, a in enumerate(members) for b in members[i+1:]
                     if isinstance(a, URIRef) and isinstance(b, URIRef))
    return [{'class': str(s), 'disjoint_ancestors': sorted([str(a), str(b)])}
            for s in sorted(classes) for a, b in sorted(pairs)
            if a in ancestors(s) and b in ancestors(s)]

def main():
    a = Graph().parse(ROOT/'sources/aicor-ontology.ttl')
    b = Graph().parse(ROOT/'sources/cram-aicor-ontology.ttl')
    dul = Graph().parse(ROOT/'DUL.owl', format='xml')
    na, nb = normalized(a), normalized(b)
    missing = set(na) - set(nb)
    assert not missing, 'The CRAM input is not a structural superset; review before merging.'
    sa = {s for s in a.subjects() if isinstance(s, URIRef)}
    sb = {s for s in b.subjects() if isinstance(s, URIRef)}
    siga, sigb = signature_function(a), signature_function(b)
    delta = []
    for s in sorted(sa & sb):
        aa = {p.n3()+' '+siga(o) for p, o in a.predicate_objects(s)}
        bb = {p.n3()+' '+sigb(o) for p, o in b.predicate_objects(s)}
        if aa != bb:
            delta.append({'subject':str(s),'removed':sorted(aa-bb),'added':sorted(bb-aa)})
    (ROOT/'reports/shared_term_changes.json').write_text(json.dumps(delta, indent=2)+'\n')
    serialize(nb, ROOT/'candidates/aicor-unified-preserved.ttl')

    # Only explicit dul_* placeholders are replaced. Same-named local terms and
    # soma_Action are not collapsed merely because their spelling is similar.
    aliases = {s: D[str(s).split('#')[-1][4:]] for s in b.subjects(RDF.type, OWL.Class)
               if str(s).startswith(str(N)+'dul_')}
    assert len(aliases) == 14
    assert all((target, RDF.type, OWL.Class) in dul for target in aliases.values())
    candidate = Graph()
    removed = Graph()
    for s, p, o in nb:
        if s == OLD or s in aliases:
            removed.add((s,p,o))
            continue
        candidate.add((s, p, aliases.get(o, o)))
    candidate.add((NEW,RDF.type,OWL.Ontology))
    candidate.add((NEW,OWL.imports,DUL))
    candidate.add((NEW,OWL.versionInfo,Literal('0.1.0-candidate')))
    candidate.add((NEW,RDFS.label,Literal('Unified AICOR with DUL: review candidate',lang='en')))
    candidate.add((NEW,RDFS.comment,Literal('Local working candidate. Does not certify code alignment or full OWL consistency. See REVIEW.md and reports.',lang='en')))

    before = hierarchy_conflicts(candidate + dul)
    (ROOT/'reports/dul_conflicts_before.json').write_text(json.dumps(before,indent=2)+'\n')
    # These inherited category edges make descendants simultaneously members of
    # disjoint DUL categories. Keep the original axioms separately for review.
    fixes = [
      ('ComputationalMedium','InformationObject','Description descendants conflict with InformationObject.'),
      ('ComputationalProcess','Process','Description descendants conflict with Event through Process.'),
      ('ReifiedTemporalEntity','InformationObject','EventType is also a Description.'),
      ('ExecutiveCapability','Quality','Description descendants conflict with Quality.'),
      ('TaskStatus','InformationObject','TaskStatus inherits Description through FluentType.'),
    ]
    removed_alignment = Graph()
    for source, target, reason in fixes:
        triple = (N[source],RDFS.subClassOf,D[target])
        assert triple in candidate
        candidate.remove(triple); removed_alignment.add(triple)
    serialize(removed_alignment,ROOT/'reports/removed_dul_parent_axioms.ttl')
    serialize(removed,ROOT/'reports/replaced_source_metadata.ttl')
    serialize(candidate,ROOT/'candidates/aicor-unified-dul.ttl')
    after = hierarchy_conflicts(candidate + dul)
    (ROOT/'reports/dul_conflicts_after.json').write_text(json.dumps(after,indent=2)+'\n')
    with (ROOT/'reports/term_mapping.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['old_iri','new_iri','decision'])
        for s,o in sorted(aliases.items()):w.writerow([s,o,'Replace explicit DUL placeholder with imported DUL class'])
    declarations = {OWL.Class,OWL.ObjectProperty,OWL.DatatypeProperty,OWL.AnnotationProperty}
    with (ROOT/'reports/inventory.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['iri','types','in_aicor','in_cram_aicor','candidate_iri'])
        for s in sorted(sa|sb):
            types=sorted(str(t).split('#')[-1] for t in b.objects(s,RDF.type) if t in declarations)
            if types:w.writerow([s,';'.join(types),s in sa,s in sb,aliases.get(s,s)])
    # Offline OWL/XML copy for external reasoning, without remote imports.
    closure = candidate + dul
    closure.remove((None,OWL.imports,None))
    closure.serialize(ROOT/'candidates/aicor-unified-dul-offline.owl',format='xml')
    assert len(Graph().parse(ROOT/'candidates/aicor-unified-dul-offline.owl')) == len(closure)
    summary = {'inputs': {'aicor':stats(a),'cram_aicor':stats(b),'dul':stats(dul)},
       'normalized':{'aicor':stats(na),'cram_aicor':stats(nb)},
       'aicor_structurally_contained_in_cram':not missing,
       'shared_named_subjects':len(sa&sb),'aicor_only_named_subjects':len(sa-sb),
       'cram_only_named_subjects':len(sb-sa),'changed_shared_subjects':len(delta),
       'candidate':stats(candidate),'dul_placeholder_replacements':len(aliases),
       'removed_parent_edges':[{'subject':s,'parent':'dul:'+t,'reason':r} for s,t,r in fixes],
       'named_hierarchy_conflicts_before':len(before),'named_hierarchy_conflicts_after':len(after),
       'scope':'Named-parent disjointness screening is NOT complete OWL consistency checking.',
       'source_sha256':{str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest()
                         for p in [ROOT/'DUL.owl',ROOT/'sources/aicor-ontology.ttl',ROOT/'sources/cram-aicor-ontology.ttl']}}
    (ROOT/'reports/validation.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__ == '__main__':
    main()
