"""Classify the three changed domains and assembly; generic consistency only.

The optional legacy AICOR bridge is excluded explicitly; its unresolved historical
reasoning boundary is not repaired or retested by this focused increment.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from rdflib import Graph, OWL
import owlready2.reasoning as reasoning
from ontology_imports import ROOT, closure

B = 'http://aicor.knowledge/ontologies/'
results = []
target = ROOT / 'reports/primary-action-reasoning.json'
worker = '''import sys,json,owlready2 as o
import owlready2.reasoning as r
r.JAVA_MEMORY=2048
w=o.World();w.get_ontology(sys.argv[1]).load();o.sync_reasoner(w,debug=0)
print(json.dumps({'consistent':True,'unsatisfiable_classes':[c.iri for c in w.inconsistent_classes() if c.iri!='http://www.w3.org/2002/07/owl#Nothing']}))
'''
def save():
    target.write_text(json.dumps({'release': '0.22.0', 'results': results,
        'optional_AICOR': 'Not attempted; unchanged legacy module and unresolved historical compatibility boundary.',
        'scope': 'OWL classification and generic consistency; not physical validation, evidence acceptance or execution.'}, indent=2) + '\n')

with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / 'input.owl'
    for name in ['object-manipulation', 'material-transfer', 'mobility-navigation', 'all-action-knowledge']:
        graph, imports = closure(B + name + '.owl')
        graph.remove((None, OWL.imports, None)); graph.serialize(path, format='xml')
        run = subprocess.run([sys.executable, '-c', worker, path.as_uri()], capture_output=True, text=True, timeout=60)
        if run.returncode:
            results.append({'ontology': name, 'status': 'failed', 'stdout': run.stdout, 'stderr': run.stderr}); save()
            raise RuntimeError(run.stderr)
        result = json.loads(run.stdout); result.update(ontology=B + name + '.owl', imports=imports, status='passed')
        results.append(result); save()
        assert result['consistent'] and not result['unsatisfiable_classes']
        print(json.dumps({'classified': name, 'unsatisfiable_named_classes': len(result['unsatisfiable_classes'])}), flush=True)
    graph, imports = closure(B + 'all-action-knowledge.owl')
    for file in (ROOT / 'examples').glob('*.ttl'):
        graph.parse(file)
    graph.parse(ROOT / 'demo/comprehensive-pouring.ttl')
    graph.remove((None, OWL.imports, None)); graph.serialize(path, format='xml')
    try:
        run = subprocess.run(['java', '-Xmx7168M', '-cp', reasoning._HERMIT_CLASSPATH,
            'org.semanticweb.HermiT.cli.CommandLine', '-k', path.as_uri()], capture_output=True, text=True, timeout=180)
        passed = run.returncode == 0 and 'is satisfiable' in run.stdout
        results.append({'scope': 'Combined generic knowledge + all examples + fictional demo',
                        'status': 'passed' if passed else 'failed', 'returncode': run.returncode,
                        'stdout': run.stdout, 'stderr': run.stderr, 'java_heap_mb': 7168, 'timeout_seconds': 180})
    except subprocess.TimeoutExpired:
        passed = False
        results.append({'scope': 'Combined generic knowledge + all examples + fictional demo',
                        'status': 'blocked', 'reason': '180-second timeout; no consistency conclusion'})
    save(); assert passed
    print(json.dumps({'combined_generic_consistency': 'passed'}), flush=True)
