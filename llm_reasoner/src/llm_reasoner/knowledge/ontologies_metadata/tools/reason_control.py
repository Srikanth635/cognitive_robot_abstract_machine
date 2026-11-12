"""Classify all four aspect modules with the core and supplied DUL offline; bound subprocess lifetime."""
from pathlib import Path
import json, subprocess, sys, os, signal, time
from rdflib import Graph, OWL
R=Path(__file__).resolve().parents[1]
g=Graph().parse(R/'DUL.owl')+Graph().parse(R/'core/action-description-core.ttl')
g += Graph().parse(R/'aspects/structure/participants-relations.ttl') + Graph().parse(R/'aspects/structure/conditions-outcomes.ttl')
g += Graph().parse(R/'aspects/structure/phases-transitions.ttl') + Graph().parse(R/'aspects/observation/monitoring-requirements.ttl')
g += Graph().parse(R/'aspects/causality/causal-mechanisms.ttl') + Graph().parse(R/'aspects/physics-control/physical-models.ttl')
g += Graph().parse(R/'aspects/planning/objectives-tradeoffs.ttl') + Graph().parse(R/'aspects/physics-control/dynamics-control.ttl')
for path in (R/'integrations').glob('*.ttl'):g += Graph().parse(path)
g.remove((None,OWL.imports,None))
pth=R/'reports/control-reasoning-input.owl';g.serialize(pth,format='xml')
worker='''
import json, sys
import owlready2 as o
import owlready2.reasoning as r
r.JAVA_MEMORY=2048
w=o.World();w.get_ontology(sys.argv[1]).load()
try:
 o.sync_reasoner(w,debug=0)
 print(json.dumps({'consistent':True,'unsatisfiable_classes':[c.iri for c in w.inconsistent_classes() if c.iri != 'http://www.w3.org/2002/07/owl#Nothing']}))
except o.OwlReadyInconsistentOntologyError:
 print(json.dumps({'consistent':False}))
'''
start=time.monotonic();p=subprocess.Popen([sys.executable,'-c',worker,pth.as_uri()],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True)
try:
 stdout,stderr=p.communicate(timeout=60)
 result=json.loads(stdout) if p.returncode==0 else {'error':stderr,'returncode':p.returncode}
except subprocess.TimeoutExpired:
 os.killpg(p.pid,signal.SIGKILL);p.communicate();result={'status':'timeout','limit_seconds':60}
result['scope']='TBox and controlled vocabulary: all eight aspects + core + DUL; excludes AICOR and action examples'
result['elapsed_seconds']=round(time.monotonic()-start,2)
(R/'reports/control-hermit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
pth.unlink()
if result.get('consistent') is not True or result.get('unsatisfiable_classes'):sys.exit(1)
