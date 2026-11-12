"""Classify each modular import closure independently, offline, with a time bound."""
import json,subprocess,sys,tempfile,time,os,signal
from pathlib import Path
from rdflib import OWL
from ontology_imports import ROOT,REGISTRY,closure
worker='''
import json,sys,owlready2 as o
import owlready2.reasoning as r
r.JAVA_MEMORY=2048
w=o.World();w.get_ontology(sys.argv[1]).load()
try:
 o.sync_reasoner(w,debug=0)
 print(json.dumps({'consistent':True,'unsatisfiable_classes':[c.iri for c in w.inconsistent_classes() if c.iri!='http://www.w3.org/2002/07/owl#Nothing']}))
except o.OwlReadyInconsistentOntologyError: print(json.dumps({'consistent':False}))
'''
results=[]
with tempfile.TemporaryDirectory() as tmp:
 for iri,path in REGISTRY.items():
  if not path.startswith(('aspects/','integrations/')):continue
  g,imports=closure(iri);g.remove((None,OWL.imports,None));p=Path(tmp)/'closure.owl';g.serialize(p,format='xml');start=time.monotonic()
  proc=subprocess.Popen([sys.executable,'-c',worker,p.as_uri()],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True)
  try:
   out,err=proc.communicate(timeout=30);result=json.loads(out) if proc.returncode==0 else {'error':err,'returncode':proc.returncode}
  except subprocess.TimeoutExpired:
   os.killpg(proc.pid,signal.SIGKILL);proc.communicate();result={'status':'timeout'}
  result.update(ontology=iri,imports=imports,seconds=round(time.monotonic()-start,2));results.append(result);print(iri.split('/')[-1],result.get('consistent'),result.get('unsatisfiable_classes',result.get('status')),flush=True)
(ROOT/'reports/import-classification.json').write_text(json.dumps(results,indent=2)+'\n');assert all(r.get('consistent') is True and r.get('unsatisfiable_classes')==[] for r in results)
