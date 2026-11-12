"""Classify tool refinements, recipes and assembly; check all examples and the demo, with optional CRAM."""
import json,tempfile,subprocess,sys
from pathlib import Path
from rdflib import Graph,OWL
import owlready2.reasoning as reasoning
from ontology_imports import ROOT,closure
B='http://aicor.knowledge/ontologies/';reports=[]
worker='''import sys,json,owlready2 as o
import owlready2.reasoning as r
r.JAVA_MEMORY=2048
w=o.World();w.get_ontology(sys.argv[1]).load();o.sync_reasoner(w,debug=0)
print(json.dumps({'consistent':True,'unsatisfiable_classes':[c.iri for c in w.inconsistent_classes() if c.iri!='http://www.w3.org/2002/07/owl#Nothing']}))
'''
with tempfile.TemporaryDirectory() as temp:
 p=Path(temp)/'input.owl'
 for name in ['specialized-tool-interactions','action-recipes','all-action-knowledge']:
  graph,imports=closure(B+name+'.owl');graph.remove((None,OWL.imports,None));graph.serialize(p,format='xml')
  r=subprocess.run([sys.executable,'-c',worker,p.as_uri()],capture_output=True,text=True,timeout=45)
  if r.returncode:raise RuntimeError(r.stderr)
  result=json.loads(r.stdout);result.update(ontology=B+name+'.owl',imports=imports);reports.append(result)
  assert result['consistent'] and not result['unsatisfiable_classes']
 for with_cram in [False,True]:
  graph,imports=closure(B+'all-action-knowledge.owl')
  if with_cram:
   cram,_=closure(B+'cram-action-bridge.owl');graph+=cram
  for path in (ROOT/'examples').glob('*.ttl'):graph.parse(path)
  graph.parse(ROOT/'demo/comprehensive-pouring.ttl')
  graph.remove((None,OWL.imports,None));graph.serialize(p,format='xml')
  scope='All action knowledge + all examples'+(' + optional CRAM bridge' if with_cram else '')
  try:
   r=subprocess.run(['java','-Xmx7168M','-cp',reasoning._HERMIT_CLASSPATH,'org.semanticweb.HermiT.cli.CommandLine','-k',p.as_uri()],capture_output=True,text=True,timeout=180)
   passed=r.returncode==0 and 'is satisfiable' in r.stdout
   result={'scope':scope,'status':'passed' if passed else 'blocked','returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'javaHeapMB':7168,'timeLimitSeconds':180,'limitation':'Consistency only, not full AICOR classification.'}
  except subprocess.TimeoutExpired as error:
   def decoded(value):return value.decode(errors='replace') if isinstance(value,bytes) else (value or '')
   passed=False
   result={'scope':scope,'status':'blocked','reason':'timeout','returncode':None,'stdout':decoded(error.stdout),'stderr':decoded(error.stderr),'javaHeapMB':7168,'timeLimitSeconds':180,'limitation':'No consistency conclusion: subprocess exceeded the time limit.'}
  reports.append(result)
  (ROOT/'reports/recipes-specialized-reasoning.json').write_text(json.dumps(reports,indent=2)+'\n')
  if not with_cram:assert passed
print(json.dumps(reports,indent=2))
