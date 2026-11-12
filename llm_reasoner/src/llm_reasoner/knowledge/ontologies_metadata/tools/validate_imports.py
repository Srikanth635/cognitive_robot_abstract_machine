from pathlib import Path
import json,sys,zipfile
from rdflib import Graph,URIRef,RDF,OWL
from rdflib.compare import isomorphic
from ontology_imports import ROOT,REGISTRY,closure
B='http://aicor.knowledge/ontologies/'
results=[];details={}
def record(n,ok):results.append({'case':n,'passed':bool(ok)})
# Every package reference must have a declaration within the actual imported closure.
for iri in REGISTRY:
 g,imports=closure(iri);details[iri]=imports
 missing=set()
 for triple in g:
  if triple[1]==OWL.imports:continue
  for t in triple:
   if isinstance(t,URIRef) and str(t).startswith(B) and str(t) not in REGISTRY and not list(g.predicate_objects(t)):missing.add(str(t))
 record('closed-local-references:'+iri,not missing)
 if missing:print('Unresolved terms',iri,missing)
import xml.etree.ElementTree as ET
for directory in sorted(p.parent for p in ROOT.rglob('catalog-v001.xml')):
 catalog=ET.parse(directory/'catalog-v001.xml')
 valid=True
 for node in catalog.getroot():
  path=(directory/node.attrib['uri']).resolve()
  part=Graph().parse(path)
  valid=valid and (URIRef(node.attrib['name']),RDF.type,OWL.Ontology) in part
 record('local-catalog:'+str(directory.relative_to(ROOT)),valid)
for module,excluded in [('objectives-tradeoffs',['physical-models','causal-mechanisms','phases-transitions']),('physical-models',['causal-mechanisms','phases-transitions']),('causal-mechanisms',['conditions-outcomes','phases-transitions']),('monitoring-requirements',['phases-transitions']),('capabilities-strategies',['failure-diagnosis','physical-models']),('failure-diagnosis',['capabilities-strategies','physical-models'])]:
 _,imports=closure(B+module+'.owl');record('independent:'+module,all(B+n+'.owl' not in imports for n in excluded))
# Compare the previous complete aspect axiom set against the new complete assembly.
if len(sys.argv)>1:
 previous=Graph()
 with zipfile.ZipFile(sys.argv[1]) as z:
  for name in z.namelist():
   if '/aspects/' in name and name.endswith('.ttl') and 'shapes' not in name:previous+=Graph().parse(data=z.read(name).decode(),format='turtle')
 current=Graph()
 for folder in ['aspects','integrations']:
  for p in (ROOT/folder).rglob('*.ttl'):
   if 'shapes' not in p.name:current+=Graph().parse(p)
 def axioms(g):
  for ontology in list(g.subjects(RDF.type,OWL.Ontology)):g.remove((ontology,None,None))
  return g
 record('combined-aspect-axioms-preserved',isomorphic(axioms(previous),axioms(current)))
else:print('Baseline comparison skipped; pass the prior v0.5.0 ZIP to repeat it.')
(ROOT/'reports/import-validation.json').write_text(json.dumps({'checks':results,'closures':details},indent=2)+'\n')
print(json.dumps({'checks':len(results),'passed':sum(r['passed'] for r in results),'failures':[r for r in results if not r['passed']]},indent=2));assert all(r['passed'] for r in results)
