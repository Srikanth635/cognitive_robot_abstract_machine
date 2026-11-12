"""Verify a filesystem-only aspect refactor against the 0.14.0 release ZIP."""
import json,hashlib,zipfile,sys,ast
from pathlib import Path
from ontology_imports import ROOT,REGISTRY,closure
baseline=Path(sys.argv[1]);moves=json.loads((ROOT/'aspect-layout-migration.json').read_text());checks=[]
def check(name,ok):checks.append({'case':name,'passed':bool(ok)})
with zipfile.ZipFile(baseline) as archive:
 registry=json.loads(archive.read('gpt_cram_ontologies/ontology-registry.json'))
 check('same-ontology-IRIs',set(registry)==set(REGISTRY))
 for iri,old in registry.items():check('registry-relocation:'+iri,REGISTRY[iri]==moves.get(old,old))
 for name in archive.namelist():
  if name.endswith(('.ttl','.owl')):
   relative='/'.join(name.split('/')[1:]);current=ROOT/moves.get(relative,relative)
   check('identical-RDF-bytes:'+relative,current.is_file() and archive.read(name)==current.read_bytes())
 for old,new in moves.items():
  check('one-copy-only:'+old,not (ROOT/old).exists() and (ROOT/new).is_file())
 # New script references must resolve; historical audit intentionally uses baseline paths.
 for path in (ROOT/'tools').glob('*.py'):
  tree=ast.parse(path.read_text())
  if path.name=='audit_consolidation.py':continue
  for node in ast.walk(tree):
   if isinstance(node,ast.Constant) and isinstance(node.value,str) and node.value.startswith('aspects/') and node.value.endswith('.ttl'):
    check('tool-path:'+path.name+':'+node.value,(ROOT/node.value).is_file())
 _,imports=closure('http://aicor.knowledge/ontologies/all-action-knowledge.owl');check('both-domains-still-imported',all('http://aicor.knowledge/ontologies/'+name+'.owl' in imports for name in ['robot-motion','object-manipulation']))
report={'release':'0.14.1','baseline':baseline.name,'baseline_sha256':hashlib.sha256(baseline.read_bytes()).hexdigest(),'moved_files':moves,'checks':checks,'passed':sum(c['passed'] for c in checks)}
(ROOT/'reports/aspect-layout-validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'checks':len(checks),'passed':report['passed'],'failures':[c for c in checks if not c['passed']]}));assert all(c['passed'] for c in checks)
