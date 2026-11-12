"""Focused generation-artifact and RDF string-enumeration regressions."""
import hashlib
import json
import subprocess
import sys
from rdflib import Graph, Namespace, RDF, BNode, Literal, XSD
from pyshacl import validate
from ontology_imports import ROOT

B='http://aicor.knowledge/ontologies/'
MT=Namespace(B+'material-transfer.owl#'); K=Namespace(B+'knowledge-context.owl#')
E=Namespace(B+'primary-generation-tests.owl#')
checks=[]
def check(name,ok):checks.append({'case':name,'passed':bool(ok)})
path=ROOT/'generation/primary-action-templates.json'
before=path.read_bytes()
subprocess.run([sys.executable,str(ROOT/'tools/build_primary_action_templates.py')],check=True,capture_output=True,text=True)
check('template-builder-byte-reproduction',before==path.read_bytes())
d=json.loads(path.read_text());templates={t['action']:t for t in d['templates']}
expected={'pickup':{'manipulator','object','initial_support'},'place':{'manipulator','object','destination_support','target_region'},
          'navigate':{'robot','target_region','footprint','frame'},'pour':{'source','destination','material','transferred_portion'}}
check('four-requested-actions-only',set(templates)==set(expected))
for name,t in templates.items():
 check(name+':required-symbolic-roles',set(t['entity_roles'])==expected[name])
 check(name+':no-physical-scalar-defaults',all(p['scalar_default'] is None for p in t['parameters'].values()))
 symbols={'entity:'+s for s in t['entity_roles']}|{'parameter:'+p for p in t['parameters']}
 used=set()
 for condition in t['conditions'].values():
  for atom in condition['atoms']:used.update(atom['arguments'])
  for comparison in condition['comparisons']:used.update([comparison['left']['parameter'],comparison['right']['parameter']])
 check(name+':no-dangling-or-world-argument-symbols',used<=symbols)
check('not-a-full-output-schema-or-accepted-evidence',d['complete_output_schema'] is False and 'Proposed' in d['epistemic_policy'])
check('ontology-input-hashes-current',all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in d['ontology_sha256'].items()))

all_shapes=Graph().parse(ROOT/'domains/tool-transfer-shapes.ttl');shapes=Graph();todo=[MT.PrimaryTransferConsistencyShape];seen=set()
while todo:
 n=todo.pop()
 if n in seen:continue
 seen.add(n)
 for p,o in all_shapes.predicate_objects(n):
  shapes.add((n,p,o))
  if isinstance(o,BNode):todo.append(o)
for label,value,expected in [('plain-volume',Literal('volume'),True),('typed-volume',Literal('volume',datatype=XSD.string),True),
 ('plain-mass',Literal('mass'),True),('typed-mass',Literal('mass',datatype=XSD.string),True),
 ('count',Literal('count',datatype=XSD.string),False),('language-tagged',Literal('volume',lang='en'),False),
 ('trailing-newline',Literal('volume\n',datatype=XSD.string),False)]:
 g=Graph()
 for triple in [(E.action,RDF.type,MT.MaterialTransferDescription),(E.action,MT.quantityBasis,value),
                (E.action,MT.evaluationContext,E.context),(E.context,K.scopedDescription,E.action)]:g.add(triple)
 ok,_,_=validate(g,shacl_graph=shapes,inference='none',advanced=True)
 check('quantity-basis:'+label,bool(ok)==expected)
report={'release':'0.22.0','checks':checks,'passed':sum(c['passed'] for c in checks),
        'scope':'Symbolic generation template reproduction and RDF literal compatibility; no LLM transport or physical execution.'}
(ROOT/'reports/primary-generation-validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'checks':len(checks),'passed':report['passed'],'failures':[c for c in checks if not c['passed']]}))
assert all(c['passed'] for c in checks)
