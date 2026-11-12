"""Scoped descriptive query cases; no evidence acceptance or execution."""
import json
from rdflib import Graph, Namespace
from ontology_imports import ROOT
E=Namespace('http://aicor.knowledge/ontologies/articulated-examples.owl#')
g=Graph()
for p in (ROOT/'examples').glob('*.ttl'):g.parse(p)
queries={name:(ROOT/'queries'/('articulated-'+name+'.rq')).read_text() for name in ['targets','guards']}
checks=[]
def check(name,ok):checks.append({'case':name,'passed':bool(ok)})
for name in ['revoluteConfiguration','prismaticConfiguration','revoluteOpening','prismaticOpening','revoluteClosing','prismaticClosing','releaseLatch','secureLatch']:
 rows=list(g.query(queries['targets'],initBindings={'record':E[name+'Record']}))
 check(name+':only-selected-targets',len(rows)==1 and rows[0].action==E[name] and rows[0].joint==E[name+'Joint'])
 check(name+':latch-action-has-no-invented-panel-target',bool(rows[0].target) if name not in ['releaseLatch','secureLatch'] else rows[0].target is None)
 rows=list(g.query(queries['guards'],initBindings={'record':E[name+'Record']}))
 check(name+':owned-guards',len(rows)==(6 if name in ['releaseLatch','secureLatch'] else 8) and all(r.action==E[name] for r in rows))
for q in queries:
 check(q+':unknown-record-empty',not list(g.query(queries[q],initBindings={'record':E.foreignRecord})))
report={'checks':checks,'scope':'Explicitly bound records; descriptive retrieval only'}
(ROOT/'reports/articulated-mechanisms-query-validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'checks':len(checks),'passed':sum(c['passed'] for c in checks)}));assert all(c['passed'] for c in checks)
