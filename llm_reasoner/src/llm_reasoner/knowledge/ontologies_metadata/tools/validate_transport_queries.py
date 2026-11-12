"""Explicit-record retrieval tests; no interval/evidence acceptance is performed by SPARQL."""
import json
from rdflib import Graph,Namespace
from ontology_imports import ROOT
E=Namespace('http://aicor.knowledge/ontologies/transport-examples.owl#')
g=Graph()
for p in (ROOT/'examples').glob('*.ttl'):g.parse(p)
queries={name:(ROOT/'queries'/('transport-'+name+'.rq')).read_text() for name in ['components','orientation']}
checks=[]
def check(name,ok):checks.append({'case':name,'passed':bool(ok)})
for name in ['carrying','routeCarrying','orientationCarrying','routeOrientationCarrying']:
 rows=list(g.query(queries['components'],initBindings={'record':E[name+'Record']}))
 check(name+':selected-components',len(rows)==1 and rows[0].action==E[name] and rows[0].load==E[name+'Load'])
 row=rows[0];check(name+':contexts-remain-distinct',len({row.context,row.holdingContext,row.mobilityContext})==3)
 rows=list(g.query(queries['orientation'],initBindings={'record':E[name+'Record']}))
 check(name+':orientation-not-fabricated',len(rows)==(1 if 'orientation' in name.lower() else 0))
for q in queries:check(q+':unknown-record-empty',not list(g.query(queries[q],initBindings={'record':E.foreignRecord})))
(ROOT/'reports/transport-carrying-query-validation.json').write_text(json.dumps({'checks':checks,'scope':'Selected descriptive records only'},indent=2)+'\n')
print(json.dumps({'checks':len(checks),'passed':sum(c['passed'] for c in checks)}));assert all(c['passed'] for c in checks)
