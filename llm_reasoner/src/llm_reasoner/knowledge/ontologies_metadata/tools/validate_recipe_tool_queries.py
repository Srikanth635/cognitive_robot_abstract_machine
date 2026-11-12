"""Selected-record descriptive query contracts, without evidence acceptance."""
import json
from rdflib import Graph,Namespace
from ontology_imports import ROOT
B='http://aicor.knowledge/ontologies/';E=Namespace(B+'recipe-examples.owl#');S=Namespace(B+'specialized-tool-examples.owl#');g=Graph()
for p in (ROOT/'examples').glob('*.ttl'):g.parse(p)
queries={name:(ROOT/'queries'/(name+'.rq')).read_text() for name in ['recipe-stages','recipe-handoffs','specialized-tool-targets']};checks=[]
def check(name,ok):checks.append({'case':name,'passed':bool(ok)})
for name,count in [('pickCarryPlace',3),('routeOrientedPickCarryPlace',3),('cuttingCycle',6),('wipingCycle',6)]:
 rows=list(g.query(queries['recipe-stages'],initBindings={'record':E[name+'Record']}));check(name+':owned-ordered-stages',len(rows)==count and [int(r['index']) for r in rows]==list(range(1,count+1)) and all(r.recipe==E[name] for r in rows));check(name+':stage-contexts-distinct',len({r.context for r in rows})==count)
 rows=list(g.query(queries['recipe-handoffs'],initBindings={'record':E[name+'Record']}));check(name+':adjacent-handoff-inventory',len(rows)==count-1 and all(r.recipe==E[name] for r in rows))
for name in ['cutting','wiping','pressing','insertion']:
 rows=list(g.query(queries['specialized-tool-targets'],initBindings={'record':S[name+'Record']}));check(name+':selected-technique',len(rows)==1 and rows[0].action==S[name]);check(name+':depth-only-when-declared',rows[0].depth is not None if name=='insertion' else rows[0].depth is None)
for name,q in queries.items():check(name+':foreign-record-empty',not list(g.query(q,initBindings={'record':E.foreignRecord})))
(ROOT/'reports/recipe-tool-query-validation.json').write_text(json.dumps({'checks':checks,'scope':'Selected descriptive records, not accepted observations'},indent=2)+'\n');print(json.dumps({'checks':len(checks),'passed':sum(c['passed'] for c in checks)}));assert all(c['passed'] for c in checks)
