import json
from decimal import Decimal
from rdflib import Graph,Namespace,RDF,RDFS,URIRef,Literal,XSD
from pyshacl import validate
from ontology_imports import ROOT,closure
from aspect_runtime import A,C,P,one,structural_errors
from tool_material_transfer import TI,MT,TC,TOOL_KINDS,signature,tool_errors,material_errors,assess
from physical_contracts import inspect_equation
from temporal_runtime import temporal_errors
B='http://aicor.knowledge/ontologies/';E=Namespace(B+'tool-transfer-examples.owl#');D=Namespace(B+'comprehensive-pouring-demo.owl#')
schema,_=closure(B+'all-action-knowledge.owl');schema-=Graph().parse(ROOT/'DUL.owl');data=Graph().parse(ROOT/'examples/tool-material-transfer-profiles.ttl');demo=Graph().parse(ROOT/'demo/comprehensive-pouring.ttl');g=data+demo+schema
shapes=Graph().parse(ROOT/'core/action-core-shapes.ttl')+Graph().parse(ROOT/'core/knowledge-context-shapes.ttl')
for folder in ['aspects','domains']:
 for p in (ROOT/folder).rglob('*shapes.ttl'):shapes.parse(p)
def valid(graph):
 ok,_,text=validate(graph,shacl_graph=shapes,ont_graph=schema,inference='rdfs',advanced=True)
 return bool(ok),text
checks=[]
def check(n,ok):checks.append({'case':n,'passed':bool(ok)})
ok,text=valid(data+demo);(ROOT/'reports/tool-transfer-shacl.txt').write_text(text);check('seven-profiles-and-comprehensive-demo-shapes',ok)
check('relation-expression-integrity',not structural_errors(g));check('tool-contract-integrity',not tool_errors(g));check('material-contract-integrity',not material_errors(g));check('demo-phase-monitor-integrity',not temporal_errors(g))
profiles=['toolPositioning','toolEngagement','toolUse','toolDisengagement','partialTransfer','completeTransfer','pouringTransfer']
def satisfying(action):
 atoms,pairs=signature(g,one(g,one(g,action,A.hasSuccessCriterion),C.hasExpression));facts={(scope,rel,args):not neg for scope,rel,args,neg in atoms};quantities={}
 for op,scope,left,rscope,right in pairs:
  quantities[(scope,left)]=('0.2',str(one(g,left,A.unitIRI)));quantities[(rscope,right)]=('0.25',str(one(g,right,A.unitIRI)))
  if op==C.GreaterOrEqual:quantities[(scope,left)]=('0.26',str(one(g,left,A.unitIRI)))
 return facts,quantities
for name in profiles:
 action=E[name];facts,quantities=satisfying(action)
 check(name+':unknown-without-evidence',assess(g,action)['success'] is None)
 check(name+':supplied-state-satisfies',assess(g,action,facts,quantities)['success'] is True)
 check(name+':not-execution-approval',not assess(g,action,facts,quantities)['executionReadinessAssessed'])
 key=next(iter(facts));bad=dict(facts);bad[key]=not bad[key];check(name+':violated-required-state-fails',assess(g,action,bad,quantities)['success'] is False)
 bad=dict(facts);bad.pop(key);check(name+':missing-required-state-is-unknown',assess(g,action,bad,quantities)['success'] is None)
# Tool motion/contact are not themselves functional effects.
check('contact-alone-not-functional-use',assess(g,E.toolUse,{(C.PostState,P.InContactWith,(E.toolUseWorkingpart,E.toolUseTarget)):True})['success'] is None)
f,q=satisfying(E.toolDisengagement);f[(C.PostState,P.InContactWith,(E.toolDisengagementWorkingpart,E.toolDisengagementTarget))]=True;check('still-contacting-not-disengaged',assess(g,E.toolDisengagement,f,q)['success'] is False)
for label,s,p,o in [('foreign-working-part',E.toolUseWorkingpart,TI.workingPartOf,E.toolEngagementTool),('missing-effect-binding',E.toolUse,TI.intendedEffect,E.toolEngagementIntendedeffect),('policy-without-required-contact',E.toolUse,TI.contactPolicy,Literal('required',datatype=XSD.string))]:
 h=Graph()+g;h.set((s,p,o));check('reject-'+label,bool(tool_errors(h)))
for name in ['partialTransfer','completeTransfer','pouringTransfer']:
 action=E[name];f,q=satisfying(action);contract=one(g,action,TC.hasTransferContract);key=(C.PostState,one(g,contract,TC.deliveredAmount));q[key]=('0.1',q[key][1]);check(name+':under-delivery-fails',assess(g,action,f,q)['success'] is False)
 f,q=satisfying(action);q[(C.PostState,one(g,contract,TC.spillAmount))]=('1',str(one(g,one(g,contract,TC.spillAmount),A.unitIRI)));check(name+':excess-spillage-fails',assess(g,action,f,q)['success'] is False)
 f,q=satisfying(action);key=next(iter(q));q[key]=('-1',q[key][1])
 try:assess(g,action,f,q);check(name+':negative-amount-rejected',False)
 except ValueError:check(name+':negative-amount-rejected',True)
f,q=satisfying(E.completeTransfer);q[(C.PostState,one(g,E.completeTransfer,MT.residualAmount))]=('0.5','http://qudt.org/vocab/unit/M3');check('complete-transfer-needs-small-residual',assess(g,E.completeTransfer,f,q)['success'] is False)
f,q=satisfying(E.partialTransfer);check('partial-transfer-no-empty-source-default',assess(g,E.partialTransfer,f,q)['success'] is True)
for label,s,p,o in [('collapsed-transfer-ends',E.partialTransferContract,TC.destination,E.partialTransferSource),('wrong-transfer-unit',E.partialTransferDeliveredamount,A.unitIRI,Literal('http://qudt.org/vocab/unit/KiloGM',datatype=XSD.anyURI)),('wrong-mode',E.completeTransferContract,TC.transferMode,Literal('partial',datatype=XSD.string)),('wrong-basis',E.partialTransfer,MT.quantityBasis,Literal('mass',datatype=XSD.string))]:
 h=Graph()+g;h.set((s,p,o));check('reject-'+label,bool(material_errors(h)))
# Both mass and volume are explicit scalar bases; no conversion inferred.
h=Graph()+g;h.set((E.partialTransfer,MT.quantityBasis,Literal('mass',datatype=XSD.string)))
for p in [TC.deliveredAmount,TC.targetAmount,TC.spillAmount,TC.spillTolerance]:h.set((one(h,E.partialTransferContract,p),A.unitIRI,Literal('http://qudt.org/vocab/unit/KiloGM',datatype=XSD.anyURI)))
check('mass-basis-profile-supported',not material_errors(h))
# Demo arithmetic and state are fictional inputs, kept apart from observations.
view=json.loads((ROOT/'demo/comprehensive-pouring.json').read_text());check('demo-marked-fictional',view['illustrationOnly'] and view['hypothesizedOutcome']['notAnObservation'])
for field in ['tool','workingPart','target','intendedEffect']:
 check('demo-readable-and-RDF-tool-'+field,view['toolInteraction'][field]==str(one(demo,D.toolUse,TI[field])))
check('demo-shared-working-part',one(demo,D.toolUse,TI.workingPart)==D.bottleMouth)
check('demo-shared-effect',one(demo,D.toolUse,TI.intendedEffect)==D.flowIntoGlass)
atoms,pairs=signature(g,one(g,one(g,D.pouring,A.hasSuccessCriterion),C.hasExpression));facts={(scope,rel,args):not neg for scope,rel,args,neg in atoms};quantities={}
for op,scope,left,rscope,right in pairs:
 for key in [(scope,left),(rscope,right)]:quantities[key]=(str(one(g,key[1],A.numericValue)),str(one(g,key[1],A.unitIRI)))
check('fictional-demo-criteria-satisfy',assess(g,D.pouring,facts,quantities)['success'] is True)
values={D.sourceBefore:('0.000600','http://qudt.org/vocab/unit/M3'),D.sourceAfter:('0.000347','http://qudt.org/vocab/unit/M3'),D.deliveredAmount:('0.000252','http://qudt.org/vocab/unit/M3'),D.spillAmount:('0.000001','http://qudt.org/vocab/unit/M3')};accounting=inspect_equation(g,D.volumeAccountingContract,values);check('fictional-volume-accounting-balanced',accounting['satisfied'] is True)
check('demo-partial-retains-source-content',Decimal(str(one(g,D.sourceAfter,A.numericValue)))>0)
check('no-world-bindings',not list((data+demo).triples((None,A.hasBinding,None))))
# Each new record carries exact claims for its linked descriptive graph.
for graph in [data,demo]:
 for record in graph.subjects(RDF.type,A.KnowledgeRecord):
  covered=set()
  for claim in graph.objects(record,A.hasClaim):
   st=one(graph,claim,A.hasStatement);objects=list(graph.objects(st,A.statementObject))+list(graph.objects(st,A.statementLiteral));covered.add((one(graph,st,A.statementSubject),str(one(graph,st,A.statementPredicateIRI)),objects[0]))
  root=one(graph,record,A.expressesDescription);ns=str(root).split('#')[0]+'#';seen=set();todo=[root]
  while todo:
   node=todo.pop()
   if node in seen:continue
   seen.add(node);todo.extend(o for o in graph.objects(node) if isinstance(o,URIRef) and str(o).startswith(ns))
  expected={(s,str(p),o) for s in seen for p,o in graph.predicate_objects(s) if p not in [RDF.type,RDFS.label,RDFS.comment]};check('exact-provenance:'+str(record),bool(expected) and expected<=covered)
new_data=data+demo
check('all-new-claims-proposed',all(new_data.value(c,A.hasStatus)==A.Proposed for c in new_data.subjects(RDF.type,A.KnowledgeClaim)))
report={'release':'0.17.0','checks':checks,'fictionalDemoAccounting':accounting,'limitation':'All example values and assumed outcomes are invented; no physical simulation or safe operating limits established.'};(ROOT/'reports/tool-material-transfer-validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'checks':len(checks),'passed':sum(x['passed'] for x in checks),'failures':[x for x in checks if not x['passed']]}));assert all(x['passed'] for x in checks)
