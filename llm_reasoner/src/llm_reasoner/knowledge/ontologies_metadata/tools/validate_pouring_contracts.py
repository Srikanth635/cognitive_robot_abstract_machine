import json
from decimal import Decimal
from rdflib import Graph,Namespace,RDF,Literal,XSD
from pyshacl import validate
from ontology_imports import ROOT,closure
from aspect_runtime import A,C,P,evaluate,structural_errors,one
from temporal_runtime import temporal_errors
from physical_contracts import V,inspect_equation,inspect_control_contract
from transfer_contracts import transfer_errors
B='http://aicor.knowledge/ontologies/';E=Namespace(B+'action-examples.owl#');T=Namespace(B+'transfer-contracts.owl#');DB=Namespace(B+'diagnostic-causal-bridge.owl#')
g=Graph()
for p in (ROOT/'examples').glob('*.ttl'):g.parse(p)
schema,_=closure(B+'all-action-knowledge.owl');schema-=Graph().parse(ROOT/'DUL.owl');shapes=Graph().parse(ROOT/'core/action-core-shapes.ttl')
for p in (ROOT/'aspects').rglob('*shapes.ttl'):shapes.parse(p)
def valid(data):
 ok,_,text=validate(data,shacl_graph=shapes,ont_graph=schema,inference='rdfs',advanced=True)
 if data is g:(ROOT/'reports/pouring-contract-shacl.txt').write_text(text)
 return bool(ok)
tests=[]
def check(n,ok):tests.append({'case':n,'passed':bool(ok)})
check('all-example-shapes',valid(g));g+=schema;check('expression-integrity',not structural_errors(g+schema));check('monitor-guard-dependencies-updated',not temporal_errors(g))
check('transfer-and-diagnostic-link-integrity',not transfer_errors(g))
h=Graph()+g;h.set((E.destinationBinding2V8,P.bindsEntity,E.sourceContainerV8));check('wrong-bound-destination-rejected',bool(transfer_errors(h)))
h=Graph()+g;h.set((E.pourTransferContractV8,T.targetAmount,E.deliveredAmountV8));h.set((E.pourTransferContractV8,T.spillTolerance,E.sourceAfterV8));check('unreferenced-contract-quantity-rejected',bool(transfer_errors(h)))
h=Graph()+g;h.set((E.pourTransferContractV8,T.source,E.receivingContainerV8));check('collapsed-source-destination-rejected',bool(transfer_errors(h)))
h=Graph()+g;h.set((E.pourTransferContractV8,T.substance,E.unrelatedSubstance));check('foreign-substance-rejected',bool(transfer_errors(h)))
F=lambda name:E[name+'V8']
params={(C.PostState,F('deliveredAmount')):('0.3','volume'),(C.TaskContext,F('targetAmount')):('0.3','volume'),(C.PostState,F('spillAmount')):('0','volume'),(C.TaskContext,F('spillTolerance')):('0.01','volume')}
key=(C.PostState,P.ContainedIn,(E.transferredPortionV8,F('receivingContainer')))
check('destination-unknown-prevents-success',evaluate(g,F('pourSuccessExpression'),quantities=params) is None)
check('intended-destination-success',evaluate(g,F('pourSuccessExpression'),{key:True},params) is True)
check('wrong-destination-fails',evaluate(g,F('pourSuccessExpression'),{key:False},params) is False)
wrong=(C.PostState,P.ContainedIn,(E.transferredPortionV8,F('sourceContainer')))
check('other-container-does-not-substitute',evaluate(g,F('pourSuccessExpression'),{wrong:True},params) is None)
params[(C.PostState,F('spillAmount'))]=('0.02','volume');check('excess-spill-fails',evaluate(g,F('pourSuccessExpression'),{key:True},params) is False)
params[(C.PostState,F('spillAmount'))]=('0','volume');params[(C.PostState,F('deliveredAmount'))]=('0.2','volume');check('insufficient-transfer-fails',evaluate(g,F('pourSuccessExpression'),{key:True},params) is False)
check('partial-pour-does-not-require-empty-source',str(g.value(E.pourTransferContractV8,T.transferMode))=='partial')
for label,subject,prop in [('missing-destination',E.pourTransferContractV8,T.destination),('missing-quantity-referent',E.pourTransferContractV8,T.deliveredAmount),('missing-diagnostic-cause',E.spillDiagnosticAssociationV8,DB.causeCondition)]:
 h=Graph()+g;h.remove((subject,prop,None));check('reject-'+label,not valid(h))
h=Graph()+g;h.set((E.spillDiagnosticAssociationV8,DB.causeCondition,F('pourCauseCondition')));check('reject-mismatched-causal-antecedent',not valid(h))
unit='http://qudt.org/vocab/unit/M3';values={F('sourceBefore'):('0.5',unit),F('sourceAfter'):('0.19',unit),F('deliveredAmount'):('0.3',unit),F('spillAmount'):('0.01',unit)}
check('unpopulated-accounting-remains-unknown',inspect_equation(g,E.volumeAccountingContractV8)['satisfied'] is None)
accounting=inspect_equation(g,E.volumeAccountingContractV8,values);check('balanced-accounting',accounting['satisfied'] is True)
values[F('sourceAfter')]=('0.2',unit);check('unbalanced-accounting-detected',inspect_equation(g,E.volumeAccountingContractV8,values)['satisfied'] is False)
h=Graph()+g;h.set((E.sourceAfterQuantityContractV8,V.dimensionVector,Literal('1 0 0 0 0 0 0',datatype=XSD.string)));check('dimension-mismatch-detected',inspect_equation(h,E.volumeAccountingContractV8)['dimensionCheck']=='incompatible')
h=Graph()+g;h.set((E.sourceAfterQuantityContractV8,V.unitIRI,Literal('urn:test:other-volume-unit',datatype=XSD.anyURI)));check('conversion-never-assumed',inspect_equation(h,E.volumeAccountingContractV8)['unitCheck']=='conversion-required')
control=inspect_control_contract(g,F('pourController'));check('missing-dynamics-reported','state evolution equation' in control['unresolved']);check('unknown-control-units-reported',any('quantity contract' in x for x in control['unresolved']));check('controller-not-synthesized',not control['controllerSynthesized'])
h=Graph()+g;q=E.badTimeContractV8
for triple in [(q,RDF.type,V.QuantityContract),(q,V.parameter,F('pourSamplingPeriod')),(q,V.dimensionVector,Literal('3 0 0 0 0 0 0',datatype=XSD.string)),(q,V.unitIRI,Literal(unit,datatype=XSD.anyURI))]:h.add(triple)
check('wrong-controller-time-dimension-rejected',bool(inspect_control_contract(h,F('pourController'))['contractErrors']))
# Exact provenance of each descriptive edge in this revision, not blanket component endorsement.
claims=set(g.objects(E.pourRecordV8,A.hasClaim));covered=set()
for claim in claims:
 st=one(g,claim,A.hasStatement);obj=list(g.objects(st,A.statementObject))+list(g.objects(st,A.statementLiteral));covered.add((one(g,st,A.statementSubject),str(one(g,st,A.statementPredicateIRI)),obj[0]))
reachable=set();todo=[E.pourActionV8]
while todo:
 node=todo.pop()
 if node in reachable:continue
 reachable.add(node)
 for value in g.objects(node):
  if str(value).startswith(str(E)):todo.append(value)
from rdflib import RDFS
edges={(s,str(p),o) for s in reachable for p,o in g.predicate_objects(s) if p not in [RDF.type,RDFS.label,RDFS.comment]}
check('revision-substantive-claims-covered',bool(edges) and edges<=covered)
check('revision-claims-stay-proposed',all(g.value(c,A.hasStatus)==A.Proposed for c in claims))
check('old-success-unchanged',evaluate(g,E.pourSuccessExpression,quantities={(C.PostState,E.deliveredAmount):('0.3','volume'),(C.TaskContext,E.targetAmount):('0.3','volume'),(C.PostState,E.spillAmount):('0','volume'),(C.TaskContext,E.spillTolerance):('0.01','volume')}) is True)
report={'release':'0.12.0','checks':tests,'accounting':accounting,'control':control,'limitation':'Synthetic contract checks only; no physics solver, system identification, control synthesis or execution.'}
(ROOT/'reports/pouring-contract-validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'checks':len(tests),'passed':sum(x['passed'] for x in tests),'failures':[x for x in tests if not x['passed']]}));assert all(x['passed'] for x in tests)
