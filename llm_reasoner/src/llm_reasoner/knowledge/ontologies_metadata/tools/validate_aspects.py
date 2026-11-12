"""Validate aspect examples, structural faults, and three-valued condition semantics."""
from pathlib import Path
from itertools import product
from decimal import Decimal
import json
from rdflib import Graph, Namespace, RDF, Literal, XSD
from pyshacl import validate
from aspect_runtime import A,P,C,structural_errors,evaluate
R=Path(__file__).resolve().parents[1];E=Namespace('http://aicor.knowledge/ontologies/action-examples.owl#')
schema=Graph()
for name in ['core/action-description-core.ttl','aspects/structure/participants-relations.ttl','aspects/structure/conditions-outcomes.ttl']:schema+=Graph().parse(R/name)
shapes=Graph().parse(R/'core/action-core-shapes.ttl')+Graph().parse(R/'aspects/validation/aspect-shapes.ttl')
data=Graph().parse(R/'examples/stereotypical-actions.ttl')+Graph().parse(R/'examples/aspect-actions.ttl')
results=[]
def record(name,passed):results.append({'case':name,'passed':bool(passed)})
def check(g):
 ok,_,txt=validate(g,shacl_graph=shapes,ont_graph=schema,inference='rdfs',advanced=True)
 errors=structural_errors(g+schema)
 return bool(ok) and not errors,txt,errors
ok,txt,errors=check(data);record('valid-two-action-aspect-examples',ok)
(R/'reports/aspects-shacl.txt').write_text(txt+'\n'+json.dumps(errors,indent=2))
mutations={
'missing-relation-argument':lambda g:g.remove((E.sourceContainmentPattern,P.hasArgumentBinding,E.sourceContainmentPatternBinding2)),
'foreign-signature-argument':lambda g:g.set((E.sourceContainmentPatternBinding2,P.bindsArgument,P.ControlsArgument2)),
'duplicate-argument-position':lambda g:g.set((E.sourceContainmentPatternBinding2,P.bindsArgument,P.ContainedInArgument1)),
'missing-state-scope':lambda g:g.remove((E.sourcePreAtom,C.inScope,None)),
'negation-two-operands':lambda g:g.add((E.lostControlExpression,C.hasOperand,E.contactPost)),
'cyclic-expression':lambda g:g.add((E.graspSuccessExpression,C.hasOperand,E.graspSuccessExpression)),
'wrong-comparison-operand-type':lambda g:g.set((E.pourAmountExpression,C.leftOperand,E.sourcePreAtom)),
'unknown-comparison-operator':lambda g:g.set((E.pourAmountExpression,C.operator,E.unimplementedOperator)),
'quantity-without-unit':lambda g:(g.add((E.constant,RDF.type,C.QuantityConstant)),g.add((E.constant,C.decimalValue,Literal('1',datatype=XSD.decimal)))),
'outcome-without-after-condition':lambda g:g.remove((E.pourOutcome,C.afterCondition,None)),
'extraneous-expression-field':lambda g:g.add((E.controlPost,C.hasOperand,E.contactPost)),
}
for name,mutate in mutations.items():
 g=Graph()+data;mutate(g);ok,_,_=check(g);record('reject-'+name,not ok)
g=data+schema
# Complete strong-Kleene AND, OR and NOT tables using two injected atomic conditions.
atom1=E.controlPost;atom2=E.contactPost;root=E.truthTest
key1=(C.PostState,P.Controls,(E.gripperSpecification,E.graspableObject));key2=(C.PostState,P.InContactWith,(E.gripperSpecification,E.graspableObject))
for kind in [C.Conjunction,C.Disjunction]:
 g.set((root,RDF.type,kind));g.add((root,C.hasOperand,atom1));g.add((root,C.hasOperand,atom2))
 for left,right in product([True,False,None],repeat=2):
  expected=(False if False in (left,right) else None if None in (left,right) else True) if kind==C.Conjunction else (True if True in (left,right) else None if None in (left,right) else False)
  record(f'{kind.split("#")[-1]}-{left}-{right}',evaluate(g,root,{key1:left,key2:right}) is expected)
for v in [True,False,None]:record('Negation-'+str(v),evaluate(g,E.lostControlExpression,{key1:v}) is (None if v is None else not v))
# Quantities below are synthetic evaluator fixtures, not recommended pouring amounts.
unit='http://qudt.org/vocab/unit/L';amountKey=(C.PostState,E.deliveredAmount);targetKey=(C.TaskContext,E.targetAmount)
record('unknown-missing-quantity',evaluate(g,E.pourAmountExpression) is None)
record('unknown-unit-mismatch',evaluate(g,E.pourAmountExpression,quantities={amountKey:('0.3',unit),targetKey:('300','http://qudt.org/vocab/unit/MilliL')}) is None)
record('partial-transfer-can-satisfy-goal',evaluate(g,E.pourAmountExpression,quantities={amountKey:('0.3',unit),targetKey:('0.2',unit)}) is True)
record('insufficient-delivery-fails-goal',evaluate(g,E.pourAmountExpression,quantities={amountKey:('0.1',unit),targetKey:('0.2',unit)}) is False)

baseQuantities={amountKey:('0.3',unit),targetKey:('0.2',unit)}
record('success-unknown-without-spill-data',evaluate(g,E.pourSuccessExpression,quantities=baseQuantities) is None)
record('success-false-on-excessive-spill',evaluate(g,E.pourSuccessExpression,quantities={**baseQuantities,(C.PostState,E.spillAmount):('0.02',unit),(C.TaskContext,E.spillTolerance):('0.01',unit)}) is False)
record('success-true-with-acceptable-spill',evaluate(g,E.pourSuccessExpression,quantities={**baseQuantities,(C.PostState,E.spillAmount):('0.005',unit),(C.TaskContext,E.spillTolerance):('0.01',unit)}) is True)
record('pre-and-post-inputs-distinct',evaluate(g,E.controlPost,{(C.PreState,P.Controls,key1[2]):True}) is None)
record('no-world-relation-materialized',(E.transferableSubstance,P.ContainedIn,E.sourceContainer) not in data)
# Exercise every comparison operator and constant quantity expressions.
l=E.leftConstant;r=E.rightConstant;cmp=E.comparisonTest
for n,v in [(l,'1'),(r,'2')]:g.add((n,RDF.type,C.QuantityConstant));g.add((n,C.decimalValue,Literal(v,datatype=XSD.decimal)));g.add((n,C.unitIRI,Literal(unit,datatype=XSD.anyURI)))
g.add((cmp,RDF.type,C.Comparison));g.add((cmp,C.leftOperand,l));g.add((cmp,C.rightOperand,r))
for op,expected in [(C.Equal,False),(C.NotEqual,True),(C.LessThan,True),(C.LessOrEqual,True),(C.GreaterThan,False),(C.GreaterOrEqual,False)]:g.set((cmp,C.operator,op));record('operator-'+op.split('#')[-1],evaluate(g,cmp) is expected)
(R/'reports/aspects-validation.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps({'checks':len(results),'passed':sum(r['passed'] for r in results),'failures':[r for r in results if not r['passed']]},indent=2))
assert all(r['passed'] for r in results)
