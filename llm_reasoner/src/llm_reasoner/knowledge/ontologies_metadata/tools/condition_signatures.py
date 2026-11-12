"""Finite conjunction/negation/quantity signature shared by domain authoring checks.

This is structural inspection, not arbitrary logical entailment.
"""
from rdflib import RDF
from aspect_runtime import C, P, one

def signature(g,expression):
 atoms=set();comparisons=set();active=set()
 def visit(n,neg=False):
  if n in active:raise ValueError('Cyclic expression')
  active.add(n)
  if (n,RDF.type,C.Conjunction) in g:
   if neg:raise ValueError('Negated conjunction outside profile')
   for c in g.objects(n,C.hasOperand):visit(c)
  elif (n,RDF.type,C.Negation) in g:visit(one(g,n,C.hasOperand),not neg)
  elif (n,RDF.type,C.AtomicCondition) in g:
   pattern=one(g,n,C.hasPattern);bindings=[]
   for b in g.objects(pattern,P.hasArgumentBinding):bindings.append((int(one(g,one(g,b,P.bindsArgument),P.argumentPosition)),one(g,b,P.bindsEntity)))
   atoms.add((one(g,n,C.inScope),one(g,pattern,P.usesRelation),tuple(e for _,e in sorted(bindings)),neg))
  elif (n,RDF.type,C.Comparison) in g:
   op=one(g,n,C.operator)
   if neg:
    inverse={C.GreaterThan:C.LessOrEqual,C.LessThan:C.GreaterOrEqual,C.GreaterOrEqual:C.LessThan,C.LessOrEqual:C.GreaterThan,C.Equal:C.NotEqual,C.NotEqual:C.Equal}
    if op not in inverse:raise ValueError('Unsupported comparison')
    op=inverse[op]
   left=one(g,n,C.leftOperand);right=one(g,n,C.rightOperand);comparisons.add((op,one(g,left,C.inScope),one(g,left,C.parameter),one(g,right,C.inScope),one(g,right,C.parameter)))
  else:raise ValueError('Unsupported profile expression')
  active.remove(n)
 visit(expression)
 if any((scope,rel,args,not neg) in atoms for scope,rel,args,neg in atoms):
  raise ValueError('Contradictory positive and negative atom in one conjunction/scope')
 inverse={C.GreaterThan:C.LessOrEqual,C.LessThan:C.GreaterOrEqual,C.GreaterOrEqual:C.LessThan,C.LessOrEqual:C.GreaterThan,C.Equal:C.NotEqual,C.NotEqual:C.Equal}
 if any((inverse.get(op),scope,left,rscope,right) in comparisons for op,scope,left,rscope,right in comparisons):
  raise ValueError('Contradictory complementary comparisons in one conjunction/scope')
 return atoms,comparisons


def criteria_assessment(precondition,success,fault):
 """Three-valued conjunction of supplied prerequisites, outcome and no-fault.

 This is not evidence acceptance, occurrence verification or execution readiness.
 In particular, absent fault evidence does not mean an explicitly false fault.
 """
 values=[precondition,success,None if fault is None else not fault]
 result=False if False in values else None if None in values else True
 return {'criteriaSatisfied':result,'successAndFaultReported':success is True and fault is True,
         'acceptedObservations':0,'worldUpdated':False,
         'executionReadinessAssessed':False,'executed':False}

