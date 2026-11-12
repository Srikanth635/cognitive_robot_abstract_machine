"""Reference semantics for the bounded condition language; not a robot controller.

Run structural validation before evaluation. Missing facts and incompatible units
produce None (unknown). No world lookup, LLM invocation, or RDF truth materialization.
"""
from decimal import Decimal
from rdflib import Namespace, RDF
B='http://aicor.knowledge/ontologies/'
A=Namespace(B+'action-description-core.owl#');P=Namespace(B+'participants-relations.owl#');C=Namespace(B+'conditions-outcomes.owl#')
KINDS=[C.AtomicCondition,C.Conjunction,C.Disjunction,C.Negation,C.Comparison,C.ParameterReference,C.QuantityConstant]
EDGES=[C.hasOperand,C.leftOperand,C.rightOperand]
def one(g,s,p):
 values=list(g.objects(s,p))
 if len(values)!=1:raise ValueError(f'Expected exactly one {p} on {s}')
 return values[0]
def structural_errors(g):
 errors=[]
 for rel in g.subjects(RDF.type,P.RelationDefinition):
  try:
   arity=int(one(g,rel,P.arity));args=list(g.objects(rel,P.hasArgumentDefinition));positions=[int(one(g,a,P.argumentPosition)) for a in args]
   if sorted(positions)!=list(range(1,arity+1)):errors.append('non-contiguous relation signature: '+str(rel))
  except (ValueError,TypeError):errors.append('invalid relation signature: '+str(rel))
 for pattern in g.subjects(RDF.type,P.RelationPattern):
  try:
   rel=one(g,pattern,P.usesRelation);expected=set(g.objects(rel,P.hasArgumentDefinition));bindings=list(g.objects(pattern,P.hasArgumentBinding));actual=[one(g,b,P.bindsArgument) for b in bindings]
   if set(actual)!=expected or len(actual)!=len(expected):errors.append('incomplete, foreign or duplicate argument binding: '+str(pattern))
  except ValueError:errors.append('invalid relation pattern: '+str(pattern))
 # Reject cycles: recursive symbolic expressions are outside this finite AST language.
 visiting=set();done=set()
 def visit(n):
  if n in visiting:errors.append('expression cycle: '+str(n));return
  if n in done:return
  visiting.add(n)
  for p in EDGES:
   for child in g.objects(n,p):visit(child)
  visiting.remove(n);done.add(n)
 nodes={s for kind in KINDS for s in g.subjects(RDF.type,kind)}
 allowed={C.AtomicCondition:{C.hasPattern,C.inScope},C.Conjunction:{C.hasOperand},C.Disjunction:{C.hasOperand},C.Negation:{C.hasOperand},C.Comparison:{C.leftOperand,C.rightOperand,C.operator},C.ParameterReference:{C.parameter,C.inScope},C.QuantityConstant:{C.decimalValue,C.unitIRI}}
 structural_props=set().union(*allowed.values())
 for n in nodes:
  visit(n);kinds=[k for k in KINDS if (n,RDF.type,k) in g]
  if len(kinds)!=1:errors.append('ambiguous expression type: '+str(n));continue
  if (set(g.predicates(n))&structural_props)-allowed[kinds[0]]:errors.append('unexpected expression fields: '+str(n))
 return errors

def evaluate(g,expression,facts=None,quantities=None):
 """Return True, False or None. Inputs use RDF URIRef keys and decimal/unit pairs."""
 facts={} if facts is None else facts;quantities={} if quantities is None else quantities
 stack=set()
 def kind(n):
  ks=[k for k in KINDS if (n,RDF.type,k) in g]
  if len(ks)!=1:raise ValueError('Expression must have exactly one concrete type')
  return ks[0]
 def quantity(n):
  k=kind(n)
  if k==C.QuantityConstant:raw=(one(g,n,C.decimalValue),str(one(g,n,C.unitIRI)))
  elif k==C.ParameterReference:raw=quantities.get((one(g,n,C.inScope),one(g,n,C.parameter)))
  else:raise ValueError('Expected quantity expression')
  if raw is None:return None
  value,unit=raw
  if not unit:return None
  value=Decimal(str(value))
  return (value,str(unit)) if value.is_finite() else None
 def run(n):
  if n in stack:raise ValueError('Cyclic expression')
  stack.add(n)
  try:
   k=kind(n)
   if k==C.AtomicCondition:
    pattern=one(g,n,C.hasPattern);rel=one(g,pattern,P.usesRelation);args=[]
    for b in g.objects(pattern,P.hasArgumentBinding):args.append((int(one(g,one(g,b,P.bindsArgument),P.argumentPosition)),one(g,b,P.bindsEntity)))
    key=(one(g,n,C.inScope),rel,tuple(entity for _,entity in sorted(args,key=lambda a:a[0])))
    result=facts.get(key)
    if result is not None and type(result) is not bool:raise ValueError('Atomic truth must be bool or None')
    return result
   if k in [C.Conjunction,C.Disjunction]:
    children=list(g.objects(n,C.hasOperand))
    if len(children)<2:raise ValueError('AND/OR needs two or more distinct operands')
    values=[run(c) for c in children]
    if k==C.Conjunction:return False if False in values else (None if None in values else True)
    return True if True in values else (None if None in values else False)
   if k==C.Negation:
    v=run(one(g,n,C.hasOperand));return None if v is None else not v
   if k==C.Comparison:
    left=quantity(one(g,n,C.leftOperand));right=quantity(one(g,n,C.rightOperand))
    if left is None or right is None or left[1]!=right[1]:return None
    l,r=left[0],right[0];op=one(g,n,C.operator)
    operations={C.Equal:lambda:l==r,C.NotEqual:lambda:l!=r,C.LessThan:lambda:l<r,C.LessOrEqual:lambda:l<=r,C.GreaterThan:lambda:l>r,C.GreaterOrEqual:lambda:l>=r}
    if op not in operations:raise ValueError('Unsupported comparison')
    return operations[op]()
   raise ValueError('Root must be boolean')
  finally:stack.remove(n)
 return run(expression)
