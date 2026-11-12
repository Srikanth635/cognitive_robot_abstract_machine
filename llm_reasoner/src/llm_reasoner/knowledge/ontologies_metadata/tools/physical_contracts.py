"""Bounded dimensional/accounting checks, not a fluid solver or controller."""
from decimal import Decimal
from rdflib import Namespace,RDF
from aspect_runtime import one
V=Namespace('http://aicor.knowledge/ontologies/model-contracts.owl#')
def quantity_contracts(g):
 result={}
 for q in g.subjects(RDF.type,V.QuantityContract):
  param=one(g,q,V.parameter);dims=tuple(int(x) for x in str(one(g,q,V.dimensionVector)).split())
  if len(dims)!=7:raise ValueError('Seven SI exponents required')
  item=(dims,str(one(g,q,V.unitIRI)))
  if param in result and result[param]!=item:raise ValueError('Conflicting quantity declarations')
  result[param]=item
 return result
def inspect_equation(g,contract,values=None,tolerance=Decimal('0')):
 values={} if values is None else values;tolerance=Decimal(str(tolerance))
 if not tolerance.is_finite() or tolerance<0:raise ValueError('Explicit finite nonnegative tolerance required')
 specs=quantity_contracts(g);result=one(g,contract,V.resultParameter);terms=list(g.objects(contract,V.hasTerm));params=[result]+[one(g,t,V.termParameter) for t in terms]
 if len(terms)<1:raise ValueError('Missing equation terms')
 missing=[str(p) for p in params if p not in specs]
 if missing:return {'dimensionCheck':'unknown','missingDeclarations':missing,'residual':None,'satisfied':None,'physicalValidityAssessed':False}
 signatures=[specs[p] for p in params]
 if len({x[0] for x in signatures})!=1:return {'dimensionCheck':'incompatible','residual':None,'satisfied':None,'physicalValidityAssessed':False}
 if len({x[1] for x in signatures})!=1:return {'dimensionCheck':'compatible','unitCheck':'conversion-required','residual':None,'satisfied':None,'physicalValidityAssessed':False}
 vals={}
 for p in params:
  if p not in values:return {'dimensionCheck':'compatible','unitCheck':'identical','residual':None,'satisfied':None,'physicalValidityAssessed':False}
  value,unit=values[p];value=Decimal(str(value))
  if not value.is_finite():raise ValueError('Nonfinite quantity')
  if str(unit)!=specs[p][1]:raise ValueError('Input unit does not match declaration')
  vals[p]=value
 total=Decimal('0')
 for term in terms:
  coefficient=Decimal(str(one(g,term,V.coefficient)))
  if not coefficient.is_finite():raise ValueError('Nonfinite coefficient')
  total+=coefficient*vals[one(g,term,V.termParameter)]
 residual=vals[result]-total
 return {'dimensionCheck':'compatible','unitCheck':'identical','residual':str(residual),'satisfied':abs(residual)<=tolerance,'physicalValidityAssessed':False}

def inspect_control_contract(g,controller):
 from control_runtime import inspect_controller,S,F
 result=inspect_controller(g,controller);specs=quantity_contracts(g);model=one(g,controller,S.controlsModel);unresolved=list(result['unresolved']);errors=[]
 for assignment in g.objects(model,S.hasAssignment):
  variable=one(g,assignment,S.variable);parameter=one(g,variable,F.parameter)
  if parameter not in specs:unresolved.append('quantity contract: '+str(parameter))
 for prop in [S.samplingPeriod,S.predictionHorizon]:
  for parameter in g.objects(controller,prop):
   if parameter not in specs:unresolved.append('time dimension/unit contract: '+str(parameter))
   elif specs[parameter][0]!=(0,0,1,0,0,0,0):errors.append('controller time parameter has non-time dimension')
 return {**result,'unresolved':sorted(set(unresolved)),'contractErrors':errors,'controllerSynthesized':False,'physicalValidityAssessed':False}
