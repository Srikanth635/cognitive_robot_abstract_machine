"""Structural checks and completeness inspection; no controller or optimizer."""
from rdflib import Namespace,RDF
from aspect_runtime import A,one
B='http://aicor.knowledge/ontologies/';F=Namespace(B+'physical-models.owl#');O=Namespace(B+'objectives-tradeoffs.owl#');S=Namespace(B+'dynamics-control.owl#')
def control_errors(g):
 errors=[]
 for model in g.subjects(RDF.type,S.DynamicalModelSpecification):
  variables=set(g.objects(model,F.hasVariable));positions={};dimensions={};rolevars={}
  for assignment in g.objects(model,S.hasAssignment):
   try:
    variable=one(g,assignment,S.variable);role=one(g,assignment,S.role);position=int(one(g,assignment,S.position));dimension=int(one(g,assignment,S.dimension))
    if variable not in variables:errors.append('foreign vector variable')
    positions.setdefault(role,[]).append(position);rolevars.setdefault(role,[]).append(variable)
    if variable in dimensions and dimensions[variable]!=dimension:errors.append('inconsistent variable dimension')
    dimensions[variable]=dimension
   except (ValueError,TypeError):errors.append('malformed vector assignment')
  for role,values in positions.items():
   if sorted(values)!=list(range(1,len(values)+1)):errors.append('noncontiguous or duplicate vector positions')
   if len(set(rolevars[role]))!=len(rolevars[role]):errors.append('duplicate vector variable')
  if not {S.State,S.Input,S.Output}<=set(positions):errors.append('missing state/input/output role')
  for p in [S.evolutionEquation,S.outputEquation]:
   for eq in g.objects(model,p):
    if (model,F.hasEquation,eq) not in g:errors.append('dynamic equation not declared by model')
    if not set(g.objects(eq,F.usesVariable))<=variables:errors.append('foreign equation variable')
 for objset in g.subjects(RDF.type,O.ObjectiveSet):
  policy=g.value(objset,O.aggregationPolicy);ranks=[];weights=[]
  for term in g.objects(objset,O.hasTerm):
   if policy==O.Lexicographic:
    try:ranks.append(int(one(g,term,O.rank)))
    except (ValueError,TypeError):errors.append('missing lexicographic rank')
    if any(g.objects(term,O.weightParameter)):errors.append('lexicographic term has weight')
   if policy==O.WeightedSum:
    try:
     weight=one(g,term,O.weightParameter);norm=one(g,term,O.normalizedBy);scale=one(g,norm,O.scaleParameter)
     w=g.value(weight,A.numericValue);v=g.value(scale,A.numericValue)
     if w is not None:
      w=float(w);weights.append(w)
      if w<0:errors.append('negative objective weight')
     if v is not None and float(v)<=0:errors.append('nonpositive normalization scale')
    except (ValueError,TypeError):errors.append('weighted term lacks weight or normalization')
    if any(g.objects(term,O.rank)):errors.append('weighted term has rank')
   if policy==O.Pareto and (any(g.objects(term,O.rank)) or any(g.objects(term,O.weightParameter))):errors.append('Pareto term carries aggregation settings')
  if len(ranks)!=len(set(ranks)):errors.append('duplicate lexicographic ranks')
  if weights and len(weights)==len(list(g.objects(objset,O.hasTerm))) and not any(w>0 for w in weights):errors.append('all objective weights zero')
 for ctrl in g.subjects(RDF.type,S.ControllerSpecification):
  model=g.value(ctrl,S.controlsModel)
  for p in [S.samplingPeriod,S.predictionHorizon]:
   for param in g.objects(ctrl,p):
    v=g.value(param,A.numericValue)
    if v is not None and float(v)<=0:errors.append('nonpositive controller time parameter')
  parameters=set()
  for m in [model]+list(g.objects(model,S.describesPlant)):
   for var in g.objects(m,F.hasVariable):parameters.update(g.objects(var,F.parameter))
  for adaptation in g.objects(ctrl,S.hasAdaptation):
   if g.value(adaptation,S.updatesParameter) not in parameters:errors.append('adaptation updates foreign parameter')
 return errors

def inspect_controller(g,ctrl):
 model=one(g,ctrl,S.controlsModel);missing=[];vectors={}
 for assignment in g.objects(model,S.hasAssignment):
  role=str(one(g,assignment,S.role)).split('#')[-1];vectors[role]=vectors.get(role,0)+int(one(g,assignment,S.dimension))
 if not list(g.objects(model,S.evolutionEquation)):missing.append('state evolution equation')
 if not list(g.objects(model,S.outputEquation)):missing.append('output equation')
 for p in [S.samplingPeriod,S.predictionHorizon]:
  for param in g.objects(ctrl,p):
   if g.value(param,A.numericValue) is None:missing.append('value: '+str(param))
 objset=one(g,ctrl,S.objectiveSet)
 if g.value(objset,O.aggregationPolicy)==O.WeightedSum:
  for term in g.objects(objset,O.hasTerm):
   for param in list(g.objects(term,O.weightParameter))+[g.value(g.value(term,O.normalizedBy),O.scaleParameter)]:
    if param is not None and g.value(param,A.numericValue) is None:missing.append('value: '+str(param))
 return {'controller':str(ctrl),'model':str(model),'vectorDimensions':vectors,'unresolved':sorted(missing),'executionReadinessAssessed':False}
