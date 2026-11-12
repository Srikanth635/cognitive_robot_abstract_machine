from ontology_imports import aspect_file
from pathlib import Path
import json
from rdflib import Graph,Namespace,RDF,Literal
from pyshacl import validate
from aspect_runtime import A
B='http://aicor.knowledge/ontologies/';K=Namespace(B+'causal-mechanisms.owl#');F=Namespace(B+'physical-models.owl#');E=Namespace(B+'action-examples.owl#')
R=Path(__file__).resolve().parents[1]
schema=Graph();data=Graph();shapes=Graph()
for name in ['core/action-description-core.ttl']+[str(aspect_file(n+'.ttl').relative_to(R)) for n in ['participants-relations','conditions-outcomes','phases-transitions','monitoring-requirements','causal-mechanisms','physical-models']]:schema+=Graph().parse(R/name)
for name in ['stereotypical-actions','aspect-actions','phase-actions','causal-physical-actions']:data+=Graph().parse(R/'examples'/(name+'.ttl'))
for name in ['core/action-core-shapes.ttl','aspects/validation/aspect-shapes.ttl','aspects/validation/temporal-shapes.ttl','aspects/validation/causal-physical-shapes.ttl']:shapes+=Graph().parse(R/name)
for path in (R/'integrations').glob('*.ttl'):schema += Graph().parse(path)
def check(g):
 ok,_,report=validate(g,shacl_graph=shapes,ont_graph=schema,inference='rdfs',advanced=True);errors=[]
 for model in g.subjects(RDF.type,F.PhysicalModelSpecification):
  vs=set(g.objects(model,F.hasVariable));symbols=[str(g.value(v,F.symbol)) for v in vs]
  if len(symbols)!=len(set(symbols)):errors.append('duplicate symbols')
  for eq in g.objects(model,F.hasEquation):
   if not set(g.objects(eq,F.usesVariable))<=vs:errors.append('foreign equation variable')
 # Causal explanatory chains are finite; feedback needs explicit temporal modelling.
 for model in g.subjects(RDF.type,K.CausalModel):
  members=set(g.objects(model,K.hasLink));done=set();active=set()
  def visit(n):
   if n in active:errors.append('causal chain cycle');return
   if n in done:return
   active.add(n)
   for dst in g.objects(n,K.precedesLink):
    if dst not in members:errors.append('foreign causal chain link')
    else:visit(dst)
   active.remove(n);done.add(n)
  for n in members:visit(n)
 return bool(ok) and not errors,report,errors
results=[]
def record(name,ok):results.append({'case':name,'passed':bool(ok)})
ok,txt,errors=check(data);record('valid-two-action-model-examples',ok);(R/'reports/physical-shacl.txt').write_text(txt+'\n'+str(errors))
mutations={
'missing-mechanism':lambda g:g.remove((E.pourCausalLink,K.mechanism,None)),
'missing-cause':lambda g:g.remove((E.pourCausalLink,K.antecedent,None)),
'missing-effect':lambda g:g.remove((E.pourCausalLink,K.consequent,None)),
'missing-mechanism-assumption':lambda g:g.remove((E.pourMechanism,K.assumption,None)),
'missing-model-scope':lambda g:g.remove((E.pourPhysicalModel,F.applicability,None)),
'foreign-equation-variable':lambda g:g.add((E.pourVolumeEquation,F.usesVariable,E.graspVariable_mu)),
'duplicate-variable-symbol':lambda g:g.set((E.pourVariable_V_before,F.symbol,Literal('V_after'))),
'missing-property-entity':lambda g:g.remove((E.pourProperty,F.describesEntity,None)),
'missing-property-parameter':lambda g:g.remove((E.graspProperty,F.valueParameter,None)),
'equation-without-language':lambda g:g.remove((E.pourVolumeEquation,F.expressionLanguage,None)),
'causal-cycle':lambda g:g.add((E.pourCausalLink,K.precedesLink,E.pourCausalLink)),
'foreign-causal-link':lambda g:g.add((E.pourCausalLink,K.precedesLink,E.graspCausalLink)),
'study-without-model':lambda g:g.remove((E.pourStudy,F.studiesModel,None)),
}
for name,mutate in mutations.items():
 g=Graph()+data;mutate(g);ok,_,_=check(g);record('reject-'+name,not ok)
record('properties-remain-unspecified',not any(data.objects(E.pourPropertyParameter,A.numericValue)) and not any(data.objects(E.graspPropertyParameter,A.numericValue)))
record('study-is-not-run-evidence',not any(data.subjects(RDF.type,F.ModelValidationRecord)))
record('new-claims-remain-proposed',all(data.value(E[n+'PhysicalClaim'+str(i)],A.hasStatus)==A.Proposed for n in ['pour','grasp'] for i in range(4)))
(R/'reports/physical-validation.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps({'checks':len(results),'passed':sum(r['passed'] for r in results),'failures':[r for r in results if not r['passed']]},indent=2));assert all(r['passed'] for r in results)
