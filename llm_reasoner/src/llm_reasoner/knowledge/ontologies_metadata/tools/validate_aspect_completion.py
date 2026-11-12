import json
from rdflib import Graph,Namespace,RDF,Literal,XSD
from pyshacl import validate
from ontology_imports import ROOT,closure
from aspect_runtime import A
B='http://aicor.knowledge/ontologies/';F=Namespace(B+'force-dynamics.owl#');I=Namespace(B+'image-schemas.owl#');M=Namespace(B+'entities-materials-affordances.owl#');S=Namespace(B+'sensorimotor-events.owl#');E=Namespace(B+'action-examples.owl#')
schema,_=closure(B+'all-action-knowledge.owl');schema-=Graph().parse(ROOT/'DUL.owl');data=Graph();shapes=Graph()
for p in (ROOT/'examples').glob('*.ttl'):data+=Graph().parse(p)
for p in [ROOT/'core/action-core-shapes.ttl']+list((ROOT/'aspects').rglob('*shapes.ttl')):shapes+=Graph().parse(p)
def integrity(g):
 errors=[]
 for n in g.subjects(RDF.type,F.ForceDynamicDescription):
  if g.value(n,F.agonist)==g.value(n,F.antagonist):errors.append('force roles collapsed')
 for schema in g.subjects(RDF.type,I.SourcePathGoalSchema):
  elements=list(g.objects(schema,I.hasPathElement));positions=[int(g.value(e,I.position) or 0) for e in elements]
  if sorted(positions)!=list(range(1,len(elements)+1)):errors.append('invalid path order')
 for rel,cls in [(M.constituent,M.MaterialSpecification),(M.hasOpening,M.GeometrySpecification)]:
  done=set();active=set()
  def visit(n):
   if n in active:errors.append('cyclic descriptive composition');return
   if n in done:return
   active.add(n)
   for child in g.objects(n,rel):visit(child)
   active.remove(n);done.add(n)
  for n in g.subjects(RDF.type,cls):visit(n)
 for geometry in set(g.subjects(M.assetIRI,None)):
  if not list(g.objects(geometry,M.assetVersion)):errors.append('unversioned geometry asset')
 for boundary in g.subjects(RDF.type,S.EventBoundarySpecification):
  if g.value(boundary,S.beforePhase)==g.value(boundary,S.afterPhase):errors.append('boundary phases identical')
 for entitydesc in g.subjects(RDF.type,M.EntityDescription):
  for prop in g.objects(entitydesc,M.physicalProperty):
   pm=Namespace(B+'physical-models.owl#')
   if g.value(prop,pm.describesEntity)!=g.value(entitydesc,M.describesEntity):errors.append('property describes another entity')
 for action in g.subjects(A.hasComponent,None):
  phases=set(g.objects(action,A.hasPhase))
  for cp in g.objects(action,A.hasComponent):
   if (cp,RDF.type,S.ControlPointSpecification) in g:
    boundary=g.value(cp,S.atBoundary)
    if not {g.value(boundary,S.beforePhase),g.value(boundary,S.afterPhase)}<=phases:errors.append('control point outside action phases')
 return errors
def check(g):
 ok,_,txt=validate(g,shacl_graph=shapes,ont_graph=schema,inference='rdfs',advanced=True);errors=integrity(g);return bool(ok) and not errors,txt,errors
results=[]
def record(n,ok):results.append({'case':n,'passed':bool(ok)})
ok,txt,errors=check(data);record('valid-combined-aspect-knowledge',ok);(ROOT/'reports/completion-shacl.txt').write_text(txt+'\n'+str(errors))
mutations={
'missing-agonist':lambda g:g.remove((E.pourForcePattern,F.agonist,None)),
'collapsed-force-roles':lambda g:g.set((E.pourForcePattern,F.antagonist,E.pourAgonist)),
'missing-force-scope':lambda g:g.remove((E.pourForcePattern,F.applicability,None)),
'missing-source-path-goal':lambda g:g.remove((E.pourPathSchema,I.goal,None)),
'missing-contact-participant':lambda g:g.remove((E.graspContactSchema,I.secondContactEntity,None)),
'cyclic-material':lambda g:g.add((E.pourMaterialSpec,M.constituent,E.pourMaterialSpec)),
'cyclic-opening':lambda g:g.add((E.pourGeometry,M.hasOpening,E.pourGeometry)),
'unversioned-asset':lambda g:g.add((E.pourGeometry,M.assetIRI,Literal('urn:asset:example',datatype=XSD.anyURI))),
'missing-material-state':lambda g:g.remove((E.pourMaterialSpec,M.materialState,None)),
'affordance-without-context':lambda g:g.remove((E.pourAffordance,M.validWhen,None)),
'affordance-without-capability':lambda g:g.remove((E.pourAffordance,M.requiresCapability,None)),
'boundary-without-condition':lambda g:g.remove((E.pourBoundary,S.boundaryCondition,None)),
'identical-boundary-phases':lambda g:g.set((E.pourBoundary,S.afterPhase,E.pourTransfer)),
'foreign-boundary-phase':lambda g:g.set((E.pourBoundary,S.afterPhase,E.graspVerify)),
'expectation-without-modality':lambda g:g.remove((E.pourSensoryExpectation,S.modality,None)),
'controlpoint-without-observation-capability':lambda g:g.remove((E.pourControlPoint,S.observationRequirement,None)),
'mismatch-without-criterion':lambda g:g.remove((E.pourMismatch,S.discrepancyCondition,None)),
}
for n,mutate in mutations.items():
 g=Graph()+data;mutate(g);ok,_,_=check(g);record('reject-'+n,not ok)
record('qualitative-strength-remains-unknown',data.value(E.pourForcePattern,F.relativeStrength)==F.UnknownStrength)
record('no-observation-asserted-by-expectation',not any(data.subjects(RDF.type,Namespace('http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#').Event)))
record('new-claims-proposed',all(data.value(E[p+'RemainingClaim'+str(i)],A.hasStatus)==A.Proposed for p in ['pour','grasp'] for i in range(4)))
record('shared-entity-not-mutated-with-material-links',not list(data.objects(E.transferableSubstance,M.hasMaterial)))
(ROOT/'reports/completion-validation.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps({'checks':len(results),'passed':sum(r['passed'] for r in results),'failures':[r for r in results if not r['passed']]},indent=2));assert all(r['passed'] for r in results)
