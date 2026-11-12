from pathlib import Path
import json
from rdflib import Graph,Namespace,RDF
from aspect_runtime import A
B='http://aicor.knowledge/ontologies/';K=Namespace(B+'causal-mechanisms.owl#');F=Namespace(B+'physical-models.owl#');E=Namespace(B+'action-examples.owl#');R=Path(__file__).resolve().parents[1]
g=Graph()
for n in ['stereotypical-actions','aspect-actions','phase-actions','causal-physical-actions']:g+=Graph().parse(R/'examples'/(n+'.ttl'))
def val(n,p):v=g.value(n,p);return str(v) if v is not None else None
def refs(n,p):return sorted(str(v) for v in g.objects(n,p))
views=[]
for record in [E.pourRecordV4,E.graspRecordV4]:
 action=g.value(record,A.expressesDescription);out={'record':str(record),'description':str(action),'causalModels':[],'physicalModels':[],'properties':[],'claims':[]}
 for model in g.objects(action,A.hasCausalModel):
  out['causalModels'].append({'id':str(model),'links':[{'id':str(link),'antecedent':val(link,K.antecedent),'consequent':val(link,K.consequent),'mechanism':val(link,K.mechanism),'phase':val(link,K.appliesDuring)} for link in g.objects(model,K.hasLink)]})
 for model in g.objects(action,A.hasPhysicalModel):
  out['physicalModels'].append({'id':str(model),'assumptions':refs(model,F.assumption),'applicability':refs(model,F.applicability),'mechanisms':refs(model,F.realizesMechanism),'variables':[{'id':str(v),'symbol':val(v,F.symbol),'parameter':val(v,F.parameter)} for v in g.objects(model,F.hasVariable)],'equations':[{'text':val(e,F.expressionText),'language':val(e,F.expressionLanguage),'variables':refs(e,F.usesVariable)} for e in g.objects(model,F.hasEquation)]})
 for p in g.objects(action,F.hasPhysicalProperty):out['properties'].append({'id':str(p),'entity':val(p,F.describesEntity),'kind':val(p,F.propertyKind),'parameter':val(p,F.valueParameter),'value':val(g.value(p,F.valueParameter),A.numericValue)})
 for claim in g.objects(record,A.hasClaim):
  if 'PhysicalClaim' in str(claim):out['claims'].append({'id':str(claim),'status':val(claim,A.hasStatus),'evidence':refs(claim,A.hasEvidence),'applicability':refs(claim,A.hasApplicability)})
 views.append(out)
assert len(views)==2 and sum(len(v['claims']) for v in views)==8
(R/'examples/causal-physical-projections.json').write_text(json.dumps(views,indent=2)+'\n');print('Produced two model views with eight proposed claims.')
