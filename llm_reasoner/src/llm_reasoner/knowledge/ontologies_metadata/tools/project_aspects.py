"""Read-only inspectable projections; preserve proposed status, scope and references."""
from pathlib import Path
import json
from rdflib import Graph, Namespace, RDF
from aspect_runtime import A,P,C,KINDS,one
R=Path(__file__).resolve().parents[1];E=Namespace('http://aicor.knowledge/ontologies/action-examples.owl#')
g=Graph()
for name in ['examples/stereotypical-actions.ttl','examples/aspect-actions.ttl','aspects/structure/participants-relations.ttl']:g+=Graph().parse(R/name)
def short(n):return str(n).split('#')[-1]
def expression(n):
 kind=next(k for k in KINDS if (n,RDF.type,k) in g);out={'id':str(n),'type':short(kind)}
 if kind in [C.Conjunction,C.Disjunction,C.Negation]:out['operands']=[expression(o) for o in sorted(g.objects(n,C.hasOperand))]
 elif kind==C.AtomicCondition:
  pattern=one(g,n,C.hasPattern);out.update(scope=short(one(g,n,C.inScope)),predicate=str(one(g,pattern,P.usesRelation)))
  args=[]
  for b in g.objects(pattern,P.hasArgumentBinding):
   arg=one(g,b,P.bindsArgument);args.append({'position':int(one(g,arg,P.argumentPosition)),'argument':str(arg),'entitySpecification':str(one(g,b,P.bindsEntity))})
  out['arguments']=sorted(args,key=lambda a:a['position'])
 elif kind==C.Comparison:out.update(operator=short(one(g,n,C.operator)),left=expression(one(g,n,C.leftOperand)),right=expression(one(g,n,C.rightOperand)))
 elif kind==C.ParameterReference:out.update(parameter=str(one(g,n,C.parameter)),scope=short(one(g,n,C.inScope)),value=None)
 elif kind==C.QuantityConstant:out.update(value=str(one(g,n,C.decimalValue)),unit=str(one(g,n,C.unitIRI)))
 return out
outputs=[]
for record in [E.pourRecordV2,E.graspRecordV2]:
 action=one(g,record,A.expressesDescription);out={'record':str(record),'version':str(one(g,record,A.recordVersion)),'description':str(action),'kind':str(one(g,action,A.hasActionKind)),'roles':[],'conditions':[]}
 for role in sorted(g.objects(action,A.hasRoleSpecification)):out['roles'].append({'role':str(one(g,role,A.specifiesRole)),'entitySpecification':str(one(g,role,A.hasEntitySpecification))})
 for predicate in [A.hasPrecondition,A.hasGoal,A.hasSuccessCriterion,A.hasFailureCriterion]:
  for cond in sorted(g.objects(action,predicate)):
   claims=[c for c in g.objects(record,A.hasClaim) if (c,A.aboutComponent,cond) in g]
   out['conditions'].append({'relationship':short(predicate),'condition':str(cond),'expression':expression(one(g,cond,C.hasExpression)),'claims':[{'id':str(c),'status':short(one(g,c,A.hasStatus)),'evidence':[str(e) for e in g.objects(c,A.hasEvidence)]} for c in sorted(claims)]})
 outputs.append(out)
(R/'examples/aspect-projections.json').write_text(json.dumps(outputs,indent=2)+'\n')
rows=list(g.query((R/'queries/aspect-conditions.rq').read_text()));assert len(rows)==8
print('Produced two aspect projections; competency query returns 8 evidence-bearing conditions.')
