"""Checks links in the bounded partial-transfer example contract, not world truth."""
from rdflib import Namespace,RDF
from aspect_runtime import A,C,P,one
T=Namespace('http://aicor.knowledge/ontologies/transfer-contracts.owl#')
DB=Namespace('http://aicor.knowledge/ontologies/diagnostic-causal-bridge.owl#')
FD=Namespace('http://aicor.knowledge/ontologies/failure-diagnosis.owl#')
CM=Namespace('http://aicor.knowledge/ontologies/causal-mechanisms.owl#')
def transfer_errors(g):
 errors=[]
 for contract in g.subjects(RDF.type,T.TransferContract):
  source=one(g,contract,T.source);destination=one(g,contract,T.destination);portion=one(g,contract,T.transferredPortion)
  if source==destination:errors.append('source and destination collapsed')
  for action in g.subjects(T.hasTransferContract,contract):
   entities={e for role in g.objects(action,A.hasRoleSpecification) for e in g.objects(role,A.hasEntitySpecification)}
   if not {source,destination,one(g,contract,T.substance)}<=entities:errors.append('transfer referent absent from action participants')
  criterion=one(g,contract,T.successCriterion);expression=one(g,criterion,C.hasExpression)
  nodes=set();todo=[expression]
  while todo:
   node=todo.pop()
   if node in nodes:continue
   nodes.add(node)
   for p in [C.hasOperand,C.leftOperand,C.rightOperand]:todo.extend(g.objects(node,p))
  references={p for n in nodes for p in g.objects(n,C.parameter)}
  required={one(g,contract,p) for p in [T.deliveredAmount,T.targetAmount,T.spillAmount,T.spillTolerance]}
  if not required<=references:errors.append('success criterion omits bound transfer quantity')
  matches=False
  for node in nodes:
   if (node,RDF.type,C.AtomicCondition) not in g or g.value(node,C.inScope)!=C.PostState:continue
   pattern=one(g,node,C.hasPattern)
   if g.value(pattern,P.usesRelation)!=P.ContainedIn:continue
   bindings={g.value(b,P.bindsArgument):g.value(b,P.bindsEntity) for b in g.objects(pattern,P.hasArgumentBinding)}
   if bindings.get(P.ContainedInArgument1)==portion and bindings.get(P.ContainedInArgument2)==destination:matches=True
  if not matches:errors.append('success criterion omits intended portion/destination containment')
 for association in g.subjects(RDF.type,DB.DiagnosticCausalAssociation):
  hypothesis=one(g,association,DB.hypothesis);link=one(g,association,DB.causalLink)
  failures={g.value(f,FD.detectedBy) for f in g.objects(hypothesis,FD.explainsFailure)}
  if one(g,link,CM.consequent) not in failures:errors.append('diagnostic causal consequent does not match detected failure')
 return errors
