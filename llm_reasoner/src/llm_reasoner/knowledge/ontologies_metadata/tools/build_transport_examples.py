"""Rebuild only four synthetic carrying records from the preserved domain fixtures."""
from rdflib import Graph,Namespace,RDF,RDFS,Literal,XSD,URIRef
from ontology_imports import ROOT
R=ROOT;B='http://aicor.knowledge/ontologies/'
A=Namespace(B+'action-description-core.owl#');N=Namespace(B+'transport-carrying.owl#');OM=Namespace(B+'object-manipulation.owl#');MN=Namespace(B+'mobility-navigation.owl#');RM=Namespace(B+'robot-motion.owl#');K=Namespace(B+'knowledge-context.owl#');C=Namespace(B+'conditions-outcomes.owl#');P=Namespace(B+'participants-relations.owl#');T=Namespace(B+'phases-transitions.owl#');M=Namespace(B+'monitoring-requirements.owl#');E=Namespace(B+'transport-examples.owl#');SH=Namespace('http://www.w3.org/ns/shacl#')
# EXAMPLE BUILDER START
source=Graph()
for f in ['object-manipulation-profiles.ttl','mobility-navigation-profiles.ttl']:source.parse(R/'examples'/f)
g=Graph();g.bind('tc',N);g.bind('ex',E);g.bind('ac',A)
def node(name,typ):
 n=E[name];g.add((n,RDF.type,typ));return n
def add(s,p,o):g.add((s,p,o));return s
def param(name,unit):
 n=node(name,A.ParameterSpecification);add(n,A.parameterKey,Literal(name,datatype=XSD.string));add(n,A.unitIRI,Literal('http://qudt.org/vocab/unit/'+unit,datatype=XSD.anyURI));return n
def clone(oldroot,prefix,overrides):
 oldns=str(oldroot).split('#')[0]+'#';seen=set();todo=[oldroot]
 while todo:
  n=todo.pop()
  if n in seen:continue
  seen.add(n);todo.extend(o for o in source.objects(n) if isinstance(o,URIRef) and str(o).startswith(oldns))
 mapping={n:overrides.get(n,E[prefix+str(n).split('#')[-1]]) for n in seen}
 for n in seen:
  for p,o in source.predicate_objects(n):g.add((mapping[n],p,mapping.get(o,o)))
 return mapping
def atomic(name,rel,args,scope,namespace=N):
 pattern=node(name+'Pattern',P.RelationPattern);add(pattern,P.usesRelation,rel)
 for i,arg in enumerate(args,1):
  b=node(name+'Binding'+str(i),P.ArgumentBinding);add(pattern,P.hasArgumentBinding,b);add(b,P.bindsArgument,namespace[str(rel).split('#')[-1]+'Argument'+str(i)]);add(b,P.bindsEntity,arg)
 e=node(name,C.AtomicCondition);add(e,C.inScope,scope);add(e,C.hasPattern,pattern);return e
def expression(name,typ,parts):
 e=node(name,typ)
 for p in parts:add(e,C.hasOperand,p)
 return e
def cond(name,typ,e):return add(node(name,typ),C.hasExpression,e)
profiles=[('carrying',False,False),('routeCarrying',True,False),('orientationCarrying',False,True),('routeOrientationCarrying',True,True)]
for name,route,oriented in profiles:
 action=node(name,N.RouteCarryingDescription if route else N.CarryingDescription)
 carrier=node(name+'Carrier',A.EntitySpecification);load=node(name+'Load',A.EntitySpecification);man=node(name+'Manipulator',A.EntitySpecification)
 holdroot=E[name+'Holding'];navroot=E[name+'Mobility'];OMEX=Namespace(B+'object-manipulation-examples.owl#');MNEX=Namespace(B+'mobility-examples.owl#');oldnav='routeNavigation' if route else 'navigation'
 holdmap=clone(OMEX.holding,name+'Hold', {OMEX.holding:holdroot,OMEX.holdingManipulator:man,OMEX.holdingObject:load,OMEX.holdingContext:E[name+'HoldingContext']})
 navmap=clone(MNEX[oldnav],name+'Nav',{MNEX[oldnav]:navroot,MNEX[oldnav+'Robot']:carrier,MNEX[oldnav+'Target']:E[name+'Destination'],MNEX[oldnav+'Footprint']:E[name+'LoadedFootprint'],MNEX[oldnav+'Context']:E[name+'MobilityContext']})
 target=g.value(navroot,MN.targetRegion);foot=g.value(navroot,MN.footprint);frame=g.value(navroot,MN.referenceFrame)
 add(foot,RDF.type,N.LoadedFootprintSpecification);add(foot,N.footprintCarrier,carrier);add(foot,N.footprintLoad,load);add(man,N.manipulatorOf,carrier)
 context=node(name+'Context',K.KnowledgeContext);add(context,K.contextKey,Literal('synthetic-'+name,datatype=XSD.string));add(context,K.scopedDescription,action)
 for p,o in [(N.carrier,carrier),(N.load,load),(N.manipulator,man),(N.destination,target),(N.loadedFootprint,foot),(N.holdingRequirement,holdroot),(N.mobilityRequirement,navroot),(N.evaluationContext,context),(N.intervalPolicy,Literal('coextensive',datatype=XSD.string)),(A.hasActionKind,N.RouteCarrying if route else N.Carrying)]:add(action,p,o)
 for child in [holdroot,navroot]:
  add(action,A.hasSubActionDescription,child)
  for cap in g.objects(child,A.hasCapabilityRequirement):add(action,A.hasCapabilityRequirement,cap)
  for role in g.objects(child,A.hasRoleSpecification):add(action,A.hasRoleSpecification,role)
 assessment_cap=node(name+'CarryingAssessmentCapability',A.CapabilityRequirement);add(action,A.hasCapabilityRequirement,assessment_cap)
 localpre=cond(name+'LocalPre',A.ConditionSpecification,expression(name+'LocalPreExpression',C.Conjunction,[atomic(name+'EnvelopeQualified',N.LoadedEnvelopeQualified,[carrier,load,foot],C.TaskContext),atomic(name+'BorneBefore',N.LoadBorneBy,[carrier,load],C.PreState)]))
 successparts=[atomic(name+'EnvelopeMaintained',N.LoadEnvelopeMaintained,[carrier,load,foot],C.PostState),atomic(name+'BorneThroughout',N.LoadBorneThroughout,[carrier,load],C.PostState),atomic(name+'LoadArrived',MN.InRegion,[load,target],C.PostState,MN)]
 if oriented:
  requirement=node(name+'Orientation',N.OrientationRequirement);orientation=node(name+'OrientationTarget',RM.OrientationTarget);error=param(name+'MaxDeviation','RAD');tolerance=param(name+'DeviationTolerance','RAD')
  add(orientation,RM.orientationConvention,Literal('quaternion-xyzw',datatype=XSD.string))
  for component in ['qx','qy','qz','qw']:add(orientation,RM[component+'Parameter'],param(name+component,'UNITLESS'))
  for p,o in [(N.orientedLoad,load),(N.orientationFrame,frame),(N.orientationTarget,orientation),(N.maximumDeviation,error),(N.deviationTolerance,tolerance)]:add(requirement,p,o)
  add(action,N.orientationRequirement,requirement)
  compare=node(name+'DeviationWithin',C.Comparison);add(compare,C.operator,C.LessOrEqual)
  for p,q,scope,suffix in [(C.leftOperand,error,C.PostState,'DeviationRef'),(C.rightOperand,tolerance,C.TaskContext,'DeviationToleranceRef')]:
   ref=node(name+suffix,C.ParameterReference);add(ref,C.parameter,q);add(ref,C.inScope,scope);add(compare,p,ref)
  successparts.append(compare)
 local_success=cond(name+'LocalSuccess',A.ConditionSpecification,expression(name+'LocalSuccessExpression',C.Conjunction,successparts) if len(successparts)>1 else successparts[0])
 local_fault=cond(name+'LocalFault',A.ConditionSpecification,atomic(name+'LoadFault',N.LoadTransportFault,[carrier,load],C.CurrentState))
 for p,o in [(N.localPrecondition,localpre),(N.localSuccess,local_success),(N.localFailure,local_fault)]:add(action,p,o)
 for prop,typ,local,logical,label in [(A.hasPrecondition,A.PreconditionSpecification,localpre,C.Conjunction,'Pre'),(A.hasSuccessCriterion,A.SuccessCriterion,local_success,C.Conjunction,'Success'),(A.hasFailureCriterion,A.FailureCriterion,local_fault,C.Disjunction,'Failure')]:
  children=[g.value(holdroot,prop),g.value(navroot,prop),local];e=expression(name+label+'Expression',logical,[g.value(c,C.hasExpression) for c in children]);add(action,prop,cond(name+label,typ,e))
 add(action,A.hasGoal,cond(name+'Goal',A.GoalSpecification,g.value(g.value(action,A.hasSuccessCriterion),C.hasExpression)))
 # A composition-scoped monitor attaches to the cloned mobility execution phase.
 phase=navmap[MNEX[oldnav+'Execute']];add(action,N.motionPhase,phase)
 current=cond(name+'CurrentLoadControl',A.ConditionSpecification,expression(name+'CurrentControlExpression',C.Conjunction,[atomic(name+'CurrentHold',OM.Holds,[man,load],C.CurrentState,OM),atomic(name+'CurrentEnvelopeClear',N.LoadEnvelopeClear,[carrier,load,foot],C.CurrentState),atomic(name+'CurrentLoadBorne',N.LoadBorneBy,[carrier,load],C.CurrentState)]))
 monitor=node(name+'LoadMonitor',M.MonitoringRequirement);add(phase,M.hasMonitoringRequirement,monitor);add(monitor,N.monitorContext,context);add(monitor,M.monitorsCondition,current);add(monitor,M.atPoint,M.DuringPhase)
 for p,o in [(M.whenTrue,M.ReportSatisfied),(M.whenFalse,M.RequestSuspension),(M.whenUnknown,M.RequestSuspension)]:add(monitor,p,o)
 from temporal_runtime import dependencies
 for i,(kind,value,scope) in enumerate(sorted(dependencies(g,g.value(current,C.hasExpression)),key=str)):
  obs=node(name+'LoadObservation'+str(i),M.RelationObservation);add(obs,M.observesPattern,value);add(obs,M.inScope,scope);add(monitor,M.requiresObservation,obs);add(obs,M.requiresCapability,assessment_cap)
 # Root snapshot includes reachable child descriptions, not independent mutable stores.
 seen=set();todo=[action]
 while todo:
  n=todo.pop()
  if n in seen:continue
  seen.add(n);todo.extend(o for o in g.objects(n) if isinstance(o,URIRef) and str(o).startswith(str(E)))
 for parameter in list(seen):
  if (parameter,RDF.type,A.ParameterSpecification) in g:add(action,A.hasParameter,parameter)
 triples={(sub,pred,obj) for sub in seen for pred,obj in g.predicate_objects(sub) if pred not in [RDF.type,RDFS.label,RDFS.comment]}
 record=node(name+'Record',A.KnowledgeRecord);annotation=node(name+'Annotation',A.AnnotationRecord);evidence=node(name+'Evidence',A.EvidenceRecord);bundle=node(name+'ClaimsBundle',A.ActionComponent)
 for p,o in [(A.expressesDescription,action),(A.hasAnnotation,annotation),(A.recordVersion,Literal('synthetic-composition-1',datatype=XSD.string))]:add(record,p,o)
 add(annotation,A.annotatesDescription,action);add(annotation,A.producerIdentifier,Literal('synthetic-carrying-composition',datatype=XSD.string));add(evidence,A.evidenceKind,A.CuratedExample);add(evidence,A.evidenceText,Literal('Synthetic Proposed composition of previous templates. No carried object, measured trajectory, validated footprint or accepted observation.',datatype=XSD.string))
 for i,(sub,pred,obj) in enumerate(sorted(triples,key=lambda x:tuple(map(str,x)))):
  claim=node(name+'Claim'+str(i),A.KnowledgeClaim);st=node(name+'Statement'+str(i),A.Statement)
  for x,p,o in [(record,A.hasClaim,claim),(annotation,A.contributesClaim,claim),(claim,A.aboutComponent,bundle),(claim,A.hasStatus,A.Proposed),(claim,A.hasEvidence,evidence),(claim,A.hasStatement,st),(st,A.statementSubject,sub),(st,A.statementPredicateIRI,Literal(str(pred),datatype=XSD.anyURI)),(st,A.statementLiteral if isinstance(obj,Literal) else A.statementObject,obj)]:add(x,p,o)
g.serialize(R/'examples/transport-carrying-profiles.ttl',format='turtle')

print({'records':len(profiles),'triples':len(g)})
