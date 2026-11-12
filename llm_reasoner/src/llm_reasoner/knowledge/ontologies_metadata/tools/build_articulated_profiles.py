"""Rebuild only the eight synthetic Proposed mechanism example records.

Ontology/shapes are maintained separately. No source input or older example is edited.
"""
from pathlib import Path
from rdflib import Graph,Namespace,RDF,RDFS,Literal,XSD
from ontology_imports import ROOT
R=ROOT;B='http://aicor.knowledge/ontologies/'
A=Namespace(B+'action-description-core.owl#');N=Namespace(B+'articulated-mechanisms.owl#');RM=Namespace(B+'robot-motion.owl#');OM=Namespace(B+'object-manipulation.owl#');P=Namespace(B+'participants-relations.owl#');C=Namespace(B+'conditions-outcomes.owl#');K=Namespace(B+'knowledge-context.owl#');T=Namespace(B+'phases-transitions.owl#');M=Namespace(B+'monitoring-requirements.owl#');E=Namespace(B+'articulated-examples.owl#')
g=Graph()
for prefix,ns in [('ac',A),('am',N),('rm',RM),('co',C),('pt',T),('mr',M),('ex',E)]:g.bind(prefix,ns)
def node(key,typ,label=None):
 n=E[key];g.add((n,RDF.type,typ))
 if label:g.add((n,RDFS.label,Literal(label,lang='en')))
 return n
def add(s,p,o):g.add((s,p,o));return s
def parameter(key,unit):
 n=node(key,A.ParameterSpecification);add(n,A.parameterKey,Literal(key,datatype=XSD.string));add(n,A.unitIRI,Literal('http://qudt.org/vocab/unit/'+unit,datatype=XSD.anyURI));return n
def atom(key,rel,args,scope):
 pat=node(key+'Pattern',P.RelationPattern);add(pat,P.usesRelation,rel)
 for i,arg in enumerate(args,1):
  binding=node(key+'Binding'+str(i),P.ArgumentBinding);add(pat,P.hasArgumentBinding,binding);add(binding,P.bindsArgument,N[str(rel).split('#')[-1]+'Argument'+str(i)]);add(binding,P.bindsEntity,arg)
 n=node(key,C.AtomicCondition);add(n,C.hasPattern,pat);add(n,C.inScope,scope);return n
def conjunction(key,expressions):
 if len(expressions)==1:return expressions[0]
 n=node(key,C.Conjunction)
 for e in expressions:add(n,C.hasOperand,e)
 return n
def condition(key,typ,expression):return add(node(key,typ),C.hasExpression,expression)
def comparison(key,error,tol,scope):
 n=node(key,C.Comparison);add(n,C.operator,C.LessOrEqual)
 for prop,p,suffix,sc in [(C.leftOperand,error,'ErrorRef',scope),(C.rightOperand,tol,'ToleranceRef',C.TaskContext)]:
  ref=node(key+suffix,C.ParameterReference);add(ref,C.parameter,p);add(ref,C.inScope,sc);add(n,prop,ref)
 return n
def monitor(key,host,cond,point):
 n=node(key,M.MonitoringRequirement);add(host,M.hasMonitoringRequirement,n);add(n,M.monitorsCondition,cond);add(n,M.atPoint,point)
 for p,o in [(M.whenTrue,M.ReportSatisfied),(M.whenFalse,M.RequestSuspension),(M.whenUnknown,M.RequestEvidence)]:add(n,p,o)
 from temporal_runtime import dependencies
 for i,(kind,target,scope) in enumerate(sorted(dependencies(g,g.value(cond,C.hasExpression)),key=str)):
  obs=node(key+'Observation'+str(i),M.RelationObservation if kind=='relation' else M.QuantityObservation);add(n,M.requiresObservation,obs);add(obs,M.inScope,scope);add(obs,M.observesPattern if kind=='relation' else M.observesParameter,target)
 return n
profiles=[('revoluteConfiguration',N.ConfigurationChangeDescription,'angular'),('prismaticConfiguration',N.ConfigurationChangeDescription,'linear'),('revoluteOpening',N.OpeningDescription,'angular'),('prismaticOpening',N.OpeningDescription,'linear'),('revoluteClosing',N.ClosingDescription,'angular'),('prismaticClosing',N.ClosingDescription,'linear'),('releaseLatch',N.LatchReleaseDescription,'angular'),('secureLatch',N.LatchSecuringDescription,'angular')]
for name,typ,kind in profiles:
 action=node(name,typ,name);assembly=node(name+'Assembly',A.EntitySpecification);fixed=node(name+'FixedPart',A.EntitySpecification);moving=node(name+'MovingPart',A.EntitySpecification);operator=node(name+'Operator',A.EntitySpecification);frame=node(name+'Frame',RM.FrameSpecification);mech=node(name+'Mechanism',N.MechanismSpecification);joint=node(name+'Joint',N.JointSpecification);coordinate=node(name+'Coordinate',RM.CoordinateSpecification);axis=node(name+'Axis',N.AxisSpecification);origin=node(name+'Origin',RM.PositionTarget);handle=node(name+'Handle',A.EntitySpecification);latch=node(name+'Latch',N.LatchSpecification)
 for p,o in [(N.mechanism,mech),(N.operator,operator)]:add(action,p,o)
 for p,o in [(N.describesAssembly,assembly),(N.fixedPart,fixed),(N.movingPart,moving),(N.joint,joint),(N.referenceFrame,frame),(N.handle,handle),(N.latch,latch)]:add(mech,p,o)
 add(handle,N.handleOf,moving);add(latch,N.latchOf,mech)
 add(joint,N.jointKind,N.Revolute if kind=='angular' else N.Prismatic);add(joint,N.coordinate,coordinate);add(coordinate,RM.coordinateKind,Literal(kind,datatype=XSD.string));add(coordinate,RM.coordinateKey,Literal(name+'-q',datatype=XSD.string));add(joint,N.axis,axis);add(joint,N.openingDirection,Literal(-1 if name=='prismaticOpening' else 1,datatype=XSD.integer));add(axis,N.axisFrame,frame);add(axis,N.axisOrigin,origin)
 for dim in 'xyz':add(axis,N[dim+'AxisParameter'],parameter(name+'Axis'+dim,'UNITLESS'));add(origin,RM[dim+'Parameter'],parameter(name+'Origin'+dim,'M'))
 unit='RAD' if kind=='angular' else 'M'
 for p in ['lowerLimit','upperLimit','openCoordinate','closedCoordinate']:add(joint,N[p],parameter(name+p[0].upper()+p[1:],unit))
 context=node(name+'Context',K.KnowledgeContext);add(context,K.contextKey,Literal('synthetic-'+name,datatype=XSD.string));add(context,K.scopedDescription,action);add(action,N.evaluationContext,context)
 cap=node(name+'ActuationCapability',A.CapabilityRequirement);add(action,A.hasCapabilityRequirement,cap)
 observation_cap=node(name+'AssessmentCapability',A.CapabilityRequirement);add(action,A.hasCapabilityRequirement,observation_cap)
 is_config=typ in [N.ConfigurationChangeDescription,N.OpeningDescription,N.ClosingDescription]
 action_kind={N.ConfigurationChangeDescription:N.ConfiguringMechanism,N.OpeningDescription:N.OpeningMechanism,N.ClosingDescription:N.ClosingMechanism,N.LatchReleaseDescription:N.ReleasingLatch,N.LatchSecuringDescription:N.SecuringLatch}[typ];add(action,A.hasActionKind,action_kind)
 prereq=[atom(name+'Ready',N.MechanismReady,[operator,assembly],C.PreState)]
 if is_config:prereq.append(atom(name+'Clear',N.PathClear,[operator,assembly],C.PreState))
 # Latch presence alone does not require release for every mechanism technique.
 release_required=is_config and typ!=N.ClosingDescription
 if is_config:add(action,N.latchReleaseRequired,Literal(release_required,datatype=XSD.boolean))
 if release_required:prereq.append(atom(name+'ReleasedBefore',N.LatchReleased,[latch],C.PreState))
 if typ==N.LatchSecuringDescription:prereq.append(atom(name+'ClosedBefore',N.ClosedConfiguration,[assembly,moving],C.PreState))
 pre=condition(name+'Precondition',A.PreconditionSpecification,conjunction(name+'PreExpression',prereq));add(action,A.hasPrecondition,pre)
 error=tol=target=None
 if is_config:
  target=g.value(joint,N.openCoordinate) if typ==N.OpeningDescription else g.value(joint,N.closedCoordinate) if typ==N.ClosingDescription else parameter(name+'Target',unit)
  error=parameter(name+'Error',unit);tol=parameter(name+'Tolerance',unit);add(error,N.errorCoordinate,coordinate);add(error,N.errorTarget,target)
  for p,o in [(N.targetParameter,target),(N.errorParameter,error),(N.toleranceParameter,tol)]:add(action,p,o)
 def success_expr(scope,tag):
  parts=[]
  if is_config:
   parts=[comparison(name+tag+'ErrorWithin',error,tol,scope)]
   if typ in [N.OpeningDescription,N.ClosingDescription]:parts.append(atom(name+tag+'State',N.OpenConfiguration if typ==N.OpeningDescription else N.ClosedConfiguration,[assembly,moving],scope))
  else:
   parts=[atom(name+tag+'Latch',N.LatchReleased if typ==N.LatchReleaseDescription else N.LatchSecured,[latch],scope)]
   if typ==N.LatchSecuringDescription:parts.append(atom(name+tag+'Closed',N.ClosedConfiguration,[assembly,moving],scope))
  return conjunction(name+tag+'Expression',parts)
 success=condition(name+'Success',A.SuccessCriterion,success_expr(C.PostState,'Post'));add(action,A.hasSuccessCriterion,success)
 goal=condition(name+'Goal',A.GoalSpecification,g.value(success,C.hasExpression));add(action,A.hasGoal,goal)
 fault=condition(name+'Fault',A.FailureCriterion,atom(name+'FaultCurrent',N.MechanismFault,[operator,assembly],C.CurrentState));add(action,A.hasFailureCriterion,fault)
 structure=node(name+'Phases',T.PhaseStructure);add(action,T.hasPhaseStructure,structure)
 names=['Prepare','Engage','Move','Verify','Complete','Abort'] if is_config else ['Prepare','Operate','Verify','Complete','Abort']
 phases={key:node(name+key,T.SuccessfulTerminalPhase if key=='Complete' else T.FailedTerminalPhase if key=='Abort' else A.PhaseSpecification,key) for key in names}
 add(structure,T.initialPhase,phases['Prepare']);add(action,N.actuationPhase,phases['Move'] if is_config else phases['Operate']);add(action,N.verificationPhase,phases['Verify'])
 if is_config:add(action,N.engagementPhase,phases['Engage'])
 for phase in phases.values():add(structure,T.hasMemberPhase,phase);add(action,A.hasPhase,phase)
 current=condition(name+'CurrentTarget',A.ConditionSpecification,success_expr(C.CurrentState,'Current'))
 engaged=None
 if is_config:
  engagement=[atom(name+'EngagedCurrent',N.ActuationEngaged,[operator,assembly],C.CurrentState)]
  if release_required:engagement.append(atom(name+'ReleasedCurrent',N.LatchReleased,[latch],C.CurrentState))
  engagement.append(atom(name+'ClearCurrent',N.PathClear,[operator,assembly],C.CurrentState))
  engaged=condition(name+'EngagementCondition',A.ConditionSpecification,conjunction(name+'EngagementExpression',engagement))
 edges=[('Prepare','Engage',pre),('Engage','Move',engaged),('Move','Verify',current),('Verify','Complete',success)] if is_config else [('Prepare','Operate',pre),('Operate','Verify',current),('Verify','Complete',success)]
 for i,(src,dst,guard) in enumerate(edges):
  edge=node(name+'Transition'+str(i),T.OrdinaryTransition)
  for p,o in [(T.fromPhase,phases[src]),(T.toPhase,phases[dst]),(T.guardCondition,guard)]:add(edge,p,o)
  add(structure,T.hasTransition,edge);monitor(name+'GuardMonitor'+str(i),edge,guard,M.TransitionCheck)
 for src in names[:-2]:
  edge=node(name+src+'FaultTransition',T.FailureTransition)
  for p,o in [(T.fromPhase,phases[src]),(T.toPhase,phases['Abort']),(T.guardCondition,fault)]:add(edge,p,o)
  add(structure,T.hasTransition,edge)
 if is_config:
  add(phases['Move'],T.invariantCondition,engaged);monitor(name+'EngagementMonitor',phases['Move'],engaged,M.DuringPhase)
 monitor(name+'VerificationMonitor',phases['Verify'],success,M.PhaseExit)
 # Snapshot each profile's entire reachable descriptive graph before metadata.
 seen=set();todo=[action]
 while todo:
  x=todo.pop()
  if x in seen:continue
  seen.add(x);todo.extend(o for o in g.objects(x) if isinstance(o,type(action)) and str(o).startswith(str(E)))
 triples={(s,p,o) for s in seen for p,o in g.predicate_objects(s) if p not in [RDF.type,RDFS.label,RDFS.comment]}
 record=node(name+'Record',A.KnowledgeRecord);annotation=node(name+'Annotation',A.AnnotationRecord);evidence=node(name+'Evidence',A.EvidenceRecord);bundle=node(name+'Bundle',A.ActionComponent)
 for p,o in [(A.expressesDescription,action),(A.recordVersion,Literal('synthetic-profile-1',datatype=XSD.string)),(A.hasAnnotation,annotation)]:add(record,p,o)
 add(annotation,A.annotatesDescription,action);add(annotation,A.producerIdentifier,Literal('curated-articulated-schema-fixture',datatype=XSD.string));add(evidence,A.evidenceKind,A.CuratedExample);add(evidence,A.evidenceText,Literal('Synthetic Proposed engineering profile. No physical body, observation, measured limit or executable controller.',datatype=XSD.string))
 for i,(s,p,o) in enumerate(sorted(triples,key=lambda t:tuple(map(str,t)))):
  claim=node(name+'Claim'+str(i),A.KnowledgeClaim);st=node(name+'Statement'+str(i),A.Statement)
  for subject,predicate,value in [(record,A.hasClaim,claim),(annotation,A.contributesClaim,claim),(claim,A.aboutComponent,bundle),(claim,A.hasStatus,A.Proposed),(claim,A.hasEvidence,evidence),(claim,A.hasStatement,st),(st,A.statementSubject,s),(st,A.statementPredicateIRI,Literal(str(p),datatype=XSD.anyURI)),(st,A.statementLiteral if isinstance(o,Literal) else A.statementObject,o)]:add(subject,predicate,value)
g.serialize(R/'examples/articulated-mechanism-profiles.ttl',format='turtle')
print({'profiles':len(profiles),'triples':len(g)})
