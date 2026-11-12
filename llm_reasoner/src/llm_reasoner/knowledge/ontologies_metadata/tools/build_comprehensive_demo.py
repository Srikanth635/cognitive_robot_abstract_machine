"""Build one explicitly fictional authoring illustration, not a projection/execution API."""
from pathlib import Path
import json
from rdflib import Graph,Namespace,RDF,RDFS,URIRef,Literal,XSD
from ontology_imports import ROOT
B='http://aicor.knowledge/ontologies/';A=Namespace(B+'action-description-core.owl#');E=Namespace(B+'action-examples.owl#');S=Namespace(B+'tool-transfer-examples.owl#');D=Namespace(B+'comprehensive-pouring-demo.owl#');MT=Namespace(B+'material-transfer.owl#');TI=Namespace(B+'tool-interaction.owl#');TC=Namespace(B+'transfer-contracts.owl#');K=Namespace(B+'knowledge-context.owl#');RM=Namespace(B+'robot-motion.owl#');PM=Namespace(B+'physical-models.owl#');EMA=Namespace(B+'entities-materials-affordances.owl#');DC=Namespace(B+'dynamics-control.owl#');C=Namespace(B+'conditions-outcomes.owl#')
source=Graph()
for p in (ROOT/'examples').glob('*.ttl'):source.parse(p)
g=Graph();g.bind('demo',D);g.bind('ac',A);g.bind('mt',MT);g.bind('ti',TI)
def clone(root,old_namespace,overrides):
 seen=set();todo=[root]
 while todo:
  node=todo.pop()
  if node in seen:continue
  seen.add(node);todo.extend(o for o in source.objects(node) if isinstance(o,URIRef) and str(o).startswith(str(old_namespace)))
 mapping={node:overrides.get(node,D[str(node).split('#')[-1].removesuffix('V8')]) for node in seen}
 for node in seen:
  for p,o in source.predicate_objects(node):g.add((mapping[node],p,mapping.get(o,o)))
 return mapping
mapping=clone(E.pourActionV8,E,{E.pourActionV8:D.pouring,E.sourceContainerV8:D.sourceBottle,E.receivingContainerV8:D.destinationGlass,E.transferableSubstanceV8:D.water,E.transferredPortionV8:D.transferredWater,E.Pouring:MT.Pouring})
def old(name):return mapping[E[name+'V8']]
toolmap=clone(S.toolUse,S,{S.toolUse:D.toolUse,S.toolUseOperator:D.robotEndEffector,S.toolUseTool:D.sourceBottle,S.toolUseWorkingpart:D.bottleMouth,S.toolUseTarget:D.destinationGlass,S.toolUseIntendedeffect:D.flowIntoGlass})
# Profiles, never physical occurrences or world bindings.
g.add((D.pouring,RDF.type,MT.PouringDescription));g.set((D.pouring,A.hasActionKind,MT.Pouring));g.add((D.pouring,MT.quantityBasis,Literal('volume',datatype=XSD.string)));g.add((D.pouring,MT.usesToolInteraction,D.toolUse))
for t in [(D.pouringContext,RDF.type,K.KnowledgeContext),(D.pouringContext,K.contextKey,Literal('fictional-pouring-illustration',datatype=XSD.string)),(D.pouringContext,K.scopedDescription,D.pouring),(D.pouring,MT.evaluationContext,D.pouringContext)]:g.add(t)
# Reuse one illustrative frame across source and tool descriptions.
frame=g.value(D.toolUse,TI.referenceFrame);g.set((D.toolUse,TI.referenceFrame,D.demoFrame));g.add((D.demoFrame,RDF.type,RM.FrameSpecification));g.add((D.demoFrame,RDFS.label,Literal('Fictional task frame; no world transform bound',lang='en')))
values={'sourceBefore':('0.000600','http://qudt.org/vocab/unit/M3'),'sourceAfter':('0.000347','http://qudt.org/vocab/unit/M3'),'deliveredAmount':('0.000252','http://qudt.org/vocab/unit/M3'),'targetAmount':('0.000250','http://qudt.org/vocab/unit/M3'),'spillAmount':('0.000001','http://qudt.org/vocab/unit/M3'),'spillTolerance':('0.000005','http://qudt.org/vocab/unit/M3'),'pourPropertyParameter':('998','http://qudt.org/vocab/unit/KiloGM-PER-M3'),'pourSamplingPeriod':('0.02','http://qudt.org/vocab/unit/SEC'),'pourRetryBudget':('2','http://qudt.org/vocab/unit/UNITLESS'),'pourCompletionTime':('6.5','http://qudt.org/vocab/unit/SEC')}
for name,(value,unit) in values.items():g.set((old(name),A.numericValue,Literal(value,datatype=XSD.decimal)));g.set((old(name),A.unitIRI,Literal(unit,datatype=XSD.anyURI)))
def parameter(name,value,unit):
 p=D[name]
 for t in [(p,RDF.type,A.ParameterSpecification),(p,A.parameterKey,Literal(name,datatype=XSD.string)),(p,A.numericValue,Literal(str(value),datatype=XSD.decimal)),(p,A.unitIRI,Literal(unit,datatype=XSD.anyURI)),(D.pouring,A.hasParameter,p)]:g.add(t)
 return p
# Entity descriptions, properties and poses are fictional specifications.
for entity,label,material,state in [(D.sourceBottle,'Example bottle','Glass',EMA.Solid),(D.destinationGlass,'Example drinking glass','Glass',EMA.Solid),(D.water,'Example water portion','Water',EMA.Liquid)]:
 facet=old('pourEntityDescription') if entity==D.water else D[str(entity).split('#')[-1]+'Description'];mat=old('pourMaterialSpec') if entity==D.water else D[str(entity).split('#')[-1]+'Material']
 g.remove((mat,EMA.materialState,None));g.remove((mat,RDFS.label,None))
 for t in [(entity,RDFS.label,Literal(label,lang='en')),(facet,RDF.type,EMA.EntityDescription),(facet,EMA.describesEntity,entity),(facet,EMA.hasMaterial,mat),(mat,RDF.type,EMA.MaterialSpecification),(mat,EMA.materialState,state),(mat,RDFS.label,Literal(material,lang='en')),(D.pouring,A.hasComponent,facet)]:g.add(t)
for entity,name,kind,value,unit in [(D.sourceBottle,'BottleDryMass',PM.Mass,'0.42','http://qudt.org/vocab/unit/KiloGM'),(D.destinationGlass,'GlassCapacity',PM.Capacity,'0.0004','http://qudt.org/vocab/unit/M3'),(D.water,'WaterViscosity',PM.Viscosity,'0.0011','http://qudt.org/vocab/unit/PA-SEC')]:
 prop=D[name+'Property'];p=parameter(name,value,unit)
 for t in [(prop,RDF.type,PM.PhysicalPropertySpecification),(prop,PM.describesEntity,entity),(prop,PM.propertyKind,kind),(prop,PM.valueParameter,p),(D.pouring,PM.hasPhysicalProperty,prop)]:g.add(t)
settings={name:parameter(name,value,unit) for name,value,unit in [('PourTilt','1.15','http://qudt.org/vocab/unit/RAD'),('MaximumAngularSpeed','0.35','http://qudt.org/vocab/unit/RAD-PER-SEC'),('MaximumAcceleration','0.4','http://qudt.org/vocab/unit/M-PER-SEC2'),('HypotheticalForceLimit','8','http://qudt.org/vocab/unit/N')]}
g.add((D.toolUse,TI.forceLimit,settings['HypotheticalForceLimit']))
# Explicit pose target for the intended pour location; it is not an observed pose.
pose=D.pourPose;position=D.pourPosition
for t in [(pose,RDF.type,RM.PoseTarget),(pose,RM.referenceFrame,D.demoFrame),(pose,RM.position,position),(position,RDF.type,RM.PositionTarget),(D.pouring,A.hasComponent,pose)]:g.add(t)
for axis,value in [('x','0.62'),('y','-0.18'),('z','0.94')]:g.add((position,RM[axis+'Parameter'],parameter('PourPose'+axis,value,'http://qudt.org/vocab/unit/M')))
# Link an explanatory frame schema without generating a new working frame projection.
frameview=D.framePerspective
for t in [(frameview,RDF.type,A.PerspectiveDescription),(frameview,RDFS.label,Literal('Illustrative frame-schema reference',lang='en')),(frameview,RDFS.seeAlso,E.PouringFrame),(D.pouring,A.hasPerspective,frameview)]:g.add(t)
# Recreate record-level exact statements for the complete linked description snapshot.
metadata={node for cls in [A.KnowledgeRecord,A.AnnotationRecord,A.KnowledgeClaim,A.Statement] for node in g.subjects(RDF.type,cls)}
for node in metadata:g.remove((node,None,None))
triples=[t for t in g if t[1] not in [RDF.type,RDFS.label,RDFS.comment]]
record=D.record;annotation=D.annotation;evidence=D.fictionalEvidence
for t in [(record,RDF.type,A.KnowledgeRecord),(record,A.expressesDescription,D.pouring),(record,A.recordVersion,Literal('fictional-demo-1',datatype=XSD.string)),(record,A.hasAnnotation,annotation),(annotation,RDF.type,A.AnnotationRecord),(annotation,A.annotatesDescription,D.pouring),(annotation,A.producerIdentifier,Literal('fictional-comprehensive-pouring-example',datatype=XSD.string)),(evidence,RDF.type,A.EvidenceRecord),(evidence,A.evidenceKind,A.CuratedExample),(evidence,A.evidenceText,Literal('Every numerical value and assumed outcome is invented for illustration. No empirical evidence, execution, calibration or certified safe limit.',datatype=XSD.string))]:g.add(t)
bundle=D.illustrationBundle;g.add((bundle,RDF.type,A.ActionComponent))
for i,(s,p,o) in enumerate(sorted(triples,key=lambda t:tuple(map(str,t)))):
 claim=D['claim'+str(i)];st=D['statement'+str(i)]
 for t in [(record,A.hasClaim,claim),(annotation,A.contributesClaim,claim),(claim,RDF.type,A.KnowledgeClaim),(claim,A.hasStatus,A.Proposed),(claim,A.hasEvidence,evidence),(claim,A.aboutComponent,bundle),(claim,A.hasStatement,st),(st,RDF.type,A.Statement),(st,A.statementSubject,s),(st,A.statementPredicateIRI,Literal(str(p),datatype=XSD.anyURI)),(st,A.statementLiteral if isinstance(o,Literal) else A.statementObject,o)]:g.add(t)
g.serialize(ROOT/'demo/comprehensive-pouring.ttl',format='turtle')
# A one-off readable authoring summary, with RDF identifiers for every linked section.
def ref(node):return str(node)
def q(p):return {'parameter':str(p),'value':float(g.value(p,A.numericValue)) if g.value(p,A.numericValue) is not None else None,'unit':str(g.value(p,A.unitIRI)) if g.value(p,A.unitIRI) is not None else None}
view={'record':ref(record),'action':ref(D.pouring),'type':'Pouring','status':'Proposed','illustrationOnly':True,'worldBound':False,'executionReadinessAssessed':False,'evidence':{'id':ref(evidence),'kind':'CuratedExample','note':'All values and assumed outcomes are fictional, not validated limits.'},'context':{'id':ref(D.pouringContext),'frame':ref(D.demoFrame)},'entities':{'source':{'id':ref(D.sourceBottle),'label':'Bottle','material':'glass','dryMass':q(D.BottleDryMass),'initialContents':{'substance':ref(D.water),'volume':q(old('sourceBefore'))}},'substance':{'id':ref(D.water),'label':'Water','density':q(old('pourPropertyParameter')),'viscosity':q(D.WaterViscosity)},'destination':{'id':ref(D.destinationGlass),'label':'Glass','capacity':q(D.GlassCapacity)},'transferredPortion':ref(D.transferredWater),'operator':ref(D.robotEndEffector),'workingPart':ref(D.bottleMouth)},'transfer':{'contract':ref(old('pourTransferContract')),'mode':'partial','quantityBasis':'volume','target':q(old('targetAmount')),'spillageTolerance':q(old('spillTolerance')),'success':['delivered >= target','spilled <= tolerance','transferred portion contained in intended destination']},'toolInteraction':{'id':ref(D.toolUse),'tool':ref(D.sourceBottle),'workingPart':ref(D.bottleMouth),'target':ref(D.destinationGlass),'intendedEffect':ref(D.flowIntoGlass),'contactPolicy':'unspecified','hypotheticalForceLimit':q(settings['HypotheticalForceLimit'])},'motion':{'targetPose':ref(pose),'position':{a:q(D['PourPose'+a]) for a in ['x','y','z']},'tilt':q(settings['PourTilt']),'maximumAngularSpeed':q(settings['MaximumAngularSpeed']),'maximumAcceleration':q(settings['MaximumAcceleration'])},'knowledgeAspects':{'preconditions':{'id':ref(old('pourPrecondition')),'summary':['source contains the substance','destination accessible','tool functional/held as separately specified']},'phases':{'id':ref(old('pourPhaseStructure')),'members':[ref(o) for o in sorted(g.objects(D.pouring,A.hasPhase))]},'monitoring':{'controlPoint':ref(old('pourControlPoint')),'unknownPolicy':'RequestEvidence; do not assume success'},'causality':{'model':ref(old('pourCausalModel')),'mechanism':ref(old('pourMechanism')),'status':'Proposed; no causal probability invented'},'physics':{'model':ref(old('pourPhysicalModel')),'accountingContract':ref(old('volumeAccountingContract')),'equation':'V_after = V_before - V_delivered - V_spilled','assumptions':['incompressible approximation','all source losses represented as delivered or spilled'],'physicalValidity':'not assessed'},'control':{'id':ref(old('pourController')),'kind':'FeedbackControl','samplingPeriod':q(old('pourSamplingPeriod')),'missing':['state evolution equation','output equation','calibrated actuator model'],'controllerSynthesized':False},'objectives':{'id':ref(old('pourObjectives')),'policy':'Lexicographic','priorities':['minimize spillage','minimize completion time']},'capabilities':{'id':ref(old('pourCapabilityProfile')),'requirements':[ref(o) for o in sorted(g.objects(old('pourCapabilityProfile'),Namespace(B+'capabilities-strategies.owl#').requiresCapability))],'providersMatched':False},'strategy':{'id':ref(old('pourExecutionStrategy')),'summary':['perform transfer','verify transfer outcome'],'executable':False},'diagnosis':{'association':ref(old('spillDiagnosticAssociation')),'status':'Proposed candidate explanation, not proven root cause'},'recovery':{'id':ref(old('pourRecoveryOption')),'retryBudget':q(old('pourRetryBudget')),'strategy':ref(old('pourRecoveryStrategy')),'executed':False},'framePerspective':{'id':ref(frameview),'schemaReference':ref(E.PouringFrame),'note':'Illustrative slot summary, no new frame projection generated','slots':{'source':ref(D.sourceBottle),'substance':ref(D.water),'destination':ref(D.destinationGlass)}},'forceDynamics':{'id':ref(old('pourForcePattern')),'relativeStrength':'UnknownStrength','note':'No quantitative physics inferred from qualitative roles'},'imageSchemas':[ref(old('pourContainmentSchema')),ref(old('pourPathSchema'))]},'hypothesizedOutcome':{'notAnObservation':True,'delivered':q(old('deliveredAmount')),'spilled':q(old('spillAmount')),'sourceRemaining':q(old('sourceAfter')),'assumedDestinationContainment':True,'illustrativeAccounting':'600 mL = 347 mL + 252 mL + 1 mL'},'projections':{'CoraPlexExecutable':None,'reason':'No runtime world/robot bindings or execution adapter supplied'}}
(ROOT/'demo/comprehensive-pouring.json').write_text(json.dumps(view,indent=2)+'\n')
print(json.dumps({'descriptiveStatements':len(triples),'record':str(record)}))
