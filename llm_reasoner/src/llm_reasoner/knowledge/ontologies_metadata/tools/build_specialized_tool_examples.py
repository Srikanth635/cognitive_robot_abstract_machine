"""Build four synthetic tool refinements without editing the original tool fixture."""
from rdflib import Graph,Namespace,RDF,Literal,XSD
from ontology_imports import ROOT
from aspect_runtime import A,C,P
from authoring_helpers import Fixture,clone_description
from context_review import K
B='http://aicor.knowledge/ontologies/';S=Namespace(B+'specialized-tool-interactions.owl#');TI=Namespace(B+'tool-interaction.owl#');E=Namespace(B+'specialized-tool-examples.owl#')
source=Graph().parse(ROOT/'examples/tool-material-transfer-profiles.ttl');OLD=Namespace(B+'tool-transfer-examples.owl#');g=Graph();g.bind('st',S);g.bind('ex',E);f=Fixture(g,E)
for name,cls,kind,rel in [('cutting',S.CuttingDescription,S.Cutting,S.SpecifiedSeparationAchieved),('wiping',S.WipingDescription,S.Wiping,S.SpecifiedSurfaceEffectAchieved),('pressing',S.PressingDescription,S.Pressing,S.SpecifiedActuationAchieved),('insertion',S.InsertionDescription,S.Inserting,S.SpecifiedSeatingAchieved)]:
 root=E[name];clone_description(source,g,OLD.toolUse,root,E,name+'Base');g.add((root,RDF.type,cls));g.set((root,A.hasActionKind,kind))
 operator=g.value(root,TI.operator);tool=g.value(root,TI.tool);part=g.value(root,TI.workingPart);target=g.value(root,TI.target);effect=g.value(root,TI.intendedEffect)
 g.set((root,TI.contactPolicy,Literal('required' if name in ['wiping','pressing'] else 'unspecified',datatype=XSD.string)))
 pre=g.value(g.value(root,A.hasPrecondition),C.hasExpression);extra=f.atom(name+'TechniqueReady',S.TechniqueReady,[tool,target,effect],C.TaskContext)
 g.set((g.value(root,A.hasPrecondition),C.hasExpression,f.combine(name+'ReadyExpression',C.Conjunction,[pre,extra])))
 success=g.value(g.value(root,A.hasSuccessCriterion),C.hasExpression)
 g.add((success,C.hasOperand,f.atom(name+'SpecificEffect',rel,[tool,target,effect] if name=='insertion' else [target,effect],C.PostState)))
 if name in ['wiping','pressing']:g.add((success,C.hasOperand,f.atom(name+'Contact',P.InContactWith,[part,target],C.PostState)))
 if name=='wiping':g.add((success,C.hasOperand,f.atom(name+'Sweep',S.SweepCoverageMaintained,[tool,target],C.PostState)))
 if name=='insertion':
  g.add((success,C.hasOperand,f.atom(name+'Alignment',S.AxisAlignmentSatisfied,[part,target],C.PostState)))
  depth=f.parameter(name+'TargetDepth','M');error=f.parameter(name+'DepthError','M');tol=f.parameter(name+'DepthTolerance','M')
  for p,o in [(S.targetDepth,depth),(S.depthError,error),(S.depthTolerance,tol)]:g.add((root,p,o));g.add((root,A.hasParameter,o))
  g.add((error,S.depthErrorTarget,depth));g.add((error,S.depthErrorFrame,g.value(root,TI.referenceFrame)))
  comparison=f.node(name+'DepthWithin',C.Comparison);f.add(comparison,C.operator,C.LessOrEqual)
  for p,q,scope,key in [(C.leftOperand,error,C.PostState,'ErrorRef'),(C.rightOperand,tol,C.TaskContext,'ToleranceRef')]:
   ref=f.node(name+key,C.ParameterReference);f.add(ref,C.parameter,q);f.add(ref,C.inScope,scope);f.add(comparison,p,ref)
  g.add((success,C.hasOperand,comparison))
 # Keep the goal aligned to the refined intended outcome as well.
 g.set((g.value(root,A.hasGoal),C.hasExpression,success))
 f.snapshot(root,name)
g.serialize(ROOT/'examples/specialized-tool-profiles.ttl',format='turtle');print({'profiles':4,'triples':len(g)})
