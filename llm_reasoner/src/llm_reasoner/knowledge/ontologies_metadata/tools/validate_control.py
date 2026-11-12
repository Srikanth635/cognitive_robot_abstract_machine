from pathlib import Path
import json
from rdflib import Graph,Namespace,RDF,Literal,XSD
from pyshacl import validate
from control_runtime import A,F,O,S,control_errors,inspect_controller
R=Path(__file__).resolve().parents[1];E=Namespace('http://aicor.knowledge/ontologies/action-examples.owl#')
schema=Graph();data=Graph();shapes=Graph()
for f in [R/'core/action-description-core.ttl']+list((R/'aspects').rglob('*.ttl')):
 if not f.name.endswith('shapes.ttl'):schema+=Graph().parse(f)
for n in ['stereotypical-actions','aspect-actions','phase-actions','causal-physical-actions','control-actions']:data+=Graph().parse(R/'examples'/(n+'.ttl'))
for n in ['core/action-core-shapes.ttl','aspects/validation/aspect-shapes.ttl','aspects/validation/temporal-shapes.ttl','aspects/validation/causal-physical-shapes.ttl','aspects/validation/control-shapes.ttl']:shapes+=Graph().parse(R/n)
for path in (R/'integrations').glob('*.ttl'):schema += Graph().parse(path)
def check(g):
 valid,_,report=validate(g,shacl_graph=shapes,ont_graph=schema,inference='rdfs',advanced=True);errors=control_errors(g+schema);return bool(valid) and not errors,report,errors
results=[]
def record(n,ok):results.append({'case':n,'passed':bool(ok)})
ok,txt,errors=check(data);record('valid-two-partial-controller-examples',ok);(R/'reports/control-shacl.txt').write_text(txt+'\n'+str(errors))
def numeric(g,param,value,unit='http://qudt.org/vocab/unit/UNITLESS'):
 g.set((param,A.numericValue,Literal(value,datatype=XSD.decimal)));g.set((param,A.unitIRI,Literal(unit,datatype=XSD.anyURI)))
mutations={
'foreign-vector-variable':lambda g:g.set((E.pourStateAssignment,S.variable,E.graspInputVariable)),
'noncontiguous-position':lambda g:g.set((E.pourStateAssignment,S.position,Literal(2,datatype=XSD.positiveInteger))),
'dimension-mismatch':lambda g:g.set((E.pourOutputAssignment,S.dimension,Literal(2,datatype=XSD.positiveInteger))),
'missing-input-role':lambda g:g.remove((E.pourDynamics,S.hasAssignment,E.pourInputAssignment)),
'foreign-evolution-equation':lambda g:g.add((E.pourDynamics,S.evolutionEquation,E.pourVolumeEquation)),
'duplicate-rank':lambda g:g.set((E.pourObjective_time,O.rank,Literal(0,datatype=XSD.nonNegativeInteger))),
'missing-rank':lambda g:g.remove((E.pourObjective_time,O.rank,None)),
'missing-weight':lambda g:g.remove((E.graspObjective_hold,O.weightParameter,None)),
'missing-normalization':lambda g:g.remove((E.graspObjective_hold,O.normalizedBy,None)),
'negative-weight':lambda g:numeric(g,E.graspWeight_hold,'-1'),
'zero-normalization':lambda g:numeric(g,E.graspScale_hold,'0'),
'zero-sampling-period':lambda g:numeric(g,E.pourSamplingPeriod,'0','http://qudt.org/vocab/unit/SEC'),
'foreign-adaptation-target':lambda g:g.set((E.pourAdaptation,S.updatesParameter,E.graspPropertyParameter)),
'all-zero-weights':lambda g:(numeric(g,E.graspWeight_hold,'0'),numeric(g,E.graspWeight_effort,'0')),
'hard-constraint-without-expression':lambda g:g.remove((E.pourSpillConstraint,Namespace('http://aicor.knowledge/ontologies/conditions-outcomes.owl#').hasExpression,None)),
'missing-controller-model':lambda g:g.remove((E.pourController,S.controlsModel,None)),
}
for n,mutate in mutations.items():
 g=Graph()+data;mutate(g)
 try:ok,_,_=check(g)
 except (ValueError,TypeError):ok=False
 record('reject-'+n,not ok)
positive=Graph()+data
for param in [E.graspWeight_hold,E.graspWeight_effort,E.graspScale_hold,E.graspScale_effort]:numeric(positive,param,'1')
ok,_,_=check(positive);record('explicit-positive-weights-and-scales-accepted',ok)
views=[inspect_controller(data,E.pourController),inspect_controller(data,E.graspController)]
record('dimension-calculated-from-entries',all(v['vectorDimensions']=={'State':1,'Input':1,'Output':1} for v in views))
record('missing-settings-explicit',all(v['unresolved'] and not v['executionReadinessAssessed'] for v in views))
record('claims-remain-proposed',all(data.value(E[p+'ControlClaim'+str(i)],A.hasStatus)==A.Proposed for p in ['pour','grasp'] for i in range(4)))
(R/'examples/controller-inspection.json').write_text(json.dumps(views,indent=2)+'\n');(R/'reports/control-validation.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps({'checks':len(results),'passed':sum(r['passed'] for r in results),'failures':[r for r in results if not r['passed']]},indent=2));assert all(r['passed'] for r in results)
