from pathlib import Path
import json
from rdflib import Graph,Namespace
from control_runtime import A,F,O,S,inspect_controller
R=Path(__file__).resolve().parents[1];E=Namespace('http://aicor.knowledge/ontologies/action-examples.owl#');g=Graph()
for n in ['stereotypical-actions','aspect-actions','phase-actions','causal-physical-actions','control-actions']:g+=Graph().parse(R/'examples'/(n+'.ttl'))
def val(n,p):v=g.value(n,p);return str(v) if v is not None else None
views=[]
for record in [E.pourRecordV5,E.graspRecordV5]:
 action=g.value(record,A.expressesDescription);controller=g.value(action,A.hasControl);objset=g.value(controller,S.objectiveSet)
 out=inspect_controller(g,controller);out.update(record=str(record),description=str(action),aggregationPolicy=val(objset,O.aggregationPolicy),hardConstraints=[str(c) for c in g.objects(objset,O.hardConstraint)],objectives=[],claims=[])
 for term in sorted(g.objects(objset,O.hasTerm)):
  out['objectives'].append({'id':str(term),'metric':val(term,O.metricParameter),'direction':val(term,O.direction),'rank':val(term,O.rank),'weightParameter':val(term,O.weightParameter),'normalization':val(term,O.normalizedBy)})
 for claim in g.objects(record,A.hasClaim):
  if 'ControlClaim' in str(claim):out['claims'].append({'id':str(claim),'status':val(claim,A.hasStatus),'evidence':[str(e) for e in g.objects(claim,A.hasEvidence)]})
 views.append(out)
assert len(views)==2 and sum(len(v['claims']) for v in views)==8
(R/'examples/control-projections.json').write_text(json.dumps(views,indent=2)+'\n');print('Projected two proposed control/objective descriptions with unresolved settings and claim provenance.')
