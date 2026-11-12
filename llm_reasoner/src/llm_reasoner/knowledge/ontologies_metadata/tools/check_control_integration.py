"""Ontology consistency only for the local combined import closure and examples."""
from pathlib import Path
import json, subprocess
from rdflib import Graph, OWL
import owlready2.reasoning as reasoning
R=Path(__file__).resolve().parents[1]
files=['DUL.owl','candidates/aicor-unified-dul.ttl','core/action-description-core.ttl','core/cram-action-bridge.ttl','examples/stereotypical-actions.ttl','examples/projection-artifacts.ttl','aspects/structure/participants-relations.ttl','aspects/structure/conditions-outcomes.ttl','examples/aspect-actions.ttl','aspects/structure/phases-transitions.ttl','aspects/observation/monitoring-requirements.ttl','examples/phase-actions.ttl','aspects/causality/causal-mechanisms.ttl','aspects/physics-control/physical-models.ttl','examples/causal-physical-actions.ttl','aspects/planning/objectives-tradeoffs.ttl','aspects/physics-control/dynamics-control.ttl','examples/control-actions.ttl']
files += sorted(str(p.relative_to(R)) for p in (R/'integrations').glob('*.ttl'))
g=Graph()
for name in files:g+=Graph().parse(R/name)

g.remove((None,OWL.imports,None));p=R/'reports/control-integration-input.owl';g.serialize(p,format='xml')
try:
 result=subprocess.run(['java','-Xmx2048M','-cp',reasoning._HERMIT_CLASSPATH,'org.semanticweb.HermiT.cli.CommandLine','-k',p.as_uri()],capture_output=True,text=True,timeout=45)
 report={'scope':files,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr,'limitation':'Consistency only; not full classification of combined AICOR model.'}
except subprocess.TimeoutExpired:
 report={'scope':files,'status':'timeout','limit_seconds':45}
p.unlink();(R/'reports/control-integration.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
