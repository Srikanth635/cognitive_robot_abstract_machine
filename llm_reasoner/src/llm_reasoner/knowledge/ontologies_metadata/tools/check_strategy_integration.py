"""Ontology consistency only for the local combined import closure and examples."""
from pathlib import Path
import json, subprocess
from rdflib import Graph, OWL
import owlready2.reasoning as reasoning
R=Path(__file__).resolve().parents[1]
from ontology_imports import closure
base,imports=closure('http://aicor.knowledge/ontologies/all-action-aspects.owl')
cram,cramimports=closure('http://aicor.knowledge/ontologies/cram-action-bridge.owl')
g=base+cram
files=sorted(str(p.relative_to(R)) for p in (R/'examples').glob('*.ttl'))
for name in files:g+=Graph().parse(R/name)
files=sorted(set(imports+cramimports))+files
g.remove((None,OWL.imports,None));p=R/'reports/strategy-integration-input.owl';g.serialize(p,format='xml')
try:
 result=subprocess.run(['java','-Xmx2048M','-cp',reasoning._HERMIT_CLASSPATH,'org.semanticweb.HermiT.cli.CommandLine','-k',p.as_uri()],capture_output=True,text=True,timeout=45)
 report={'scope':files,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr,'limitation':'Consistency only; not full classification of combined AICOR model.'}
except subprocess.TimeoutExpired:
 report={'scope':files,'status':'timeout','limit_seconds':45}
p.unlink();(R/'reports/strategy-integration.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
