"""Generate demonstration frame views from unchanged shared knowledge records."""
import json
from rdflib import Graph,Namespace
from ontology_imports import ROOT
from frame_runtime import project_frame,artifact_graph
E=Namespace('http://aicor.knowledge/ontologies/action-examples.owl#');g=Graph()
for name in ['stereotypical-actions','aspect-actions','phase-actions','causal-physical-actions','control-actions','strategy-failure-actions','frame-contracts']:g+=Graph().parse(ROOT/'examples'/(name+'.ttl'))
outputs=[];artifacts=Graph()
for record,contract in [(E.pourRecordV6,E.PouringFrameContract),(E.graspRecordV6,E.GraspingFrameContract)]:
 out=project_frame(g,record,contract);outputs.append(out);artifacts+=artifact_graph(out)
(ROOT/'examples/frame-projections.json').write_text(json.dumps(outputs,indent=2)+'\n');artifacts.serialize(ROOT/'examples/frame-projection-artifacts.ttl',format='turtle')
print('Wrote two frame views and RDF provenance artifacts from existing v6 records.')
