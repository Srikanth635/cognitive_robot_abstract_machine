import sys,json,time
from pathlib import Path
sys.path.insert(0,str(Path(sys.argv[1]).resolve()))
t0=time.monotonic()
import aicor_model as m
from dataclasses import fields
from krrood.entity_query_language.predicate import Symbol
from krrood.ontomatic.property_descriptor.property_descriptor import PropertyDescriptor
from rdflib import Graph,OWL,RDF,RDFS
base='http://aicor.knowledge/l2-ontology.owl#'
assert len(m.CLASS_BY_IRI)==1017 and len(m.MODEL.properties)==966
assert m.CLASS_BY_IRI[base+'None'].__name__=='None_'
for parent in ("AttributiveDesignator", "CRAMDesignator", "Representation"):
    assert issubclass(m.CLASS_BY_IRI[base+"ActionDesignator"], m.CLASS_BY_IRI[base+parent])
obj=m.CLASS_BY_IRI[base+"AICORReferenceManipulationExecutive"]()
part=m.ManipulationControlProgram()
assert isinstance(obj,Symbol) and isinstance(obj.has_proper_part,list)
obj.has_proper_part.append(part)
assert obj.has_proper_part[0] is part
assert isinstance(type(obj).has_proper_part,PropertyDescriptor)
critic=m.AccessibilityCritic()
assert critic.critic_dimension==[] # The restriction has not become an asserted default.
critic.critic_dimension.append('accessibility')
assert m.model_queries.definitions=={}
assert not hasattr(m,'validate')
class_records=[r for r in m.SCHEMA_AXIOMS if r['class_iri'] is not None]
restriction_nodes={s for r in class_records for s,p,o in r['triples'] if p==RDF.type.n3() and o==OWL.Restriction.n3()}
assert len(restriction_nodes)==1341
closure_text = (Path(sys.argv[1])/'source/closure.ttl').read_text()
# RDFLib remaps blank-node identifiers when reparsing Turtle. Verify the
# source-linked records against the actual serialized triple spellings.
closure_lines = set(closure_text.splitlines())
assert all(' '.join(triple)+' .' in closure_lines
           for record in m.SCHEMA_AXIOMS for triple in record['triples'])
report={'classes':len(m.CLASS_BY_IRI),'properties':len(m.MODEL.properties),'field_bindings':len(m.MODEL.bindings),'class_axiom_records':len(class_records),'global_axiom_records':len(m.SCHEMA_AXIOMS)-len(class_records),'restriction_nodes_preserved':len(restriction_nodes),'unbound_properties':len(set(m.MODEL.properties)-{b.iri for b in m.MODEL.bindings}),'runtime_check_seconds':round(time.monotonic()-t0,3)}
print(json.dumps(report,indent=2))
