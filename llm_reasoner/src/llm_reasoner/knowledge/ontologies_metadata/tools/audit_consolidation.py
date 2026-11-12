"""Check the precise schema delta against the supplied 0.9.0 release ZIP."""
import argparse, csv, hashlib, io, json, zipfile
from rdflib import Graph, Namespace, RDF, RDFS, OWL, Literal, URIRef
from rdflib.compare import isomorphic
from ontology_imports import ROOT, closure
B='http://aicor.knowledge/ontologies/'; A=Namespace(B+'action-description-core.owl#'); D=Namespace('http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#')
MAPPINGS={'core/action-description-core.ttl':['hasCapabilityRequirement'],
'aspects/capabilities-strategies.ttl':['requiresCapability'],
'aspects/failure-diagnosis.ttl':['requiresCapability'],
'aspects/monitoring-requirements.ttl':['requiresCapability'],
'aspects/entities-materials-affordances.ttl':['requiresCapability'],
'aspects/sensorimotor-events.ttl':['requiresCapability','observationRequirement']}

def main(baseline):
 checks=[]; changed=[]; unchanged=[]; later_assembly_changes=[]
 relocations=json.loads((ROOT/'aspect-layout-migration.json').read_text())
 def check(name,ok):checks.append({'case':name,'passed':bool(ok)})
 with zipfile.ZipFile(baseline) as old:
  names=[n for n in old.namelist() if n.endswith(('.ttl','.owl'))]
  for name in names:
   relative='/'.join(name.split('/')[1:]); path=ROOT/relocations.get(relative,relative)
   check('preserved-file:'+relative,path.exists())
   if not path.exists():continue
   before=old.read(name);after=path.read_bytes()
   if before==after:unchanged.append(relative);continue
   if relative=='integrations/all-action-aspects.ttl':
    later_assembly_changes.append(relative);continue
   changed.append(relative);check('authorized-schema-delta:'+relative,relative in MAPPINGS)
   if relative not in MAPPINGS:continue
   previous=Graph().parse(data=before.decode(),format='turtle');current=Graph().parse(path)
   ns=A if relative.startswith('core/') else Namespace(B+path.stem+'.owl#')
   for prop in MAPPINGS[relative]:
    triple=(ns[prop],RDFS.subPropertyOf,A.requiresCapability)
    check('new-requirement-parent:'+str(ns[prop]),triple in current);current.remove(triple)
   if relative.startswith('core/'):
    check('generic-domain-is-description',(A.requiresCapability,RDFS.domain,D.Description) in current)
    check('generic-range-is-requirement',(A.requiresCapability,RDFS.range,A.CapabilityRequirement) in current)
    current.remove((A.requiresCapability,None,None))
   current.remove((None,OWL.versionInfo,None));previous.remove((None,OWL.versionInfo,None))
   check('no-other-logical-delta:'+relative,isomorphic(previous,current))
 check('exactly-six-schema-files-modified',set(changed)==set(MAPPINGS))
 schema,_=closure(B+'all-action-aspects.owl')
 # Predicate ranges identify shared vocabulary anchors without merging their meanings.
 anchors=[A.ConditionSpecification,A.ParameterSpecification,A.CapabilityRequirement,A.EntitySpecification]
 rows=[]
 for prop in sorted(set(schema.subjects(RDF.type,OWL.ObjectProperty))):
  ranges=set(schema.objects(prop,RDFS.range))
  for anchor in anchors:
   if anchor in ranges:
    rows.append({'property':str(prop),'shared_range':str(anchor),'domains':' | '.join(sorted(str(x) for x in schema.objects(prop,RDFS.domain)))})
 with (ROOT/'review/shared-vocabulary.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=['property','shared_range','domains']);writer.writeheader();writer.writerows(rows)
 # Bounded CRAM overlap review: show asserted named parents and restriction counts.
 cram=Graph().parse(ROOT/'candidates/aicor-unified-dul.ttl');N=Namespace('http://aicor.knowledge/l2-ontology.owl#')
 candidates={'ActionDescription':['ActionDesignator','Designator','Action'], 'EntitySpecification':['ObjectDesignator','PhysicalObject'],
 'ParameterSpecification':['Parameter'],'ActionKind':['Task','Pouring','Grasping'],'ProcessModuleRequirement':['ProcessModule'],'ExecutionStrategy':['Plan']}
 rows=[]
 for generic,terms in candidates.items():
  for name in terms:
   node=N[name];parents=list(cram.objects(node,RDFS.subClassOf))
   check('reviewed-aicor-term-exists:'+name,(node,RDF.type,OWL.Class) in cram)
   rows.append({'generic_concept':generic,'aicor_class':str(node),'named_parents':[str(x) for x in parents if isinstance(x,URIRef)],
                'anonymous_parent_expressions':len([x for x in parents if not isinstance(x,URIRef)]),'decision':'No equivalence asserted; abstraction and restrictions differ or remain unverified.'})
 (ROOT/'review/aicor-overlap.json').write_text(json.dumps(rows,indent=2)+'\n')
 report={'audit_version':'0.14.1','historical_scope':'0.9 to 0.10 capability consolidation; later assembly changes excluded','later_assembly_changes_excluded':later_assembly_changes,'baseline':baseline.name,'baseline_sha256':hashlib.sha256(baseline.read_bytes()).hexdigest(),
 'changed_schema_files':changed,'byte_identical_ontology_and_example_files':unchanged,'shared_range_property_rows':len(list(csv.DictReader((ROOT/'review/shared-vocabulary.csv').open()))),
 'checks':checks,'passed':sum(x['passed'] for x in checks)}
 (ROOT/'reports/consolidation-audit-current-layout.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({'checks':len(checks),'passed':report['passed'],'changed':changed,'byte_identical':len(unchanged)}))
 assert all(x['passed'] for x in checks)

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('baseline',type=__import__('pathlib').Path);main(p.parse_args().baseline)
