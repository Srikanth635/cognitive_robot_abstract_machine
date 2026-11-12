"""Readable phase/monitoring views with source-record and claim provenance."""
from pathlib import Path
import json
from rdflib import Graph,Namespace,RDF,RDFS
from temporal_runtime import A,P,C,T,M,one
R=Path(__file__).resolve().parents[1];E=Namespace('http://aicor.knowledge/ontologies/action-examples.owl#')
g=Graph()
for name in ['examples/stereotypical-actions.ttl','examples/aspect-actions.ttl','examples/phase-actions.ttl']:g+=Graph().parse(R/name)
def value(n,p):v=g.value(n,p);return str(v) if v is not None else None
def claims(record,component):
 return [{'id':str(c),'status':value(c,A.hasStatus),'evidence':[str(e) for e in g.objects(c,A.hasEvidence)]} for c in sorted(g.objects(record,A.hasClaim)) if (c,A.aboutComponent,component) in g]
def monitor(record,n):
 obs=[]
 for o in sorted(g.objects(n,M.requiresObservation)):
  obs.append({'id':str(o),'pattern':value(o,M.observesPattern),'parameter':value(o,M.observesParameter),'scope':value(o,M.inScope),'capabilities':[str(c) for c in g.objects(o,M.requiresCapability)]})
 return {'id':str(n),'condition':value(n,M.monitorsCondition),'point':value(n,M.atPoint),'onTrue':value(n,M.whenTrue),'onFalse':value(n,M.whenFalse),'onUnknown':value(n,M.whenUnknown),'observations':obs,'claims':claims(record,n)}
views=[]
for record in [E.pourRecordV3,E.graspRecordV3]:
 action=one(g,record,A.expressesDescription);structure=one(g,action,T.hasPhaseStructure)
 out={'record':str(record),'version':value(record,A.recordVersion),'description':str(action),'structure':str(structure),'initialPhase':value(structure,T.initialPhase),'claims':claims(record,structure),'phases':[],'transitions':[]}
 for p in sorted(g.objects(structure,T.hasMemberPhase)):
  out['phases'].append({'id':str(p),'label':value(p,RDFS.label),'types':[str(t) for t in g.objects(p,RDF.type)],'entryCondition':value(p,T.entryCondition),'exitCondition':value(p,T.exitCondition),'invariants':[str(c) for c in g.objects(p,T.invariantCondition)],'expectedOutcomes':[str(o) for o in g.objects(p,T.expectedOutcome)],'monitors':[monitor(record,m) for m in sorted(g.objects(p,M.hasMonitoringRequirement))]})
 for t in sorted(g.objects(structure,T.hasTransition)):
  out['transitions'].append({'id':str(t),'types':[str(k) for k in g.objects(t,RDF.type)],'from':value(t,T.fromPhase),'to':value(t,T.toPhase),'guard':value(t,T.guardCondition),'claims':claims(record,t),'monitors':[monitor(record,m) for m in sorted(g.objects(t,M.hasMonitoringRequirement))]})
 views.append(out)
(R/'examples/phase-projections.json').write_text(json.dumps(views,indent=2)+'\n')
rows=list(g.query((R/'queries/phase-monitoring.rq').read_text()))
counts={'records':len(views),'phases':sum(len(v['phases']) for v in views),'transitions':sum(len(v['transitions']) for v in views),'monitoring_requirements':len(set(g.subjects(RDF.type,M.MonitoringRequirement))),'query_rows':len(rows)}
assert counts['records']==2 and counts['phases']==11 and counts['transitions']==13 and counts['monitoring_requirements']==7 and counts['query_rows']==19,counts
(R/'reports/temporal-projection-summary.json').write_text(json.dumps(counts,indent=2)+'\n');print(json.dumps(counts,indent=2))
