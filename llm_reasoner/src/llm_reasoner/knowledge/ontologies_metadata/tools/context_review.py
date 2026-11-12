"""Bounded context inspection: explicit time, freshness and exact-statement alternatives.
No world lookups, value materialization, automatic conflict resolution or trust scoring.
"""
from datetime import datetime, timezone
from hashlib import sha256
from rdflib import Namespace,RDF
from aspect_runtime import A,one
K=Namespace('http://aicor.knowledge/ontologies/knowledge-context.owl#')
def instant(value):
 d=datetime.fromisoformat(str(value).replace('Z','+00:00'))
 if d.tzinfo is None:raise ValueError('Timezone required')
 return d.astimezone(timezone.utc)
def verify_bytes(payload,expected):return sha256(payload).hexdigest()==str(expected)
def inspect(g,record,context,at,max_age_seconds):
 now=instant(at)
 if max_age_seconds<0:raise ValueError('Freshness limit must be nonnegative')
 if (context,K.scopedDescription,one(g,record,A.expressesDescription)) not in g:raise ValueError('Context belongs to another description')
 rows=[];groups={}
 for assessment in g.subjects(K.inRecord,record):
  if one(g,assessment,K.inContext)!=context:continue
  claim=one(g,assessment,K.assessesClaim)
  if (record,A.hasClaim,claim) not in g:raise ValueError('Claim absent from record')
  start=instant(one(g,assessment,K.validFrom));end=instant(one(g,assessment,K.validUntil));created=instant(one(g,assessment,K.assessedAt))
  if start>=end:raise ValueError('Invalid validity interval')
  if not start<=now<end or created>now:continue
  status=one(g,assessment,K.assessedStatus);evidence=list(g.objects(assessment,K.usesEvidence));freshness=[]
  for e in evidence:
   if (e,RDF.type,K.ObservationEvidence) in g:
    age=(now-instant(one(g,e,K.observedAt))).total_seconds()
    freshness.append('future' if age<0 else 'stale' if age>max_age_seconds else 'fresh')
   else:freshness.append('not-observation')
  if status==A.Supported and not evidence:raise ValueError('Supported requires evidence')
  statement=one(g,claim,A.hasStatement);subject=one(g,statement,A.statementSubject);predicate=one(g,statement,A.statementPredicateIRI)
  objects=list(g.objects(statement,A.statementObject))+list(g.objects(statement,A.statementLiteral))
  if len(objects)!=1:raise ValueError('Statement requires exactly one value')
  row={'assessment':str(assessment),'claim':str(claim),'status':str(status),'value':objects[0].n3(),'evidenceFreshness':freshness,'accepted':False}
  rows.append(row)
  if status not in [A.Rejected,A.Unknown]:groups.setdefault((subject,predicate),{}).setdefault(objects[0],[]).append(str(assessment))
 alternatives=[{'subject':str(s),'predicate':str(p),'assessments':sorted(x for vals in values.values() for x in vals)} for (s,p),values in groups.items() if len(values)>1]
 return {'record':str(record),'context':str(context),'assessments':sorted(rows,key=lambda r:r['assessment']), 'potentialAlternativeValues':alternatives,'conflictResolutionPerformed':False}
