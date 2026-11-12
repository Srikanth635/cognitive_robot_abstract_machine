"""Bounded frame projection: single inheritance, explicit defaults and finite paths.

No external lookups, procedural attachments, world assertions or record mutation.
"""
from hashlib import sha256
from rdflib import Namespace,RDF,URIRef,Literal
from aspect_runtime import A,one
B='http://aicor.knowledge/ontologies/';F=Namespace(B+'frame-perspective.owl#')
def effective_slots(g,frame,active=None):
 active=set() if active is None else active
 if frame in active:raise ValueError('Cyclic frame inheritance')
 active.add(frame);parents=list(g.objects(frame,F.parentFrame))
 if len(parents)>1:raise ValueError('Only single inheritance is supported')
 slots=effective_slots(g,parents[0],active) if parents else {};inherited={v['slot'] for v in slots.values()}
 for slot in sorted(g.objects(frame,F.hasSlot)):
  key=str(one(g,slot,F.slotKey))
  if key in slots:raise ValueError('Duplicate or ambiguous slot key')
  default=list(g.objects(slot,F.defaultLiteral))+list(g.objects(slot,F.defaultEntity))
  if len(default)>1:raise ValueError('Multiple defaults')
  lo=int(one(g,slot,F.minimumCount));upper=list(g.objects(slot,F.maximumCount));hi=int(upper[0]) if upper else None
  if len(upper)>1 or lo<0 or (hi is not None and (hi<lo)):raise ValueError('Invalid slot cardinality')
  mappings=list(g.objects(slot,F.hasMapping))
  if len(mappings)>1:raise ValueError('Multiple slot mappings')
  path=[]
  if mappings:
   steps=list(g.objects(mappings[0],F.hasPathStep));positions=[]
   for step in steps:
    pos=int(one(g,step,F.position));pred=str(one(g,step,F.predicateIRI))
    if not __import__('re').match(r'^[A-Za-z][A-Za-z0-9+.-]*:',pred):raise ValueError('Predicate must be an absolute IRI')
    positions.append((pos,URIRef(pred)))
   if not steps or sorted(p for p,_ in positions)!=list(range(1,len(steps)+1)):raise ValueError('Invalid path positions')
   path=[p for _,p in sorted(positions)]
  slots[key]={'slot':slot,'minimum':lo,'maximum':hi,'default':default[0] if default else None,'defaultFrame':frame,'path':path}
 seen=set()
 for override in g.objects(frame,F.hasOverride):
  target=one(g,override,F.overridesSlot)
  if target not in inherited or target in seen:raise ValueError('Override must uniquely target an inherited slot')
  seen.add(target);replace=list(g.objects(override,F.replacementLiteral));supp=list(g.objects(override,F.suppressDefault))
  suppress=len(supp)==1 and supp[0].toPython() is True
  if len(supp)>1 or len(replace)>1 or (suppress and replace) or (not suppress and not replace):raise ValueError('Invalid default override')
  for key,value in slots.items():
   if value['slot']==target:slots[key]={**value,'default':None if suppress else replace[0],'defaultFrame':frame}
 active.remove(frame);return slots

def project_frame(g,record,contract):
 if (record,RDF.type,A.KnowledgeRecord) not in g:raise ValueError('Source must be a knowledge record')
 version=str(one(g,record,A.recordVersion));desc=one(g,record,A.expressesDescription);frame=one(g,contract,F.frameDefinition)
 policy=str(one(g,contract,F.evidencePolicy));missingPolicy=str(one(g,contract,A.missingInformationPolicy))
 if policy not in ['retain-with-status','require-allowed-claim']:raise ValueError('Unknown evidence policy')
 if missingPolicy not in ['omit-and-report','retain-unknown','reject']:raise ValueError('Unknown missing-information policy')
 allowed=set(g.objects(contract,F.allowedStatus))
 if policy=='require-allowed-claim' and not allowed:raise ValueError('Strict policy needs explicit allowed statuses')
 claims=set(g.objects(record,A.hasClaim));included=set();issues=[];outputs=[]
 def encode(term):
  if not isinstance(term,(URIRef,Literal)):raise ValueError('Frame output needs stable IRI or literal values')
  return {'kind':'literal','value':str(term),'datatype':str(term.datatype) if term.datatype else None,'language':term.language} if isinstance(term,Literal) else {'kind':'iri','value':str(term)}
 for key,spec in sorted(effective_slots(g,frame).items()):
  candidates={desc} if spec['path'] else set()
  for predicate in spec['path']:candidates={o for s in candidates for o in g.objects(s,predicate)}
  values=[]
  for candidate in sorted(candidates,key=str):
   associated=sorted(c for c in claims if (c,A.aboutComponent,candidate) in g);statuses={one(g,c,A.hasStatus) for c in associated}
   if policy=='require-allowed-claim' and (not statuses&allowed or statuses&{A.Rejected,A.Disputed}):
    issues.append({'slot':key,'issue':'excluded-by-evidence-policy','value':encode(candidate)});continue
   support=[{'claim':str(c),'status':str(one(g,c,A.hasStatus)),'evidence':[str(e) for e in sorted(g.objects(c,A.hasEvidence))]} for c in associated]
   included.update(associated);values.append({'term':encode(candidate),'origin':'source','support':support,'assessment':'claims-attached' if support else 'unassessed-template-content'})
  # Filtered source content must never be silently replaced by a default.
  if not candidates and spec['default'] is not None and policy=='require-allowed-claim':
   issues.append({'slot':key,'issue':'default-excluded-by-evidence-policy'})
  if not candidates and spec['default'] is not None and policy!='require-allowed-claim':
   values.append({'term':encode(spec['default']),'origin':'default','defaultFrame':str(spec['defaultFrame']),'support':[],'assessment':'default-unverified'})
  if len(values)<spec['minimum']:
   issues.append({'slot':key,'issue':'minimum-cardinality','required':spec['minimum'],'actual':len(values)})
  if spec['maximum'] is not None and len(values)>spec['maximum']:
   issues.append({'slot':key,'issue':'maximum-cardinality','required':spec['maximum'],'actual':len(values)})
  outputs.append({'key':key,'slot':str(spec['slot']),'values':values,'unknown':not values and missingPolicy=='retain-unknown'})
 if missingPolicy=='reject' and issues:raise ValueError('Projection rejected: '+str(issues))
 return {'sourceRecord':str(record),'sourceVersion':version,'description':str(desc),'contract':str(contract),'frame':str(frame),'evidencePolicy':policy,'slots':outputs,'issues':issues,'includedClaims':sorted(str(c) for c in included),'executionReadinessAssessed':False}

def artifact_graph(result):
 from rdflib import Graph
 g=Graph();token=sha256((result['sourceRecord']+'|'+result['contract']).encode()).hexdigest();artifact=URIRef(B+'frame-artifacts#'+token)
 for t in [(artifact,RDF.type,F.FrameProjectionArtifact),(artifact,A.derivedFromRecord,URIRef(result['sourceRecord'])),(artifact,A.conformsToSpecification,URIRef(result['contract'])),(artifact,F.projectedFrame,URIRef(result['frame']))]:g.add(t)
 for claim in result['includedClaims']:g.add((artifact,A.includesClaim,URIRef(claim)))
 for slot in result['slots']:
  for i,value in enumerate(slot['values']):
   node=URIRef(str(artifact)+'-'+sha256(slot['key'].encode()).hexdigest()[:12]+'-'+str(i));g.add((artifact,F.hasSlotValue,node));g.add((node,RDF.type,F.FrameSlotValue));g.add((node,F.fillsSlot,URIRef(slot['slot'])));g.add((node,F.valueOrigin,Literal(value['origin'])))
   term=value['term'];obj=URIRef(term['value']) if term['kind']=='iri' else Literal(term['value'],lang=term['language'],datatype=URIRef(term['datatype']) if term['datatype'] else None)
   g.add((node,F.entityValue if term['kind']=='iri' else F.literalValue,obj))
   for c in value['support']:g.add((node,F.sourceClaim,URIRef(c['claim'])))
 return g
