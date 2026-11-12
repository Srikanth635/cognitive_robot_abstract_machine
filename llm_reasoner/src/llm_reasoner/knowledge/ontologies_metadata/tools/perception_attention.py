"""Scoped result inspection, not sensor acquisition, identity grounding or belief publication."""
from decimal import Decimal
import math
from rdflib import Namespace,RDF
from aspect_runtime import A,C,one
from context_review import K,instant
from condition_atoms import atoms
N=Namespace('http://aicor.knowledge/ontologies/perception-attention.owl#')
FAMILIES={'AttentionDescription':N.AttentionDirected,'VisibilityAssessmentDescription':N.VisibilityResolved,'DetectionDescription':N.CandidateReported,'IdentificationDescription':N.IdentityResolved,'SearchDescription':N.SearchSucceeded}
def family(g,action):
 matches=[name for name in FAMILIES if (action,RDF.type,N[name]) in g]
 if len(matches)!=1:raise ValueError('Select exactly one concrete perception profile')
 return matches[0]
def budget(g,specification):
 p=one(g,specification,N.attemptBudget);v=Decimal(str(one(g,p,A.numericValue)))
 if not v.is_finite() or v<1 or v!=v.to_integral_value():raise ValueError('Attempt budget must be a positive integer')
 if str(one(g,p,A.unitIRI))!='http://qudt.org/vocab/unit/UNITLESS':raise ValueError('Budget must be dimensionless')
 return int(v)
def perception_errors(g):
 errors=[];actions={s for kind in FAMILIES for s in g.subjects(RDF.type,N[kind])}
 for action in actions:
  kind=family(g,action);observer=one(g,action,N.observer);target=one(g,action,N.target);region=one(g,action,N.region);frame=one(g,action,N.referenceFrame);context=one(g,action,N.evaluationContext);query=one(g,action,N.query)
  if one(g,context,K.scopedDescription)!=action:errors.append('context describes another action')
  if (one(g,query,N.queryTarget),one(g,query,N.queryRegion),one(g,query,N.queryFrame))!=(target,region,frame):errors.append('query target/region/frame differs from profile')
  args=(observer,target) if kind=='AttentionDescription' else (observer,target,region)
  actual=atoms(g,one(g,one(g,action,A.hasSuccessCriterion),C.hasExpression))
  if (C.PostState,FAMILIES[kind],args,False) not in actual:errors.append('success omits the scoped profile outcome')
  if kind=='SearchDescription':
   specification=one(g,action,N.searchSpecification);views=list(g.objects(specification,N.hasView));indices=[int(one(g,v,N.positionIndex)) for v in views]
   if sorted(indices)!=list(range(1,len(views)+1)):errors.append('invalid observation-view order')
   for view in views:
    if one(g,view,N.viewRegion)!=region or one(g,view,N.viewFrame)!=frame:errors.append('view region/frame differs from query scope')
    if len(set(g.subjects(N.hasView,view)))!=1:errors.append('indexed view occurrence must have one owner')
   try:budget(g,specification)
   except ValueError as e:errors.append(str(e))
 for proposal in g.subjects(RDF.type,N.BeliefUpdateProposal):
  record=one(g,proposal,N.proposalRecord);context=one(g,proposal,N.proposalContext);result=one(g,proposal,N.basedOnResult);claim=one(g,proposal,N.proposedClaim)
  if (record,A.hasClaim,claim) not in g:errors.append('proposal claim absent from record')
  if one(g,result,N.inRecord)!=record or one(g,result,N.inContext)!=context:errors.append('proposal source belongs to another record/context')
 return errors

def inspect(g,record,result_ids,at,max_age_seconds):
 if not math.isfinite(float(max_age_seconds)) or max_age_seconds<0:raise ValueError('Finite nonnegative freshness limit required')
 now=instant(at);action=one(g,record,A.expressesDescription);kind=family(g,action);query=one(g,action,N.query);context=one(g,action,N.evaluationContext);frame=one(g,query,N.queryFrame)
 if one(g,context,K.scopedDescription)!=action:raise ValueError('Wrong action context')
 if (one(g,query,N.queryTarget),one(g,query,N.queryRegion),frame)!=(one(g,action,N.target),one(g,action,N.region),one(g,action,N.referenceFrame)):raise ValueError('Query scope disagrees with action')
 ids=list(result_ids)
 if len(ids)!=len(set(ids)):raise ValueError('Duplicate result selection')
 def answer(status,goal=None,completed=None,**extra):return {'status':status,'goalSatisfied':goal,'operationCompleted':completed,'record':str(record),'context':str(context),'worldAbsenceInferred':False,'acceptedBeliefs':0,'worldUpdated':False,'executed':False,**extra}
 if not ids:return answer('no-result')
 rows=[]
 for result in ids:
  if (result,RDF.type,N.PerceptionResultRecord) not in g:raise ValueError('Expected an explicit perception-result record')
  if one(g,result,N.inRecord)!=record or one(g,result,N.inContext)!=context or one(g,result,N.answersQuery)!=query:raise ValueError('Foreign record/context/query result')
  if one(g,result,N.resultFrame)!=frame:raise ValueError('Result frame mismatch; no implicit transform')
  one(g,result,A.sourceIRI)
  age=(now-instant(one(g,result,K.observedAt))).total_seconds()
  if age<0:return answer('future-evidence')
  if age>max_age_seconds:return answer('stale-evidence')
  evidence_kind=one(g,result,A.evidenceKind)
  if evidence_kind not in [A.Observation,A.SimulationResult,A.CuratedExample]:return answer('unsupported-evidence-kind')
  outcome=one(g,result,N.outcome)
  if outcome not in [N.Completed,N.Failed,N.Undetermined]:raise ValueError('Unknown result outcome')
  candidates=[];entities=set()
  for candidate in g.objects(result,N.hasCandidate):
   if set(g.subjects(N.hasCandidate,candidate))!={result}:raise ValueError('Candidate hypothesis must belong to one result')
   entity=one(g,candidate,N.candidateEntity);match=one(g,candidate,N.matchAssessment)
   if match not in [N.Matches,N.DoesNotMatch,N.MatchUnknown]:raise ValueError('Unknown match assessment')
   if entity in entities:raise ValueError('Duplicate candidate entity hypotheses require consolidation')
   entities.add(entity)
   for pose in g.objects(candidate,N.poseEstimate):
    if one(g,pose,N.estimateFrame)!=frame:raise ValueError('Candidate pose estimate uses another frame')
    for axis in ['x','y','z']:
     p=one(g,pose,N[axis+'Parameter']);v=Decimal(str(one(g,p,A.numericValue)))
     if not v.is_finite() or str(one(g,p,A.unitIRI))!='http://qudt.org/vocab/unit/M':raise ValueError('Estimated position needs finite metre coordinates')
    for orientation in g.objects(pose,N.orientationEstimate):
     for axis in ['qx','qy','qz','qw']:
      p=one(g,orientation,N[axis+'Parameter']);v=Decimal(str(one(g,p,A.numericValue)))
      if not v.is_finite() or str(one(g,p,A.unitIRI))!='http://qudt.org/vocab/unit/UNITLESS':raise ValueError('Quaternion estimate components must be finite and dimensionless')
   candidates.append((candidate,entity,match))
  rows.append({'id':result,'outcome':outcome,'candidates':candidates,'synthetic':evidence_kind==A.CuratedExample})
 synthetic=any(row['synthetic'] for row in rows)
 def report(status,goal=None,completed=None,**extra):return answer(status,goal,completed,syntheticInput=synthetic,**extra)
 def candidate_outcome(row,identification=False):
  matching=[c for c,e,m in row['candidates'] if m==N.Matches];unknown=[c for c,e,m in row['candidates'] if m==N.MatchUnknown]
  if identification:
   if len(matching)>1:return ('ambiguous',None,matching)
   if unknown:return ('identity-unresolved',None,matching)
   return ('identified',True,matching) if len(matching)==1 else ('not-identified',False,[])
  if matching:return ('detected',True,matching)
  return ('matching-unresolved',None,[]) if unknown else ('not-detected-within-scope',False,[])
 if kind!='SearchDescription':
  if len(rows)!=1:return report('multiple-results-require-selection')
  row=rows[0];result=row['id']
  if list(g.objects(result,N.forView)):raise ValueError('Non-search result cannot name a search view')
  if row['outcome']==N.Failed:return report('operation-failed',None,False)
  if row['outcome']==N.Undetermined:return report('operation-undetermined')
  if kind=='AttentionDescription':
   state=one(g,result,N.attentionState)
   if state not in [N.Directed,N.NotDirected,N.AttentionUnknown]:raise ValueError('Unknown attention state')
   return report('attention-directed' if state==N.Directed else 'attention-not-directed' if state==N.NotDirected else 'attention-unresolved',True if state==N.Directed else False if state==N.NotDirected else None,True)
  if kind=='VisibilityAssessmentDescription':
   state=one(g,result,N.visibilityState)
   if state not in [N.Visible,N.Occluded,N.VisibilityUnknown]:raise ValueError('Unknown visibility state')
   return report('visibility-known' if state!=N.VisibilityUnknown else 'visibility-unresolved',True if state!=N.VisibilityUnknown else None,True,visibility=str(state))
  status,goal,candidates=candidate_outcome(row,kind=='IdentificationDescription');return report(status,goal,True,candidates=[str(c) for c in candidates])
 specification=one(g,action,N.searchSpecification);limit=budget(g,specification);views=sorted(g.objects(specification,N.hasView),key=lambda v:int(one(g,v,N.positionIndex)));by_view={}
 for row in rows:
  view=one(g,row['id'],N.forView)
  if view not in views:raise ValueError('Result names a foreign observation view')
  if view in by_view:return report('multiple-results-for-view-require-selection')
  by_view[view]=row
 attempted=[v for v in views if v in by_view]
 if attempted!=views[:len(attempted)]:return report('out-of-order-view-evidence')
 if len(attempted)>limit:return report('attempt-budget-exceeded')
 for index,view in enumerate(attempted):
  row=by_view[view]
  if row['outcome']==N.Failed:return report('search-operation-failed',None,False)
  if row['outcome']==N.Undetermined:return report('search-result-unresolved')
  status,goal,candidates=candidate_outcome(row)
  if goal is None:return report('search-match-unresolved')
  if goal is True:
   if index<len(attempted)-1:return report('results-beyond-stop-condition')
   return report('search-found',True,True,candidates=[str(c) for c in candidates],attempts=len(attempted))
 if len(attempted)==len(views):return report('declared-views-exhausted',False,True,attempts=len(attempted))
 if len(attempted)==limit:return report('attempt-budget-exhausted',False,True,attempts=len(attempted))
 return report('search-incomplete',None,None,attempts=len(attempted))
