"""Synthetic contract/negative tests; no physical tools or robot execution."""
import json
from copy import deepcopy
from datetime import datetime,timedelta,timezone
from rdflib import Graph,Namespace,RDF,RDFS,URIRef,Literal,XSD
from pyshacl import validate
from ontology_imports import ROOT,closure
from aspect_runtime import A,C,P,one,structural_errors
from condition_signatures import signature
from temporal_runtime import temporal_errors
from specialized_tool_interactions import N as ST,TI,PROFILES,specialized_errors,assess as tool_inspect
from action_recipes import N,OM,TC,recipe_errors,assess as recipe_inspect,ordered_steps
from transport_carrying import MN
from context_review import K
B='http://aicor.knowledge/ontologies/';S=Namespace(B+'specialized-tool-examples.owl#');E=Namespace(B+'recipe-examples.owl#')
schema,imports=closure(B+'action-recipes.owl');schema-=Graph().parse(ROOT/'DUL.owl');data=Graph().parse(ROOT/'examples/specialized-tool-profiles.ttl')+Graph().parse(ROOT/'examples/action-recipe-profiles.ttl');g=data+schema;shapes=Graph()
for p in [ROOT/'core/action-core-shapes.ttl',ROOT/'core/knowledge-context-shapes.ttl']+list((ROOT/'aspects/validation').glob('*shapes.ttl'))+list((ROOT/'domains').glob('*shapes.ttl'))+list((ROOT/'compositions').glob('*shapes.ttl'))+list((ROOT/'refinements').glob('*shapes.ttl')):shapes.parse(p)
checks=[]
def check(name,ok):checks.append({'case':name,'passed':bool(ok)})
def rejects(name,fn):
 try:fn();check(name,False)
 except ValueError:check(name,True)
ok,_,text=validate(data,shacl_graph=shapes,ont_graph=schema,inference='rdfs',advanced=True);(ROOT/'reports/recipes-specialized-shacl.txt').write_text(text)
check('eight-fixture-shapes',ok);check('specialized-tool-integrity',not specialized_errors(g));check('recipe-integrity',not recipe_errors(g));check('expression-signatures',not structural_errors(g));check('nested-phase-monitor-integrity',not temporal_errors(g))
def supplied(condition):
 atoms,pairs=signature(g,one(g,condition,C.hasExpression));facts={(sc,rel,args):not neg for sc,rel,args,neg in atoms};q={}
 for op,sc,left,rsc,right in pairs:q[(sc,left)]=('0.01',str(one(g,left,A.unitIRI)));q[(rsc,right)]=('0.02',str(one(g,right,A.unitIRI)))
 return facts,q
def conditions(action):
 f,q=supplied(one(g,action,A.hasPrecondition));ff,qq=supplied(one(g,action,A.hasSuccessCriterion));f.update(ff);q.update(qq);return f,q
def window(context,start='2026-11-12T10:00:00+00:00',end='2026-11-12T10:00:10+00:00'):return {'context':str(context),'start':start,'end':end}
for name in ['cutting','wiping','pressing','insertion']:
 action=S[name];ctx=one(g,action,TI.evaluationContext);facts,q=conditions(action);kw={'facts':facts,'quantities':q,'interval':window(ctx)}
 check(name+':unknown-without-evidence',tool_inspect(g,action,ctx)['success'] is None)
 out=tool_inspect(g,action,ctx,**kw);check(name+':specific-outcome-satisfies',out['success'] is True and not out['executed'] and out['acceptedObservations']==0)
 generic={k:v for k,v in facts.items() if k[1] not in [ST.SpecifiedSeparationAchieved,ST.SpecifiedSurfaceEffectAchieved,ST.SpecifiedActuationAchieved,ST.SpecifiedSeatingAchieved]}
 check(name+':generic-tool-effect-insufficient',tool_inspect(g,action,ctx,generic,q,window(ctx))['success'] is None)
 wrong=dict(facts)
 key=next(k for k in facts if k[1] in [ST.SpecifiedSeparationAchieved,ST.SpecifiedSurfaceEffectAchieved,ST.SpecifiedActuationAchieved,ST.SpecifiedSeatingAchieved]);wrong[key]=False
 check(name+':specific-effect-false-fails',tool_inspect(g,action,ctx,wrong,q,window(ctx))['success'] is False)
 wrong=dict(facts);value=wrong.pop(key);wrong[(key[0],key[1],tuple(S.foreignTarget if x==one(g,action,TI.target) else x for x in key[2]))]=value
 check(name+':foreign-target-effect-not-borrowed',tool_inspect(g,action,ctx,wrong,q,window(ctx))['success'] is None)
 rejects(name+':foreign-context',lambda:tool_inspect(g,action,S.foreignContext,**kw))
 check(name+':no-fault-from-missing-success',tool_inspect(g,action,ctx)['fault'] is None)
 if name=='wiping':check('wiping:coverage-without-window-withheld',tool_inspect(g,action,ctx,facts,q)['success'] is None)
 if name=='insertion':
  error=one(g,action,ST.depthError);bad=dict(q);bad[(C.PostState,error)]=('0.03','http://qudt.org/vocab/unit/M');check('insertion:seated-label-not-depth-proof',tool_inspect(g,action,ctx,facts,bad,window(ctx))['success'] is False)
  bad[(C.PostState,error)]=('-1','http://qudt.org/vocab/unit/M');rejects('insertion:negative-error',lambda:tool_inspect(g,action,ctx,facts,bad,window(ctx)))
  bad[(C.PostState,error)]=('0.01','http://qudt.org/vocab/unit/RAD');rejects('insertion:wrong-error-unit',lambda:tool_inspect(g,action,ctx,facts,bad,window(ctx)))
def carrying_value(action,interval):
 value={'context':str(one(g,action,TC.evaluationContext)),'interval':interval,'components':{}}
 for prop,cprop in [(TC.holdingRequirement,OM.evaluationContext),(TC.mobilityRequirement,MN.evaluationContext)]:
  child=one(g,action,prop);ctx=one(g,child,cprop);facts,q=conditions(child);value['components'][child]={'context':str(ctx),'interval':window(ctx,interval['start'],interval['end']),'facts':facts,'quantities':q}
 facts,q=supplied(one(g,action,TC.localPrecondition));ff,qq=supplied(one(g,action,TC.localSuccess));facts.update(ff);q.update(qq);value.update(facts=facts,quantities=q);return value
def recipe_inputs(name):
 root=E[name];steps=ordered_steps(g,root);stages={};base=datetime(2026,11,12,10,tzinfo=timezone.utc)
 for index,step in enumerate(steps):
  action=one(g,step,N.stepAction);ctx=one(g,step,N.stepContext);win=window(ctx,(base+timedelta(seconds=12*index)).isoformat(),(base+timedelta(seconds=12*index+10)).isoformat())
  if one(g,step,N.stepRole)==N.Carry:value=carrying_value(action,win)
  else:
   facts,q=conditions(action);value={'context':str(ctx),'interval':win,'facts':facts,'quantities':q}
  stages[step]=value
 context=one(g,root,N.evaluationContext);parent=window(context,stages[steps[0]]['interval']['start'],stages[steps[-1]]['interval']['end']);handoffs={}
 for h in g.objects(root,N.hasHandoff):
  left=one(g,h,N.fromStep);right=one(g,h,N.toStep);facts,_=supplied(one(g,h,N.handoffCondition));handoffs[h]={'context':str(context),'fromStep':str(left),'toStep':str(right),'start':stages[left]['interval']['end'],'end':stages[right]['interval']['start'],'facts':facts}
 facts,_=supplied(one(g,root,N.localPrecondition));return {'stages':stages,'handoffs':handoffs,'interval':parent,'facts':facts}
def inspect(name,kw=None,graph=g):return recipe_inspect(graph,E[name+'Record'],E[name+'Context'],**({} if kw is None else kw))
names=['pickCarryPlace','routeOrientedPickCarryPlace','cuttingCycle','wipingCycle']
for name in names:
 root=E[name];steps=ordered_steps(g,root);kw=recipe_inputs(name)
 check(name+':no-history-unknown',inspect(name)['success'] is None)
 out=inspect(name,kw);check(name+':qualified-history-satisfies',out['success'] is True and out['orderedIntervalBound'])
 check(name+':no-execution-or-stage-selection',not out['executed'] and not out['executionReadinessAssessed'] and out['selectedNextStage'] is None and out['acceptedObservations']==0)
 bad=deepcopy(kw);bad['handoffs']={};check(name+':stage-successes-not-handoff-proof',inspect(name,bad)['success'] is None)
 bad=deepcopy(kw);h=next(iter(bad['handoffs']));key=next(iter(bad['handoffs'][h]['facts']));bad['handoffs'][h]['facts'][key]=False;check(name+':failed-continuity-fails-recipe',inspect(name,bad)['success'] is False)
 bad=deepcopy(kw);bad['handoffs'][h]['start']=bad['handoffs'][h]['end'];rejects(name+':handoff-must-cover-gap',lambda:inspect(name,bad))
 bad=deepcopy(kw);bad['handoffs'][h]['context']=str(E.foreignContext);rejects(name+':foreign-handoff-context',lambda:inspect(name,bad))
 bad=deepcopy(kw);bad['handoffs'][h]['toStep']=str(steps[0]);rejects(name+':wrong-handoff-stage',lambda:inspect(name,bad))
 bad=deepcopy(kw);bad['stages'][steps[0]]['context']=str(E.foreignContext);rejects(name+':foreign-stage-context',lambda:inspect(name,bad))
 bad=deepcopy(kw);bad['stages'][steps[0]]['interval']['end']=bad['stages'][steps[1]]['interval']['end'];rejects(name+':overlapping-stages',lambda:inspect(name,bad))
 bad=deepcopy(kw);bad['stages'].pop(steps[0]);rejects(name+':later-stage-not-missing-history',lambda:inspect(name,bad))
 bad=deepcopy(kw);bad['stages']={steps[0]:bad['stages'][steps[0]]};bad['handoffs']={};check(name+':qualified-prefix-not-complete',inspect(name,bad)['success'] is None)
 bad=deepcopy(kw);bad['interval']=None;rejects(name+':child-interval-needs-parent',lambda:inspect(name,bad))
 bad=deepcopy(kw);bad['stages'][steps[0]]['interval']['start']='2026-11-12T10:00:00';rejects(name+':timezone-required',lambda:inspect(name,bad))
 bad=deepcopy(kw);bad['interval']['end']='2026-11-12T10:05:00+00:00';rejects(name+':parent-end-is-final-stage-end',lambda:inspect(name,bad))
 bad=deepcopy(kw);child=one(g,steps[0],N.stepAction);ff,_=supplied(one(g,child,A.hasFailureCriterion));bad['stages'][steps[0]]['facts'].update(ff);check(name+':success-and-fault-visible',inspect(name,bad)['successAndFaultReported'])
 bad=deepcopy(kw);pre,_=supplied(one(g,child,A.hasPrecondition));key=next(iter(pre));bad['stages'][steps[0]]['facts'][key]=False;check(name+':successful-outcome-does-not-waive-prerequisite',inspect(name,bad)['success'] is False)
 bad=deepcopy(kw);bad['stages'][steps[0]].pop('interval');bad['handoffs']={h:v for h,v in bad['handoffs'].items() if one(g,h,N.fromStep)!=steps[0]};check(name+':unqualified-stage-window-not-invented',inspect(name,bad)['success'] is None)
 # Shared boundaries are allowed, but still need explicit continuity evidence.
 bad=deepcopy(kw);base=datetime(2026,11,12,10,tzinfo=timezone.utc)
 for index,step in enumerate(steps):
  win=bad['stages'][step]['interval'];win['start']=(base+timedelta(seconds=10*index)).isoformat();win['end']=(base+timedelta(seconds=10*index+10)).isoformat()
  for component in bad['stages'][step].get('components',{}).values():component['interval']['start']=win['start'];component['interval']['end']=win['end']
 bad['interval']['end']=bad['stages'][steps[-1]]['interval']['end']
 for h,value in bad['handoffs'].items():value['start']=bad['stages'][one(g,h,N.fromStep)]['interval']['end'];value['end']=bad['stages'][one(g,h,N.toStep)]['interval']['start']
 check(name+':explicit-shared-boundary-continuity',inspect(name,bad)['success'] is True)
 bad=deepcopy(kw);child=one(g,steps[1],N.stepAction)
 if one(g,steps[1],N.stepRole)==N.Carry:
  inner=one(g,child,TC.holdingRequirement);bad['stages'][steps[1]]['components'][inner].pop('interval');check(name+':nested-carry-window-not-inherited',inspect(name,bad)['success'] is None)
  # Historical held-after-pickup and released-after-place both remain true in their own contexts.
  check(name+':historical-holding-and-release-not-flattened',out['stageResults'][str(steps[0])]['success'] is True and out['stageResults'][str(steps[-1])]['success'] is True)
 if name=='wipingCycle':
  use=steps[3];action=one(g,use,N.stepAction);key=next(k for k in bad['stages'][use]['facts'] if k[1]==ST.SweepCoverageMaintained);bad['stages'][use]['facts'].pop(key);check('wiping-cycle:contact-not-coverage',inspect(name,bad)['success'] is None)

def mutate(label,subject,predicate,value,validator=recipe_errors):
 h=Graph()+g;h.set((subject,predicate,value));check('reject-'+label,bool(validator(h)))
root=E.pickCarryPlace;steps=ordered_steps(g,root);success=one(g,root,A.hasSuccessCriterion)
for row in [('duplicate-stage-index',steps[1],N.positionIndex,Literal(1,datatype=XSD.positiveInteger)),('wrong-stage-role',steps[1],N.stepRole,N.Place),('wrong-final-stage',success,N.finalStep,steps[0]),('foreign-payload',root,N.payload,E.otherPayload),('wrong-recipe-kind',root,A.hasActionKind,N.AcquireUseReturnTool),('foreign-stage-action',steps[0],N.stepAction,one(g,steps[-1],N.stepAction)),('foreign-stage-context',steps[0],N.stepContext,one(g,steps[1],N.stepContext))]:mutate(*row)
h=Graph()+g;h.remove((success,N.requiresStepSuccess,steps[1]));check('reject-omitted-carry-outcome',bool(recipe_errors(h)))
h=Graph()+g;h.remove((success,N.requiresStepPrecondition,steps[1]));check('reject-omitted-stage-prerequisite',bool(recipe_errors(h)))
h=Graph()+g;handoff=next(x for x in g.objects(root,N.hasHandoff) if one(g,x,N.fromStep)==steps[0]);h.set((handoff,N.toStep,steps[-1]));check('reject-nonadjacent-handoff',bool(recipe_errors(h)))
root=E.cuttingCycle;steps=ordered_steps(g,root);mutate('wrong-tool-return-support',root,N.destinationSupport,E.otherSupport);mutate('tool-target-changed',root,N.toolTarget,E.otherTarget);mutate('tool-stage-frame',one(g,steps[1],N.stepAction),TI.referenceFrame,E.otherFrame)
bad=recipe_inputs('cuttingCycle');bad['handoffs'][E.foreignHandoff]=next(iter(bad['handoffs'].values()));rejects('foreign-handoff-input',lambda:inspect('cuttingCycle',bad))
for label,subject,predicate,value in [('missing-technique-outcome',S.cutting,A.hasSuccessCriterion,one(g,S.pressing,A.hasSuccessCriterion)),('mixed-technique-kind',S.cutting,A.hasActionKind,ST.Wiping),('depth-target-mismatch',one(g,S.insertion,ST.depthError),ST.depthErrorTarget,one(g,S.insertion,ST.depthTolerance)),('depth-unit',one(g,S.insertion,ST.depthError),A.unitIRI,Literal('http://qudt.org/vocab/unit/RAD',datatype=XSD.anyURI)),('press-without-contact-policy',S.pressing,TI.contactPolicy,Literal('unspecified',datatype=XSD.string))]:mutate(label,subject,predicate,value,specialized_errors)
reachable=set()
for graph,ns,roots in [(data,S,[S[x] for x in ['cutting','wiping','pressing','insertion']]),(data,E,[E[x] for x in names])]:
 for root in roots:
  seen=set();todo=[root]
  while todo:
   n=todo.pop()
   if n in seen:continue
   seen.add(n);todo.extend(o for o in graph.objects(n) if isinstance(o,URIRef) and str(o).startswith(str(ns)))
  reachable.update(seen);expected={(s,str(p),o) for s in seen for p,o in graph.predicate_objects(s) if p not in [RDF.type,RDFS.label,RDFS.comment]};covered=set()
  for claim in graph.objects(ns[str(root).split('#')[-1]+'Record'],A.hasClaim):
   st=one(graph,claim,A.hasStatement);obj=list(graph.objects(st,A.statementObject))+list(graph.objects(st,A.statementLiteral));covered.add((one(graph,st,A.statementSubject),str(one(graph,st,A.statementPredicateIRI)),obj[0]))
  check(str(root).split('#')[-1]+':exact-provenance',expected==covered and bool(expected))
metadata={s for typ in [A.KnowledgeRecord,A.AnnotationRecord,A.KnowledgeClaim,A.Statement,A.EvidenceRecord] for s in data.subjects(RDF.type,typ)};metadata.update(data.objects(None,A.aboutComponent))
check('no-orphan-descriptions',not(set(data.subjects())-reachable-metadata));check('all-claims-proposed',all(data.value(x,A.hasStatus)==A.Proposed for x in data.subjects(RDF.type,A.KnowledgeClaim)));check('no-physical-parameter-defaults',not list(data.triples((None,A.numericValue,None))));check('no-world-bindings',not list(data.triples((None,A.hasBinding,None))))
(ROOT/'reports/recipes-specialized-validation.json').write_text(json.dumps({'release':'0.20.0','checks':checks,'scope':'Synthetic contracts; no physics, accepted observations or robot execution.'},indent=2)+'\n');print(json.dumps({'checks':len(checks),'passed':sum(c['passed'] for c in checks),'failures':[c for c in checks if not c['passed']]}));assert all(c['passed'] for c in checks)
