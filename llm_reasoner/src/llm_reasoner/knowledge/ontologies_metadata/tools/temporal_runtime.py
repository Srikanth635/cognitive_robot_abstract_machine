"""Reference inspection semantics. Does not advance phases or execute responses."""
from collections import defaultdict
from rdflib import Namespace,RDF
from aspect_runtime import A,P,C,KINDS,one,evaluate
B='http://aicor.knowledge/ontologies/'
T=Namespace(B+'phases-transitions.owl#');M=Namespace(B+'monitoring-requirements.owl#')
TRANSITION_TYPES=[T.OrdinaryTransition,T.FailureTransition,T.RecoveryTransition]
def typed(g,n,classes):return any((n,RDF.type,c) in g for c in classes)
def dependencies(g,expression):
 """Unique observation leaves. Constants need no external observation."""
 result=set();visiting=set()
 def visit(n):
  if n in visiting:raise ValueError('Cyclic expression')
  visiting.add(n)
  try:
   if (n,RDF.type,C.AtomicCondition) in g:result.add(('relation',one(g,n,C.hasPattern),one(g,n,C.inScope)))
   elif (n,RDF.type,C.ParameterReference) in g:result.add(('quantity',one(g,n,C.parameter),one(g,n,C.inScope)))
   elif (n,RDF.type,C.QuantityConstant) in g:pass
   else:
    for p in [C.hasOperand,C.leftOperand,C.rightOperand]:
     for child in g.objects(n,p):visit(child)
  finally:visiting.remove(n)
 visit(expression);return result

def temporal_errors(g):
 errors=[];owners=defaultdict(list)
 for structure in g.subjects(RDF.type,T.PhaseStructure):
  try:
   members=set(g.objects(structure,T.hasMemberPhase));start=one(g,structure,T.initialPhase);edges=list(g.objects(structure,T.hasTransition));adj=defaultdict(set)
   if start not in members:errors.append('initial phase outside structure')
   terminal={n for n in members if typed(g,n,[T.TerminalPhase,T.SuccessfulTerminalPhase,T.FailedTerminalPhase])}
   for edge in edges:
    owners[edge].append(structure);src=one(g,edge,T.fromPhase);dst=one(g,edge,T.toPhase)
    if src not in members or dst not in members:errors.append('transition endpoint outside structure')
    if src in terminal:errors.append('terminal phase has outgoing transition')
    adj[src].add(dst)
   seen=set();todo=[start]
   while todo:
    n=todo.pop()
    if n not in seen:seen.add(n);todo.extend(adj[n]-seen)
   if members-seen:errors.append('unreachable phase')
   if not terminal or not terminal&seen:errors.append('no reachable terminal phase')
   for n in members-terminal:
    if not adj[n]:errors.append('nonterminal dead-end phase')
  except ValueError as e:errors.append(str(e))
 all_edges={n for k in [T.PhaseTransition]+TRANSITION_TYPES for n in g.subjects(RDF.type,k)}
 for edge in all_edges:
  if len(owners[edge])!=1:errors.append('transition must have one owning structure')
 # These checks validate topology, not reachability under satisfiable guards or termination.
 for monitor in g.subjects(RDF.type,M.MonitoringRequirement):
  try:
   condition=one(g,monitor,M.monitorsCondition);need=dependencies(g,one(g,condition,C.hasExpression));given=[]
   for obs in g.objects(monitor,M.requiresObservation):
    scope=one(g,obs,M.inScope)
    if (obs,RDF.type,M.RelationObservation) in g:given.append(('relation',one(g,obs,M.observesPattern),scope))
    elif (obs,RDF.type,M.QuantityObservation) in g:given.append(('quantity',one(g,obs,M.observesParameter),scope))
    else:errors.append('observation has no concrete type')
   if set(given)!=need:errors.append('monitor observation dependencies do not match expression')
   if len(given)!=len(set(given)):errors.append('duplicate observation dependency')
   hosts=list(g.subjects(M.hasMonitoringRequirement,monitor))
   if not hosts:errors.append('monitor has no phase or transition attachment')
   point=one(g,monitor,M.atPoint)
   for host in hosts:
    if point==M.TransitionCheck:
     if not typed(g,host,[T.PhaseTransition]+TRANSITION_TYPES):errors.append('transition monitor attached to non-transition')
    elif not typed(g,host,[A.PhaseSpecification,T.TerminalPhase,T.SuccessfulTerminalPhase,T.FailedTerminalPhase,T.RecoveryPhase]):errors.append('phase monitor attached to non-phase')
  except ValueError as e:errors.append(str(e))
 return errors

def assess_outgoing(g,structure,phase,facts=None,quantities=None):
 """Report guard truth only; never select or execute an edge.

 Even a uniquely true guard is not execution approval: entry/exit/invariants,
 evidence acceptance, timing and robot feasibility require an external adapter.
 """
 if (structure,T.hasMemberPhase,phase) not in g:raise ValueError('Phase is not in the structure')
 result=[]
 for edge in sorted(g.objects(structure,T.hasTransition)):
  if one(g,edge,T.fromPhase)!=phase:continue
  cond=one(g,edge,T.guardCondition);value=evaluate(g,one(g,cond,C.hasExpression),facts,quantities)
  result.append({'transition':str(edge),'target':str(one(g,edge,T.toPhase)),'guard_value':value})
 true=[r['transition'] for r in result if r['guard_value'] is True];unknown=[r['transition'] for r in result if r['guard_value'] is None]
 status='terminal' if not result and typed(g,phase,[T.TerminalPhase,T.SuccessfulTerminalPhase,T.FailedTerminalPhase]) else 'ambiguous' if len(true)>1 else 'incomplete' if unknown else 'one_guard_true' if len(true)==1 else 'no_guard_true'
 return {'phase':str(phase),'assessment':status,'transitions':result,'selected_transition':None}

def assess_monitor(g,monitor,facts=None,quantities=None):
 condition=one(g,monitor,M.monitorsCondition);value=evaluate(g,one(g,condition,C.hasExpression),facts,quantities)
 response=one(g,monitor,M.whenUnknown if value is None else M.whenTrue if value else M.whenFalse)
 return {'monitor':str(monitor),'condition_value':value,'declared_response':str(response),'response_executed':False}
