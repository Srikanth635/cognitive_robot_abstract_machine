"""Bounded coextensive holding/mobility composition, not an execution scheduler.

Load the ontology closure and validate shapes/integrity before inspection. Each
component consumes a separate explicitly scoped input dictionary. Never flatten
raw facts into one context or evaluate the parent expression as execution policy.
"""
from decimal import Decimal, InvalidOperation
from rdflib import Namespace, RDF
from aspect_runtime import A, C, one, evaluate
from context_review import K, instant
from condition_signatures import signature
from object_manipulation import N as OM, assess as assess_holding, manipulation_errors
from mobility_navigation import N as MN, assess as assess_mobility, mobility_errors, family
from temporal_runtime import T, M

B='http://aicor.knowledge/ontologies/'
N=Namespace(B+'transport-carrying.owl#')
RM=Namespace(B+'robot-motion.owl#')

def atom(scope,relation,*args):return scope,relation,args,False

def numeric(g,p,unit,nonnegative=False):
    if str(one(g,p,A.unitIRI))!=unit:raise ValueError('Declared parameter unit mismatch')
    raw=g.value(p,A.numericValue)
    if raw is None:return None
    try:value=Decimal(str(raw))
    except InvalidOperation as error:raise ValueError('Invalid numeric parameter') from error
    if not value.is_finite() or (nonnegative and value<0):raise ValueError('Invalid finite/nonnegative parameter')
    return value

def carrying_errors(g):
    errors=[]
    try:errors.extend(manipulation_errors(g));errors.extend(mobility_errors(g))
    except ValueError as error:errors.append(str(error))
    actions=set(g.subjects(RDF.type,N.CarryingDescription))|set(g.subjects(RDF.type,N.RouteCarryingDescription))
    for action in actions:
        try:
            route=(action,RDF.type,N.RouteCarryingDescription) in g
            if one(g,action,A.hasActionKind)!=(N.RouteCarrying if route else N.Carrying):raise ValueError('Action kind/profile mismatch')
            hold=one(g,action,N.holdingRequirement);nav=one(g,action,N.mobilityRequirement)
            if hold==nav or action in [hold,nav] or set(g.objects(action,A.hasSubActionDescription))!={hold,nav}:raise ValueError('Exactly the declared distinct holding and mobility requirements must be composed')
            if (hold,RDF.type,OM.HoldingDescription) not in g:raise ValueError('Carrying requires maintained holding, not pickup/release/fixation alone')
            if any((hold,RDF.type,OM[k]) in g for k in ['PickupDescription','ReleaseDescription','PlacementDescription','FixationDescription']):raise ValueError('Mixed manipulation techniques require a different composition')
            if family(g,nav)!=('RouteNavigationDescription' if route else 'NavigationDescription'):raise ValueError('Mobility family disagrees with carrying profile')
            if one(g,hold,A.hasActionKind)!=OM.Holding:raise ValueError('Holding component action kind mismatch')
            if one(g,nav,A.hasActionKind)!=(MN.FollowingRoute if route else MN.Navigating):raise ValueError('Mobility component action kind mismatch')
            carrier=one(g,action,N.carrier);load=one(g,action,N.load);man=one(g,action,N.manipulator);target=one(g,action,N.destination)
            if len({carrier,load,man,target})!=4:raise ValueError('Carrier/load/manipulator/destination must be distinct specifications')
            if one(g,hold,OM.object)!=load or one(g,hold,OM.manipulator)!=man or one(g,man,N.manipulatorOf)!=carrier:raise ValueError('Holding object/manipulator/carrier binding mismatch')
            if one(g,nav,MN.robot)!=carrier or one(g,nav,MN.targetRegion)!=target:raise ValueError('Mobility carrier/destination mismatch')
            contexts=[one(g,action,N.evaluationContext),one(g,hold,OM.evaluationContext),one(g,nav,MN.evaluationContext)]
            if len(set(contexts))!=3 or any(one(g,c,K.scopedDescription)!=a for c,a in zip(contexts,[action,hold,nav])):raise ValueError('Parent/holding/mobility contexts must be distinct and scoped correctly')
            if str(one(g,action,N.intervalPolicy))!='coextensive':raise ValueError('Only explicit coextensive interval policy supported')
            foot=one(g,action,N.loadedFootprint)
            if one(g,nav,MN.footprint)!=foot or one(g,foot,N.footprintCarrier)!=carrier or one(g,foot,N.footprintLoad)!=load:raise ValueError('Navigation must use the same carrier/load envelope specification')
            roles={e for role in g.objects(action,A.hasRoleSpecification) for e in g.objects(role,A.hasEntitySpecification)}
            if not {carrier,load,man,target,foot}<=roles:raise ValueError('Parent participant roles disagree with composed bindings')
            pre=one(g,action,N.localPrecondition);success=one(g,action,N.localSuccess);fault=one(g,action,N.localFailure)
            pre_atoms,_=signature(g,one(g,pre,C.hasExpression));success_atoms,pairs=signature(g,one(g,success,C.hasExpression));fault_atoms,_=signature(g,one(g,fault,C.hasExpression))
            if pre_atoms!={atom(C.TaskContext,N.LoadedEnvelopeQualified,carrier,load,foot),atom(C.PreState,N.LoadBorneBy,carrier,load)}:raise ValueError('Local precondition must require loaded-envelope qualification and an already borne load')
            if success_atoms!={atom(C.PostState,N.LoadEnvelopeMaintained,carrier,load,foot),atom(C.PostState,N.LoadBorneThroughout,carrier,load),atom(C.PostState,MN.InRegion,load,target)}:raise ValueError('Local success must require interval envelope/load support and actual load arrival')
            if fault_atoms!={atom(C.CurrentState,N.LoadTransportFault,carrier,load)}:raise ValueError('Fault must use explicit carried-load fault evidence')
            requirements=list(g.objects(action,N.orientationRequirement))
            if len(requirements)>1:raise ValueError('Multiple orientation requirements need an extended composition')
            if requirements:
                req=requirements[0]
                if one(g,req,N.orientedLoad)!=load:raise ValueError('Orientation constraint applies to another load')
                one(g,req,N.orientationFrame);orientation=one(g,req,N.orientationTarget)
                if str(one(g,orientation,RM.orientationConvention))!='quaternion-xyzw':raise ValueError('Unsupported orientation convention')
                parameters=[one(g,orientation,RM[c+'Parameter']) for c in ['qx','qy','qz','qw']]
                if len(set(parameters))!=4:raise ValueError('Quaternion parameters must be distinct')
                values=[numeric(g,p,'http://qudt.org/vocab/unit/UNITLESS') for p in parameters]
                if all(v is not None for v in values) and abs(sum(v*v for v in values)-Decimal(1))>Decimal('0.000001'):raise ValueError('Known quaternion must be normalized')
                error=one(g,req,N.maximumDeviation);tolerance=one(g,req,N.deviationTolerance)
                if error==tolerance or error in parameters or tolerance in parameters:raise ValueError('Target/error/tolerance parameter alias')
                for p in [error,tolerance]:numeric(g,p,'http://qudt.org/vocab/unit/RAD',True)
                if pairs!={(C.LessOrEqual,C.PostState,error,C.TaskContext,tolerance)}:raise ValueError('Orientation success must check interval maximum angular deviation')
            elif pairs:raise ValueError('Undeclared orientation metric in local success')
            # Preserve exact child expression identities and all three terms, never
            # replace holding with arrival, or change the conjunction to disjunction.
            for prop,local,logical in [(A.hasPrecondition,pre,C.Conjunction),(A.hasSuccessCriterion,success,C.Conjunction),(A.hasFailureCriterion,fault,C.Disjunction)]:
                expression=one(g,one(g,action,prop),C.hasExpression)
                required={one(g,one(g,child,prop),C.hasExpression) for child in [hold,nav]}|{one(g,local,C.hasExpression)}
                if (expression,RDF.type,logical) not in g or set(g.objects(expression,C.hasOperand))!=required:raise ValueError('Parent condition must retain exact component/local expression references')
            phase=one(g,action,N.motionPhase);structure=one(g,nav,T.hasPhaseStructure)
            if (structure,T.hasMemberPhase,phase) not in g or (phase,RDF.type,T.SuccessfulTerminalPhase) in g or (phase,RDF.type,T.FailedTerminalPhase) in g:raise ValueError('Load monitor must attach to a nonterminal mobility phase')
            initial=one(g,structure,T.initialPhase)
            incoming=[e for e in g.objects(structure,T.hasTransition) if g.value(e,T.toPhase)==phase and (e,RDF.type,T.OrdinaryTransition) in g]
            if phase==initial or len(incoming)!=1 or one(g,incoming[0],T.fromPhase)!=initial:raise ValueError('Monitored phase must be the guarded mobility execution phase')
            needed={atom(C.CurrentState,OM.Holds,man,load),atom(C.CurrentState,N.LoadEnvelopeClear,carrier,load,foot),atom(C.CurrentState,N.LoadBorneBy,carrier,load)}
            monitors=[m for m in g.objects(phase,M.hasMonitoringRequirement) if g.value(m,N.monitorContext)==contexts[0]]
            if len(monitors)!=1:raise ValueError('One composition-scoped during-motion load monitor required')
            monitor=monitors[0];actual,_=signature(g,one(g,one(g,monitor,M.monitorsCondition),C.hasExpression))
            if actual!=needed or one(g,monitor,M.atPoint)!=M.DuringPhase or one(g,monitor,M.whenUnknown)!=M.RequestSuspension:raise ValueError('Load monitor omits current hold/envelope or unknown suspension declaration')
        except (ValueError,InvalidOperation) as error:errors.append(str(action)+': '+str(error))
    return errors

def conjunction(values):return False if False in values else None if None in values else True
def disjunction(values):return True if True in values else None if None in values else False

def bounds(interval,context):
    if not isinstance(interval,dict) or not {'context','start','end'}<=set(interval):raise ValueError('Interval context/start/end required')
    if str(interval['context'])!=str(context):raise ValueError('Interval belongs to another context')
    start,end=instant(interval['start']),instant(interval['end'])
    if start>=end:raise ValueError('Interval start must precede end')
    return start,end

def assess_description(g,action,context,components=None,interval=None,facts=None,quantities=None):
    """Compose separately evaluated child answers over an exact shared window.

    Each supplied component has explicit context, facts, quantities and optional
    interval. A child interval cannot be guessed from the parent's interval.
    Observational correctness, freshness and coverage remain external contracts.
    """
    if str(context)!=str(one(g,action,N.evaluationContext)) or one(g,one(g,action,N.evaluationContext),K.scopedDescription)!=action:raise ValueError('Record/action/context mismatch')
    hold=one(g,action,N.holdingRequirement);nav=one(g,action,N.mobilityRequirement)
    components={} if components is None else components
    if set(components)-{hold,nav}:raise ValueError('Foreign component input')
    parent_bounds=None if interval is None else bounds(interval,context)
    values={};bound=[]
    for child,prop in [(hold,OM.evaluationContext),(nav,MN.evaluationContext)]:
        value=components.get(child)
        if value is None:
            values[child]={};bound.append(False);continue
        if not isinstance(value,dict) or 'context' not in value:raise ValueError('Supplied child inputs require explicit context')
        expected=one(g,child,prop)
        if str(value['context'])!=str(expected) or one(g,expected,K.scopedDescription)!=child:raise ValueError('Child input context mismatch')
        child_interval=value.get('interval')
        if child_interval is not None:
            if parent_bounds is None:raise ValueError('Child intervals require an explicit parent interval')
            if bounds(child_interval,expected)!=parent_bounds:raise ValueError('Child and parent intervals must be coextensive')
        values[child]=value;bound.append(child_interval is not None)
    coextensive=parent_bounds is not None and all(bound)
    h=values[hold];m=values[nav]
    holding=assess_holding(g,hold,h.get('facts'),h.get('interval'))
    mobility=assess_mobility(g,nav,m.get('facts'),m.get('quantities'),m.get('interval'))
    facts={} if facts is None else dict(facts);quantities={} if quantities is None else dict(quantities)
    for requirement in g.objects(action,N.orientationRequirement):
        for prop,scope in [(N.maximumDeviation,C.PostState),(N.deviationTolerance,C.TaskContext)]:
            p=one(g,requirement,prop)
            if (scope,p) not in quantities:continue
            raw,unit=quantities[(scope,p)]
            try:v=Decimal(str(raw))
            except InvalidOperation as error:raise ValueError('Invalid deviation quantity') from error
            if not v.is_finite() or v<0 or str(unit)!=str(one(g,p,A.unitIRI)):raise ValueError('Invalid finite/nonnegative angular deviation or unit')
    if not coextensive:
        facts={key:value for key,value in facts.items() if key[1] not in [N.LoadEnvelopeMaintained,N.LoadBorneThroughout]}
        for requirement in g.objects(action,N.orientationRequirement):quantities.pop((C.PostState,one(g,requirement,N.maximumDeviation)),None)
    def local(prop):return evaluate(g,one(g,one(g,action,prop),C.hasExpression),facts,quantities)
    local_pre,local_success,local_failure=local(N.localPrecondition),local(N.localSuccess),local(N.localFailure)
    success=conjunction([holding['success'],mobility['success'],local_success])
    fault=disjunction([holding['failure'],mobility['failure'],local_failure])
    return {'precondition':conjunction([holding['precondition'],mobility['precondition'],local_pre]),
            'success':success,'failure':fault,'parentIntervalBound':parent_bounds is not None,
            'coextensiveIntervalBound':coextensive,'holding':holding,'mobility':mobility,
            'localSuccess':local_success,'successAndFaultReported':success is True and fault is True,
            'executionReadinessAssessed':False,'executed':False,'acceptedObservations':0,
            'releasePerformed':False,'placementAssessed':False}


def assess(g,record,context,components=None,interval=None,facts=None,quantities=None):
    """Inspect the description explicitly selected through its root record."""
    return assess_description(g,one(g,record,A.expressesDescription),context,components,interval,facts,quantities)
