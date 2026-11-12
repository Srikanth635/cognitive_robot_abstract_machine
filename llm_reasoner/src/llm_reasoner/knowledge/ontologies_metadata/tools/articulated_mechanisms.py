"""Bounded single-coordinate mechanism authoring checks and supplied-state inspection.

No kinematics, contact solver, perception, evidence acceptance or execution.
Validate SHACL and integrity before assessment; inputs must be isolated per context.
"""
from decimal import Decimal, InvalidOperation
from rdflib import Namespace, RDF
from aspect_runtime import A, C, one, evaluate
from context_review import K
from temporal_runtime import T, M
from condition_signatures import signature

B = 'http://aicor.knowledge/ontologies/'
N = Namespace(B + 'articulated-mechanisms.owl#')
RM = Namespace(B + 'robot-motion.owl#')
OM = Namespace(B + 'object-manipulation.owl#')
PROFILE_TYPES = [N.ConfigurationChangeDescription, N.OpeningDescription,
                 N.ClosingDescription, N.LatchReleaseDescription, N.LatchSecuringDescription]
CONFIGURATION_TYPES = PROFILE_TYPES[:3]


def profile(g, action):
    """Allow materialized inheritance, reject incompatible concrete profiles."""
    kinds = {c for c in PROFILE_TYPES if (action, RDF.type, c) in g}
    if N.OpeningDescription in kinds or N.ClosingDescription in kinds:
        kinds.discard(N.ConfigurationChangeDescription)
    if len(kinds) != 1:
        raise ValueError('Select exactly one concrete mechanism profile')
    return next(iter(kinds))


def scalar(g, parameter, unit, nonnegative=False):
    if str(one(g, parameter, A.unitIRI)) != unit:
        raise ValueError('Parameter unit does not match the declared coordinate kind')
    raw = g.value(parameter, A.numericValue)
    if raw is None:
        return None
    try:
        value = Decimal(str(raw))
    except InvalidOperation as error:
        raise ValueError('Invalid scalar') from error
    if not value.is_finite() or (nonnegative and value < 0):
        raise ValueError('Invalid finite/nonnegative scalar')
    return value


def atom(scope, relation, *args):
    return scope, relation, args, False


def expression_signature(g, condition):
    return signature(g, one(g, condition, C.hasExpression))


def mechanism_errors(g):
    errors = []
    actions = {s for cls in PROFILE_TYPES + [N.ArticulatedMechanismDescription]
               for s in g.subjects(RDF.type, cls)}
    for action in actions:
        try:
            kind = profile(g, action)
            action_kinds = {N.ConfigurationChangeDescription: N.ConfiguringMechanism,
                            N.OpeningDescription: N.OpeningMechanism,
                            N.ClosingDescription: N.ClosingMechanism,
                            N.LatchReleaseDescription: N.ReleasingLatch,
                            N.LatchSecuringDescription: N.SecuringLatch}
            if one(g, action, A.hasActionKind) != action_kinds[kind]:
                raise ValueError('Canonical action kind disagrees with the declared profile')
            config = kind in CONFIGURATION_TYPES
            mech = one(g, action, N.mechanism)
            assembly = one(g, mech, N.describesAssembly)
            fixed = one(g, mech, N.fixedPart)
            moving = one(g, mech, N.movingPart)
            operator = one(g, action, N.operator)
            if len({assembly, fixed, moving, operator}) != 4:
                raise ValueError('Assembly, fixed/moving part and operator must be distinct specifications')
            if one(g, one(g, action, N.evaluationContext), K.scopedDescription) != action:
                raise ValueError('Context belongs to another description')
            joint = one(g, mech, N.joint)
            if set(g.subjects(N.joint, joint)) != {mech}:
                raise ValueError('Joint specification has another mechanism owner')
            coordinate = one(g, joint, N.coordinate)
            if set(g.subjects(N.coordinate, coordinate)) != {joint}:
                raise ValueError('Coordinate shared by different joint descriptions')
            joint_kind = one(g, joint, N.jointKind)
            unit = {N.Revolute: 'RAD', N.Prismatic: 'M'}.get(joint_kind)
            if unit is None:
                raise ValueError('Only bounded revolute/prismatic joints are supported')
            unit = 'http://qudt.org/vocab/unit/' + unit
            coordinate_kind = 'angular' if joint_kind == N.Revolute else 'linear'
            if str(one(g, coordinate, RM.coordinateKind)) != coordinate_kind:
                raise ValueError('Joint and coordinate kinds disagree')
            one(g, coordinate, RM.coordinateKey)
            axis = one(g, joint, N.axis)
            if one(g, axis, N.axisFrame) != one(g, mech, N.referenceFrame):
                raise ValueError('Axis/frame mismatch requires an explicit transform contract')
            components = [scalar(g, one(g, axis, N[c+'AxisParameter']),
                                 'http://qudt.org/vocab/unit/UNITLESS') for c in 'xyz']
            if all(v is not None for v in components):
                if abs(sum(v*v for v in components) - Decimal(1)) > Decimal('0.000001'):
                    raise ValueError('Known axis components must form a unit vector')
            origin = one(g, axis, N.axisOrigin)
            for c in 'xyz':
                scalar(g, one(g, origin, RM[c+'Parameter']), 'http://qudt.org/vocab/unit/M')
            params = {p: one(g, joint, N[p]) for p in
                      ['lowerLimit', 'upperLimit', 'openCoordinate', 'closedCoordinate']}
            values = {p: scalar(g, param, unit) for p, param in params.items()}
            lower, upper = values['lowerLimit'], values['upperLimit']
            if lower is not None and upper is not None and lower >= upper:
                raise ValueError('Joint limits must be ordered and nondegenerate')
            direction = Decimal(str(one(g, joint, N.openingDirection)))
            if direction not in [Decimal(-1), Decimal(1)]:
                raise ValueError('Opening direction must be +1 or -1')
            opened, closed = values['openCoordinate'], values['closedCoordinate']
            if opened is not None and closed is not None and direction*(opened-closed) <= 0:
                raise ValueError('Open/closed coordinates contradict the opening direction')
            for value in [opened, closed]:
                if value is not None and ((lower is not None and value < lower)
                                          or (upper is not None and value > upper)):
                    raise ValueError('Open/closed coordinate outside limits')
            for handle in g.objects(mech, N.handle):
                if one(g, handle, N.handleOf) != moving:
                    raise ValueError('Handle does not belong to the moving part')
            latches = list(g.objects(mech, N.latch))
            if len(latches) > 1:
                raise ValueError('Multiple latches require a composite profile')
            for latch in latches:
                if one(g, latch, N.latchOf) != mech:
                    raise ValueError('Latch belongs to another mechanism')
            if not config and len(latches) != 1:
                raise ValueError('Latch action requires an explicit latch')
            pre, _ = expression_signature(g, one(g, action, A.hasPrecondition))
            success, pairs = expression_signature(g, one(g, action, A.hasSuccessCriterion))
            fault, _ = expression_signature(g, one(g, action, A.hasFailureCriterion))
            required_pre = {atom(C.PreState, N.MechanismReady, operator, assembly)}
            required_success = set()
            required_fault = {atom(C.CurrentState, N.MechanismFault, operator, assembly)}
            release_required = False
            if config:
                release_required = one(g, action, N.latchReleaseRequired).toPython()
                if type(release_required) is not bool:
                    raise ValueError('Latch-release requirement must be an explicit boolean')
                if release_required and not latches:
                    raise ValueError('Release-required motion must identify its latch')
                required_pre.add(atom(C.PreState, N.PathClear, operator, assembly))
                if release_required:
                    required_pre.update(atom(C.PreState, N.LatchReleased, latch) for latch in latches)
                target = one(g, action, N.targetParameter)
                error = one(g, action, N.errorParameter)
                tolerance = one(g, action, N.toleranceParameter)
                if len({target, error, tolerance}) != 3:
                    raise ValueError('Target, error and tolerance parameters must be distinct')
                value = scalar(g, target, unit)
                scalar(g, error, unit, nonnegative=True)
                scalar(g, tolerance, unit, nonnegative=True)
                if one(g, error, N.errorCoordinate) != coordinate or one(g, error, N.errorTarget) != target:
                    raise ValueError('Error metric bound to a different coordinate/target')
                if value is not None and ((lower is not None and value < lower)
                                           or (upper is not None and value > upper)):
                    raise ValueError('Requested coordinate outside limits')
                if (C.LessOrEqual, C.PostState, error, C.TaskContext, tolerance) not in pairs:
                    raise ValueError('Success omits the bound post-error <= task-tolerance conjunct')
                if kind == N.OpeningDescription:
                    if target != params['openCoordinate']:
                        raise ValueError('Opening target must be the declared open-coordinate parameter')
                    required_success.add(atom(C.PostState, N.OpenConfiguration, assembly, moving))
                if kind == N.ClosingDescription:
                    if target != params['closedCoordinate']:
                        raise ValueError('Closing target must be the declared closed-coordinate parameter')
                    required_success.add(atom(C.PostState, N.ClosedConfiguration, assembly, moving))
            elif kind == N.LatchReleaseDescription:
                required_success.add(atom(C.PostState, N.LatchReleased, latches[0]))
            else:
                required_pre.add(atom(C.PreState, N.ClosedConfiguration, assembly, moving))
                required_success.update([atom(C.PostState, N.LatchSecured, latches[0]),
                                         atom(C.PostState, N.ClosedConfiguration, assembly, moving)])
            if not required_pre <= pre or not required_success <= success or not required_fault <= fault:
                raise ValueError('Required condition conjunct missing or bound to the wrong referent/scope')
            structure = one(g, action, T.hasPhaseStructure)
            members = set(g.objects(structure, T.hasMemberPhase))
            actuation = one(g, action, N.actuationPhase)
            verification = one(g, action, N.verificationPhase)
            if not {actuation, verification} <= members or actuation == verification:
                raise ValueError('Actuation/verification phase must be distinct members')
            ordinary = [e for e in g.objects(structure, T.hasTransition)
                        if (e, RDF.type, T.OrdinaryTransition) in g]
            initial = one(g, structure, T.initialPhase)
            complete = [p for p in members if (p, RDF.type, T.SuccessfulTerminalPhase) in g]
            if len(complete) != 1:
                raise ValueError('Exactly one successful terminal in this bounded profile')
            if config:
                engagement = one(g, action, N.engagementPhase)
                if len({initial, engagement, actuation, verification, complete[0]}) != 5:
                    raise ValueError('Preparation, engagement, actuation, verification and completion must be distinct')
                intended_edges = {(initial, engagement), (engagement, actuation),
                                  (actuation, verification), (verification, complete[0])}
            else:
                if len({initial, actuation, verification, complete[0]}) != 4:
                    raise ValueError('Preparation, actuation, verification and completion must be distinct')
                intended_edges = {(initial, actuation), (actuation, verification),
                                  (verification, complete[0])}
            actual_edges = [(one(g, e, T.fromPhase), one(g, e, T.toPhase)) for e in ordinary]
            if len(actual_edges) != len(intended_edges) or set(actual_edges) != intended_edges:
                raise ValueError('Ordinary phase path must preserve preparation/engagement/actuation/verification')
            for edge in ordinary:
                src, dst = one(g, edge, T.fromPhase), one(g, edge, T.toPhase)
                guard = one(g, edge, T.guardCondition)
                if src == initial and guard != one(g, action, A.hasPrecondition):
                    raise ValueError('Initial transition must use the action precondition')
                if src == verification and guard != one(g, action, A.hasSuccessCriterion):
                    raise ValueError('Final verification must use the action success criterion')
                if src == actuation:
                    current_atoms, current_pairs = expression_signature(g, guard)
                    expected_atoms = {(C.CurrentState if sc == C.PostState else sc, rel, args, neg)
                                      for sc, rel, args, neg in success}
                    expected_pairs = {(op, C.CurrentState if sc == C.PostState else sc, left,
                                       C.CurrentState if rsc == C.PostState else rsc, right)
                                      for op, sc, left, rsc, right in pairs}
                    if not expected_atoms <= current_atoms or not expected_pairs <= current_pairs:
                        raise ValueError('Actuation-to-verification guard omits current completion evidence')
            for phase in members:
                if (phase, RDF.type, T.SuccessfulTerminalPhase) in g or (phase, RDF.type, T.FailedTerminalPhase) in g:
                    continue
                faults = [e for e in g.objects(structure, T.hasTransition)
                          if g.value(e, T.fromPhase) == phase and (e, RDF.type, T.FailureTransition) in g]
                if not faults or not any(g.value(e, T.guardCondition) == one(g, action, A.hasFailureCriterion)
                                        and (g.value(e, T.toPhase), RDF.type, T.FailedTerminalPhase) in g for e in faults):
                    raise ValueError('Every nonterminal requires an explicit fault path to abort')
            if config:
                engagement = one(g, action, N.engagementPhase)
                if engagement not in members or engagement in [actuation, verification]:
                    raise ValueError('Engagement phase must precede distinct actuation/verification phases')
                entering = [e for e in g.objects(structure, T.hasTransition)
                            if g.value(e, T.toPhase) == actuation]
                if len(entering) != 1 or one(g, entering[0], T.fromPhase) != engagement:
                    raise ValueError('Actuation must be entered only from engagement in this bounded profile')
                guarded, _ = expression_signature(g, one(g, entering[0], T.guardCondition))
                requirements = {atom(C.CurrentState, N.ActuationEngaged, operator, assembly),
                                atom(C.CurrentState, N.PathClear, operator, assembly)}
                if release_required:
                    requirements.update(atom(C.CurrentState, N.LatchReleased, l) for l in latches)
                if not requirements <= guarded:
                    raise ValueError('Actuation entry omits current engagement/path/latch evidence')
                during = list(g.objects(actuation, M.hasMonitoringRequirement))
                if not any(requirements <= expression_signature(g, one(g, m, M.monitorsCondition))[0]
                           and g.value(m, M.atPoint) == M.DuringPhase for m in during):
                    raise ValueError('Actuation needs a current engagement/path/latch monitor')
            for child in g.objects(action, N.engagementDescription):
                intended_part = g.value(mech, N.handle) or moving
                if one(g, child, OM.manipulator) != operator or one(g, child, OM.object) != intended_part:
                    raise ValueError('Optional manipulation engagement has foreign operator/handle')
        except (ValueError, InvalidOperation) as error:
            errors.append(str(action) + ': ' + str(error))
    return errors


def assess(g, action, context, facts=None, quantities=None):
    """Evaluate supplied predicates/errors, without acquiring or accepting evidence."""
    if str(context) != str(one(g, action, N.evaluationContext)):
        raise ValueError('Explicit action context mismatch')
    if one(g, one(g, action, N.evaluationContext), K.scopedDescription) != action:
        raise ValueError('Context scopes another action')
    facts = {} if facts is None else facts
    quantities = {} if quantities is None else quantities
    kind = profile(g, action)
    if kind in CONFIGURATION_TYPES:
        for prop, scope in [(N.errorParameter, C.PostState), (N.toleranceParameter, C.TaskContext)]:
            parameter = one(g, action, prop)
            if (scope, parameter) not in quantities:
                continue
            raw, unit = quantities[(scope, parameter)]
            try:
                value = Decimal(str(raw))
            except InvalidOperation as error:
                raise ValueError('Invalid quantity') from error
            if not value.is_finite() or value < 0:
                raise ValueError('Absolute error/tolerance must be finite and nonnegative')
            if str(unit) != str(one(g, parameter, A.unitIRI)):
                raise ValueError('Input quantity unit mismatch; no conversion is inferred')
    def truth(prop):
        return evaluate(g, one(g, one(g, action, prop), C.hasExpression), facts, quantities)
    pre, success, fault = truth(A.hasPrecondition), truth(A.hasSuccessCriterion), truth(A.hasFailureCriterion)
    mech = one(g, action, N.mechanism)
    assembly, moving = one(g, mech, N.describesAssembly), one(g, mech, N.movingPart)
    opposite_reports = []
    for scope in [C.PreState, C.CurrentState, C.PostState]:
        if facts.get((scope, N.OpenConfiguration, (assembly, moving))) is True and facts.get((scope, N.ClosedConfiguration, (assembly, moving))) is True:
            opposite_reports.append('open-and-closed:' + str(scope))
        for latch in g.objects(mech, N.latch):
            if facts.get((scope, N.LatchReleased, (latch,))) is True and facts.get((scope, N.LatchSecured, (latch,))) is True:
                opposite_reports.append('released-and-secured:' + str(scope))
    return {'precondition': pre, 'success': success, 'fault': fault,
            'successAndFaultReported': success is True and fault is True,
            'incompatibleStateReports': opposite_reports,
            'executionReadinessAssessed': False, 'executed': False,
            'acceptedObservations': 0}
