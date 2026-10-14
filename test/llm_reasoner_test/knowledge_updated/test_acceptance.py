from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Optional

import pytest
from llm_reasoner.knowledge_updated.ontological.acceptance import (
    AcceptanceRule,
    AcceptanceSuite,
    AcceptanceViolation,
    Demand,
    DistinctAcross,
    Disjoint,
    Distinct,
    FlagOnPath,
    Forbidden,
    GraphShape,
    Grounded,
    Inclusion,
    MostSpecificTypes,
    Ordered,
    RequiresKind,
    RulePathError,
    TypeOrder,
    SharedValue,
    Supply,
)
from llm_reasoner.knowledge_updated.ontological.documentation import (
    DocumentationReader,
)
from llm_reasoner.knowledge_updated.ontological.entities import KnowledgeEntity
from llm_reasoner.knowledge_updated.ontological.ontology import Ontology
from llm_reasoner.knowledge_updated.ontological.properties import attribute, relation
from llm_reasoner.knowledge_updated.ontological.vocabulary import Severity

# %% a small domain to state rules over

NAMESPACE = "http://example.org/acceptance#"
UNKNOWN_FIELD = "no_such_field"


class Rule(StrEnum):
    """
    The rules the tests declare.
    """

    GATE_ENDS_SOURCE = "gate_ends_source"
    """
    A hop may only be gated by a signal that ends the stage it leaves.
    """

    STAGE_STARTS_AND_ENDS_ON_ONE_SIGNAL = "stage_starts_and_ends_on_one_signal"
    """
    A stage begun and ended by the same signal has no extent.
    """

    HOP_LEAVES_ITSELF = "hop_leaves_itself"
    """
    A hop from a stage to itself is not a hop.
    """

    GATE_IS_FAILURE = "gate_is_failure"
    """
    A failing signal must not gate a hop.
    """

    BOUNDS_INVERTED = "bounds_inverted"
    """
    A stage's lower bound must not exceed its upper bound.
    """

    GATE_SHARES_NOTHING = "gate_shares_nothing"
    """
    A hop's gate must share a signal with the stage it leaves.
    """

    STAGES_INDISTINGUISHABLE = "stages_indistinguishable"
    """
    Two stages ending on the same signals cannot be told apart.
    """

    STAGES_NOT_ONE_CHAIN = "stages_not_one_chain"
    """
    The stages must form one chain with a single entry and no orphan.
    """

    NEED_UNMET = "need_unmet"
    """
    Every need of a stage must be met by a stage or by an assumption.
    """

    HIGH_TOO_HIGH = "high_too_high"
    """
    A predicate rule for tests.
    """

    STAGE_NOT_ENDED_BY_TIMEOUT = "stage_not_ended_by_timeout"
    """
    A stage must end on a timeout.
    """

    SIGNAL_TOO_GENERAL = "signal_too_general"
    """
    A plan must name the specific kind of each of its signals.
    """

    HOP_AGAINST_THE_ORDER = "hop_against_the_order"
    """
    A hop may only lead from an earlier kind of stage to a later one.
    """


@dataclass(eq=False)
class Signal(KnowledgeEntity):
    """
    Something that can end a stage.
    """

    failing: bool = attribute(functional=True, default=False)
    """
    Whether it marks failure.
    """


@dataclass(eq=False)
class TimeoutSignal(Signal):
    """
    A signal raised when time runs out.
    """


@dataclass(eq=False)
class Need(KnowledgeEntity):
    """
    Something a stage needs.
    """

    assumed: bool = attribute(functional=True, default=False)
    """
    Whether it is presupposed of the world.
    """


@dataclass(eq=False)
class Stage(KnowledgeEntity):
    """
    A stretch of a plan.
    """

    starts_on: list[Signal] = relation()
    """
    The signals that begin it.
    """

    ends_on: list[Signal] = relation()
    """
    The signals that end it.
    """

    low: Optional[float] = attribute(functional=True)
    """
    The lower bound.
    """

    high: Optional[float] = attribute(functional=True)
    """
    The upper bound.
    """

    needs: list[Need] = relation()
    """
    What it needs.
    """

    provides: list[Need] = relation()
    """
    What it brings about.
    """


@dataclass(eq=False)
class EarlyStage(Stage):
    """
    A stage that comes first.
    """


@dataclass(eq=False)
class LateStage(Stage):
    """
    A stage that comes last.
    """


@dataclass(eq=False)
class Hop(KnowledgeEntity):
    """
    A move from one stage to the next.
    """

    from_stage: Optional[Stage] = relation(functional=True)
    """
    The stage it leaves.
    """

    to_stage: Optional[Stage] = relation(functional=True)
    """
    The stage it enters.
    """

    gated_by: list[Signal] = relation()
    """
    The signals that release it.
    """


@dataclass(eq=False)
class Plan(KnowledgeEntity):
    """
    Stages and the hops between them.
    """

    stages: list[Stage] = relation(owns=True)
    """
    The stages of the plan.
    """

    hops: list[Hop] = relation(owns=True)
    """
    The hops between the stages.
    """

    signals: list[Signal] = relation(owns=True)
    """
    The signals of the plan.
    """


def new_ontology() -> Ontology:
    """
    Return an ontology declaring the domain.
    """
    ontology = Ontology(name="acceptance")
    module = ontology.module("core", version="1.0.0", namespace=NAMESPACE)
    for cls in (Signal, TimeoutSignal, Need, Stage, EarlyStage, LateStage, Hop, Plan):
        module.concept()(cls)
    return ontology


def evaluate(
    rule: AcceptanceRule, *instances: KnowledgeEntity
) -> tuple[AcceptanceViolation, ...]:
    """
    Evaluate one rule over the given instances.
    """
    suite = AcceptanceSuite(name="test", ontology=new_ontology(), rules=[rule])
    return suite.evaluate(instances)


def fired(violations: tuple[AcceptanceViolation, ...]) -> set[str]:
    """
    Return the subjects a set of violations names.
    """
    return {violation.subject for violation in violations}


# %% comparing what two paths reach


class TestSetRules:
    """
    Rules that compare the values two paths reach.
    """

    def test_inclusion_reports_a_value_missing_from_the_target(self) -> None:
        gate, other = Signal(label="gate"), Signal(label="other")
        source = Stage(ends_on=[other])
        hop = Hop(label="hop", from_stage=source, gated_by=[gate, other])
        rule = Inclusion(
            id=Rule.GATE_ENDS_SOURCE,
            applies_to=Hop,
            source="gated_by",
            target="from_stage.ends_on",
        )

        violations = evaluate(rule, hop)

        assert fired(violations) == {"hop"}
        assert violations[0].rule is Rule.GATE_ENDS_SOURCE
        assert "gate" in violations[0].message

    def test_inclusion_is_silent_when_every_value_is_present(self) -> None:
        gate = Signal()
        hop = Hop(from_stage=Stage(ends_on=[gate]), gated_by=[gate])
        rule = Inclusion(
            id=Rule.GATE_ENDS_SOURCE,
            applies_to=Hop,
            source="gated_by",
            target="from_stage.ends_on",
        )

        assert evaluate(rule, hop) == ()

    def test_disjoint_reports_a_value_on_both_sides(self) -> None:
        shared = Signal()
        stage = Stage(label="stage", starts_on=[shared], ends_on=[shared])
        rule = Disjoint(
            id=Rule.STAGE_STARTS_AND_ENDS_ON_ONE_SIGNAL,
            applies_to=Stage,
            left="starts_on",
            right="ends_on",
        )

        assert fired(evaluate(rule, stage)) == {"stage"}

    def test_distinct_reports_two_paths_reaching_the_same_object(self) -> None:
        stage = Stage()
        hop = Hop(label="hop", from_stage=stage, to_stage=stage)
        rule = Distinct(
            id=Rule.HOP_LEAVES_ITSELF,
            applies_to=Hop,
            left="from_stage",
            right="to_stage",
        )

        assert fired(evaluate(rule, hop)) == {"hop"}

    def test_flag_on_path_reports_a_value_with_the_wrong_flag(self) -> None:
        failing = Signal(label="failing", failing=True)
        hop = Hop(label="hop", gated_by=[failing])
        rule = FlagOnPath(
            id=Rule.GATE_IS_FAILURE,
            applies_to=Hop,
            path="gated_by",
            flag="failing",
            expected=False,
        )

        assert fired(evaluate(rule, hop)) == {"hop"}

    def test_shared_value_reports_two_paths_with_nothing_in_common(self) -> None:
        hop = Hop(
            label="hop",
            from_stage=Stage(ends_on=[Signal()]),
            gated_by=[Signal()],
        )
        rule = SharedValue(
            id=Rule.GATE_SHARES_NOTHING,
            applies_to=Hop,
            left="from_stage.ends_on",
            right="gated_by",
        )

        assert fired(evaluate(rule, hop)) == {"hop"}

    def test_shared_value_is_silent_when_a_path_is_empty(self) -> None:
        hop = Hop(from_stage=Stage(ends_on=[Signal()]))
        rule = SharedValue(
            id=Rule.GATE_SHARES_NOTHING,
            applies_to=Hop,
            left="from_stage.ends_on",
            right="gated_by",
        )

        assert evaluate(rule, hop) == ()


# %% numbers and collections


class TestOrderAndDistinguishability:
    """
    Rules over ordered values and over the members of a collection.
    """

    def test_ordered_reports_a_lower_bound_above_the_upper_bound(self) -> None:
        stage = Stage(label="stage", low=5.0, high=2.0)
        rule = Ordered(
            id=Rule.BOUNDS_INVERTED, applies_to=Stage, lower="low", upper="high"
        )

        assert fired(evaluate(rule, stage)) == {"stage"}

    def test_ordered_is_silent_when_a_bound_is_missing_or_bounds_are_equal(
        self,
    ) -> None:
        rule = Ordered(
            id=Rule.BOUNDS_INVERTED, applies_to=Stage, lower="low", upper="high"
        )

        assert evaluate(rule, Stage(low=1.0), Stage(low=2.0, high=2.0)) == ()

    def test_distinct_across_reports_members_sharing_all_their_values(self) -> None:
        signal = Signal()
        plan = Plan(
            stages=[
                Stage(label="a", ends_on=[signal]),
                Stage(label="b", ends_on=[signal]),
                Stage(label="c", ends_on=[Signal()]),
            ]
        )
        rule = DistinctAcross(
            id=Rule.STAGES_INDISTINGUISHABLE,
            applies_to=Plan,
            severity=Severity.WARNING,
            collection="stages",
            key_path="ends_on",
        )

        violations = evaluate(rule, plan)

        assert fired(violations) == {"a"}
        assert violations[0].severity is Severity.WARNING

    def test_distinct_across_ignores_members_with_no_values(self) -> None:
        plan = Plan(stages=[Stage(), Stage()])
        rule = DistinctAcross(
            id=Rule.STAGES_INDISTINGUISHABLE,
            applies_to=Plan,
            collection="stages",
            key_path="ends_on",
        )

        assert evaluate(rule, plan) == ()


# %% graphs


class TestGraphShape:
    """
    A collection of members joined by edges must form one entered graph.
    """

    def rule(self) -> GraphShape:
        return GraphShape(
            id=Rule.STAGES_NOT_ONE_CHAIN,
            applies_to=Plan,
            container="stages",
            edges="hops",
            tail="from_stage",
            head="to_stage",
        )

    def test_single_chain_is_accepted(self) -> None:
        first, second = Stage(), Stage()
        plan = Plan(
            stages=[first, second], hops=[Hop(from_stage=first, to_stage=second)]
        )

        assert evaluate(self.rule(), plan) == ()

    def test_two_entries_are_reported(self) -> None:
        plan = Plan(label="plan", stages=[Stage(), Stage()])

        assert "plan" in fired(evaluate(self.rule(), plan))

    def test_an_orphan_is_reported_by_name(self) -> None:
        first, second, orphan = Stage(), Stage(), Stage(label="orphan")
        plan = Plan(
            stages=[first, second, orphan],
            hops=[Hop(from_stage=first, to_stage=second)],
        )

        assert "orphan" in fired(evaluate(self.rule(), plan))


# %% grounding


class TestGrounded:
    """
    Every need must be supplied by something, or be declared an assumption.
    """

    def rule(self) -> Grounded:
        return Grounded(
            id=Rule.NEED_UNMET,
            applies_to=Plan,
            severity=Severity.WARNING,
            container="stages",
            demand=Demand(path="needs"),
            supplies=(Supply(path="stages.provides"),),
            excused_when=lambda need: need.assumed,
        )

    def test_need_met_by_another_stage_is_accepted(self) -> None:
        need = Need()
        plan = Plan(stages=[Stage(needs=[need]), Stage(provides=[need])])

        assert evaluate(self.rule(), plan) == ()

    def test_need_met_by_nothing_is_reported(self) -> None:
        plan = Plan(stages=[Stage(label="needy", needs=[Need(label="water")])])

        violations = evaluate(self.rule(), plan)

        assert fired(violations) == {"needy"}
        assert "water" in violations[0].message

    def test_assumed_need_is_excused(self) -> None:
        plan = Plan(stages=[Stage(needs=[Need(assumed=True)])])

        assert evaluate(self.rule(), plan) == ()


# %% predicate rules


class TestForbidden:
    """
    A predicate rule fires whenever its guard holds.
    """

    def rule(self) -> Forbidden:
        return Forbidden(
            id=Rule.HIGH_TOO_HIGH,
            applies_to=Stage,
            violated_when=lambda stage: stage.high is not None and stage.high > 10,
            message="the upper bound is above ten",
        )

    def test_rule_fires_when_the_guard_holds(self) -> None:
        violations = evaluate(self.rule(), Stage(label="tall", high=11.0))

        assert fired(violations) == {"tall"}
        assert violations[0].message == "the upper bound is above ten"

    def test_rule_is_silent_when_the_guard_does_not_hold(self) -> None:
        assert evaluate(self.rule(), Stage(high=3.0)) == ()

    def test_rule_applies_only_to_its_subject_type(self) -> None:
        assert evaluate(self.rule(), Signal()) == ()

    def test_rule_with_a_precondition_skips_instances_it_rejects(self) -> None:
        rule = Forbidden(
            id=Rule.HIGH_TOO_HIGH,
            applies_to=Stage,
            violated_when=lambda stage: True,
            only_if=lambda stage: stage.high is not None,
            message="always",
        )

        assert evaluate(rule, Stage()) == ()
        assert evaluate(rule, Stage(high=1.0)) != ()


# %% kinds


class TestRequiresKind:
    """
    Some value at a path must be of a kind.
    """

    def rule(self) -> RequiresKind:
        return RequiresKind(
            id=Rule.STAGE_NOT_ENDED_BY_TIMEOUT,
            applies_to=Stage,
            path="ends_on",
            kind=TimeoutSignal,
        )

    def test_rule_is_silent_when_a_value_is_of_the_kind(self) -> None:
        stage = Stage(ends_on=[Signal(), TimeoutSignal()])

        assert evaluate(self.rule(), stage) == ()

    def test_rule_fires_when_no_value_is_of_the_kind(self) -> None:
        stage = Stage(label="plain", ends_on=[Signal()])

        assert fired(evaluate(self.rule(), stage)) == {"plain"}

    def test_rule_fires_when_nothing_is_reached(self) -> None:
        assert fired(evaluate(self.rule(), Stage(label="empty"))) == {"empty"}

    def test_unknown_field_is_refused_when_the_suite_is_built(self) -> None:
        rule = RequiresKind(
            id=Rule.STAGE_NOT_ENDED_BY_TIMEOUT,
            applies_to=Stage,
            path=UNKNOWN_FIELD,
            kind=TimeoutSignal,
        )

        with pytest.raises(RulePathError):
            AcceptanceSuite(name="test", ontology=new_ontology(), rules=[rule])


class TestMostSpecificTypes:
    """
    Members of a collection must use the most specific kind the ontology offers.
    """

    def rule(self) -> MostSpecificTypes:
        return MostSpecificTypes(
            id=Rule.SIGNAL_TOO_GENERAL, applies_to=Plan, collection="signals"
        )

    def test_rule_fires_for_a_kind_that_has_a_more_specific_one(self) -> None:
        plan = Plan(signals=[Signal(label="vague")])

        violations = evaluate(self.rule(), plan)

        assert fired(violations) == {"vague"}
        assert TimeoutSignal.__name__ in violations[0].message

    def test_rule_is_silent_for_a_kind_that_nothing_specialises(self) -> None:
        assert evaluate(self.rule(), Plan(signals=[TimeoutSignal()])) == ()

    def test_unknown_field_is_refused_when_the_suite_is_built(self) -> None:
        rule = MostSpecificTypes(
            id=Rule.SIGNAL_TOO_GENERAL, applies_to=Plan, collection=UNKNOWN_FIELD
        )

        with pytest.raises(RulePathError):
            AcceptanceSuite(name="test", ontology=new_ontology(), rules=[rule])


class TestTypeOrder:
    """
    Hops must lead forward through a declared order of kinds.
    """

    def rule(self) -> TypeOrder:
        return TypeOrder(
            id=Rule.HOP_AGAINST_THE_ORDER,
            applies_to=Plan,
            edges="hops",
            tail="from_stage",
            head="to_stage",
            expected_types=(EarlyStage, LateStage),
        )

    def plan(self, tail: Stage, head: Stage) -> Plan:
        hop = Hop(label="hop", from_stage=tail, to_stage=head)
        return Plan(stages=[tail, head], hops=[hop])

    def test_rule_is_silent_for_a_hop_that_goes_forward(self) -> None:
        assert evaluate(self.rule(), self.plan(EarlyStage(), LateStage())) == ()

    def test_rule_fires_for_a_hop_that_goes_backward(self) -> None:
        violations = evaluate(self.rule(), self.plan(LateStage(), EarlyStage()))

        assert fired(violations) == {"hop"}
        assert LateStage.__name__ in violations[0].message

    def test_rule_fires_for_a_hop_that_stays_in_one_kind(self) -> None:
        assert evaluate(self.rule(), self.plan(EarlyStage(), EarlyStage())) != ()

    def test_rule_ignores_a_stage_of_no_listed_kind(self) -> None:
        assert evaluate(self.rule(), self.plan(Stage(), EarlyStage())) == ()

    def test_unknown_field_is_refused_when_the_suite_is_built(self) -> None:
        rule = TypeOrder(
            id=Rule.HOP_AGAINST_THE_ORDER,
            applies_to=Plan,
            edges="hops",
            tail=UNKNOWN_FIELD,
            head="to_stage",
            expected_types=(EarlyStage,),
        )

        with pytest.raises(RulePathError):
            AcceptanceSuite(name="test", ontology=new_ontology(), rules=[rule])


# %% the suite


class TestSuitePaths:
    """
    A suite refuses rules that follow fields their subject cannot reach.
    """

    def test_unknown_field_is_refused_when_the_suite_is_built(self) -> None:
        rule = Inclusion(
            id=Rule.GATE_ENDS_SOURCE,
            applies_to=Hop,
            source=UNKNOWN_FIELD,
            target="from_stage.ends_on",
        )

        with pytest.raises(RulePathError) as refusal:
            AcceptanceSuite(name="test", ontology=new_ontology(), rules=[rule])

        assert refusal.value.rule is Rule.GATE_ENDS_SOURCE
        assert refusal.value.unknown_step == UNKNOWN_FIELD
        assert refusal.value.subject_type is Hop

    def test_unknown_field_relative_to_a_collection_is_refused(self) -> None:
        rule = DistinctAcross(
            id=Rule.STAGES_INDISTINGUISHABLE,
            applies_to=Plan,
            collection="stages",
            key_path=UNKNOWN_FIELD,
        )

        with pytest.raises(RulePathError):
            AcceptanceSuite(name="test", ontology=new_ontology(), rules=[rule])


class TestSuiteEvaluation:
    """
    A suite runs every rule over every instance it is given.
    """

    def test_violations_from_all_rules_are_collected(self) -> None:
        stage = Stage(label="stage", low=5.0, high=20.0)
        rules = [
            Ordered(
                id=Rule.BOUNDS_INVERTED, applies_to=Stage, lower="low", upper="high"
            ),
            Forbidden(
                id=Rule.HIGH_TOO_HIGH,
                applies_to=Stage,
                violated_when=lambda s: s.high > 10,
                message="too high",
            ),
        ]
        suite = AcceptanceSuite(name="test", ontology=new_ontology(), rules=rules)

        assert {v.rule for v in suite.evaluate([stage])} == {Rule.HIGH_TOO_HIGH}


# %% documentation


class TestAcceptanceVocabularyIsDocumented:
    """
    Every field of the acceptance vocabulary carries documentation.
    """

    @pytest.mark.parametrize(
        "dataclass_type",
        [AcceptanceViolation, Demand, Supply, AcceptanceSuite, RulePathError],
    )
    def test_every_field_is_documented(self, dataclass_type: type) -> None:
        assert DocumentationReader().find_undocumented_fields(dataclass_type) == ()
