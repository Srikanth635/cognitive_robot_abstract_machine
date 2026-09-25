from __future__ import annotations

from llm_reasoner.knowledge.microworlds.flanagan.constraints import SUITE
from llm_reasoner.knowledge.microworlds.flanagan.core import (
    EstimateUpdateDirection,
    ForwardModel,
    InverseModel,
    ObjectPropertyEstimate,
    RoleSlot,
    Subgoal,
)
from llm_reasoner.knowledge.microworlds.flanagan.object_dynamics import (
    CoordinateFrame,
    DesiredState,
    DynamicsPerturbation,
    DynamicsTransfer,
    EfferenceCopy,
    InternalModelComposition,
    LimbDynamicsModel,
    MemoryPersistence,
    MotorCommand,
    ObjectDynamicsModel,
    ObjectDynamicsPhaseStructure,
    PerceptualRepresentation,
    PerturbationKind,
    RepresentationAccuracy,
    RepresentationDissociation,
    RepresentationPurpose,
    RepresentationUpdate,
    SensorimotorMemory,
    TransferOutcome,
)
from llm_reasoner.knowledge.ontological.decorators import ConceptReference

# %% test data


def make_role() -> RoleSlot:
    """
    Create an object role for representation tests.
    """
    return RoleSlot(
        label="manipulated object",
        binds_as=ConceptReference(
            iri="http://example.org/ontology/flanagan-core#Manipulandum",
            label="Manipulandum",
        ),
    )


def make_estimate(label: str) -> ObjectPropertyEstimate:
    """
    Create a quantitative object estimate with a known dimension.
    """
    return ObjectPropertyEstimate(
        label=label,
        parameter_range=None,
        parameter_on_role=[make_role()],
    )


def make_object_model(
    estimate: ObjectPropertyEstimate,
    purpose: RepresentationPurpose = RepresentationPurpose.SENSORIMOTOR_CONTROL,
) -> ObjectDynamicsModel:
    """
    Create an object representation with explicit context.
    """
    command = MotorCommand(label="load command")
    efference_copy = EfferenceCopy(
        label="copy of load command",
        copies_command=command,
    )
    return ObjectDynamicsModel(
        label="object dynamics",
        holds_estimate=[estimate],
        coordinate_frame=CoordinateFrame.OBJECT_CENTERED,
        purpose=purpose,
        represents_role=estimate.parameter_on_role,
        uses_efference_copy=[efference_copy],
    )


# %% control composition


class TestInternalModelComposition:
    """
    The control chain keeps every causal stage explicit.
    """

    def test_composition_connects_desire_command_copy_dynamics_and_prediction(
        self,
    ) -> None:
        desired_state = DesiredState(
            label="stable lifted state",
            desired_condition=[Subgoal(label="object is stably lifted")],
        )
        inverse_model = InverseModel(label="load inverse model")
        command = MotorCommand(
            label="load-force command",
            pursues_state=desired_state,
            generated_by=inverse_model,
        )
        efference_copy = EfferenceCopy(
            label="load-force command copy",
            copies_command=command,
        )
        estimate = make_estimate("expected object load")
        dynamics_model = ObjectDynamicsModel(
            label="object-load dynamics",
            holds_estimate=[estimate],
            coordinate_frame=CoordinateFrame.OBJECT_CENTERED,
            purpose=RepresentationPurpose.SENSORIMOTOR_CONTROL,
            represents_role=estimate.parameter_on_role,
            uses_efference_copy=[efference_copy],
        )
        forward_model = ForwardModel(label="load forward model")
        composition = InternalModelComposition(
            label="predictive load control",
            desired_state=desired_state,
            inverse_model=inverse_model,
            motor_command=command,
            efference_copy=efference_copy,
            dynamics_model=dynamics_model,
            forward_model=forward_model,
        )

        assert composition.motor_command.generated_by is composition.inverse_model
        assert composition.motor_command.pursues_state is composition.desired_state
        assert composition.efference_copy.copies_command is composition.motor_command
        assert (
            composition.efference_copy in composition.dynamics_model.uses_efference_copy
        )

    def test_incoherent_command_chain_is_rejected(self) -> None:
        desired_state = DesiredState(
            label="desired state",
            desired_condition=[Subgoal(label="desired condition")],
        )
        command = MotorCommand(
            label="command",
            pursues_state=desired_state,
            generated_by=InverseModel(label="unreferenced inverse model"),
        )
        efference_copy = EfferenceCopy(label="copy", copies_command=command)
        estimate = make_estimate("load")
        dynamics_model = ObjectDynamicsModel(
            label="object dynamics",
            holds_estimate=[estimate],
            coordinate_frame=CoordinateFrame.OBJECT_CENTERED,
            purpose=RepresentationPurpose.SENSORIMOTOR_CONTROL,
            represents_role=estimate.parameter_on_role,
            uses_efference_copy=[efference_copy],
        )
        composition = InternalModelComposition(
            label="broken composition",
            desired_state=desired_state,
            inverse_model=InverseModel(label="declared inverse model"),
            motor_command=command,
            efference_copy=efference_copy,
            dynamics_model=dynamics_model,
            forward_model=ForwardModel(label="forward model"),
        )

        violations = SUITE.evaluate((composition,))

        assert [
            violation.check for violation in violations if violation.check == "C51"
        ] == ["C51"]


# %% representation context


class TestRepresentationContext:
    """
    Representations state what they model and why they are used.
    """

    def test_coordinate_frames_cover_object_limb_hand_and_gaze_contexts(self) -> None:
        assert tuple(CoordinateFrame) == (
            CoordinateFrame.OBJECT_CENTERED,
            CoordinateFrame.ARM_CENTERED,
            CoordinateFrame.HAND_CENTERED,
            CoordinateFrame.GAZE_CENTERED,
        )

    def test_purposes_separate_control_from_judgment(self) -> None:
        assert tuple(RepresentationPurpose) == (
            RepresentationPurpose.SENSORIMOTOR_CONTROL,
            RepresentationPurpose.PERCEPTUAL_JUDGMENT,
        )

    def test_object_and_limb_dynamics_are_distinct_types(self) -> None:
        assert not issubclass(ObjectDynamicsModel, LimbDynamicsModel)
        assert not issubclass(LimbDynamicsModel, ObjectDynamicsModel)

    def test_object_dynamics_requires_an_efference_copy(self) -> None:
        estimate = make_estimate("object load")
        model = ObjectDynamicsModel(
            label="incomplete object dynamics",
            holds_estimate=[estimate],
            coordinate_frame=CoordinateFrame.OBJECT_CENTERED,
            purpose=RepresentationPurpose.SENSORIMOTOR_CONTROL,
            represents_role=estimate.parameter_on_role,
        )

        violations = SUITE.evaluate((model,))

        assert any(
            violation.check == "REQ" and "uses_efference_copy" in violation.message
            for violation in violations
        )


# %% representation ownership


class TestRepresentationOwnership:
    """
    Every estimate has one explicit representational context.
    """

    def test_unowned_estimate_is_rejected(self) -> None:
        estimate = make_estimate("unowned estimate")
        structure = ObjectDynamicsPhaseStructure(
            label="incomplete dynamics structure",
            defines_parameter=[estimate],
        )

        violations = SUITE.evaluate((structure, estimate))

        assert [
            violation.check for violation in violations if violation.check == "C52"
        ] == ["C52"]

    def test_one_estimate_cannot_cross_control_and_perceptual_purposes(self) -> None:
        estimate = make_estimate("shared estimate")
        object_model = make_object_model(estimate)
        perceptual = PerceptualRepresentation(
            label="perceptual judgment",
            holds_estimate=[estimate],
            coordinate_frame=CoordinateFrame.OBJECT_CENTERED,
            purpose=RepresentationPurpose.PERCEPTUAL_JUDGMENT,
            represents_role=estimate.parameter_on_role,
        )
        structure = ObjectDynamicsPhaseStructure(
            label="silently shared estimate",
            defines_parameter=[estimate],
            defines_internal_model=[object_model, perceptual],
        )

        violations = SUITE.evaluate((structure, estimate, object_model, perceptual))

        assert [
            violation.check for violation in violations if violation.check == "C53"
        ] == ["C53"]


# %% memory and representation updates


class TestSensorimotorMemory:
    """
    Object-specific representations declare their recruitment lifetime.
    """

    def test_object_memory_can_be_scoped_to_the_coupled_object(self) -> None:
        estimate = make_estimate("object load")
        model = make_object_model(estimate)
        memory = SensorimotorMemory(
            label="grasp-scoped object memory",
            stores_representation=[model],
            persistence=MemoryPersistence.WHILE_OBJECT_COUPLED,
        )

        assert memory.stores_representation == [model]
        assert memory.persistence is MemoryPersistence.WHILE_OBJECT_COUPLED

    def test_representation_update_names_memory_and_changed_estimate(self) -> None:
        estimate = make_estimate("adapted object load")
        model = make_object_model(estimate)
        memory = SensorimotorMemory(
            label="object memory",
            stores_representation=[model],
            persistence=MemoryPersistence.WHILE_OBJECT_COUPLED,
        )
        update = RepresentationUpdate(
            label="object-dynamics adaptation",
            updates_representation=model,
            updates_estimate=[estimate],
            direction=EstimateUpdateDirection.REVISE,
            persists_in=memory,
        )

        assert update.updates_representation is model
        assert update.updates_estimate == [estimate]
        assert update.persists_in is memory


# %% perceptual dissociation


class TestRepresentationDissociation:
    """
    Perceptual bias does not invalidate accurate motor scaling.
    """

    def test_dissociation_uses_separate_estimates_and_purposes(self) -> None:
        sensorimotor_estimate = make_estimate("sensorimotor weight estimate")
        perceptual_estimate = make_estimate("perceptual weight estimate")
        sensorimotor_model = make_object_model(sensorimotor_estimate)
        perceptual_model = PerceptualRepresentation(
            label="perceptual weight representation",
            holds_estimate=[perceptual_estimate],
            coordinate_frame=CoordinateFrame.OBJECT_CENTERED,
            purpose=RepresentationPurpose.PERCEPTUAL_JUDGMENT,
            represents_role=perceptual_estimate.parameter_on_role,
        )
        dissociation = RepresentationDissociation(
            label="size-weight dissociation",
            sensorimotor_representation=sensorimotor_model,
            perceptual_representation=perceptual_model,
            sensorimotor_estimate=sensorimotor_estimate,
            perceptual_estimate=perceptual_estimate,
            sensorimotor_accuracy=RepresentationAccuracy.ACCURATE,
            perceptual_accuracy=RepresentationAccuracy.BIASED,
        )

        violations = SUITE.evaluate(
            (
                sensorimotor_estimate,
                perceptual_estimate,
                sensorimotor_model,
                perceptual_model,
                dissociation,
            )
        )

        assert (
            dissociation.sensorimotor_estimate is not dissociation.perceptual_estimate
        )
        assert not [violation for violation in violations if violation.check == "C54"]


# %% dynamics contradictions


class TestDynamicsContradictions:
    """
    Malformed dynamics contexts are rejected by focused constraints.
    """

    def test_nonreciprocal_transfer_context_is_rejected(self) -> None:
        estimate = make_estimate("object load")
        model = make_object_model(estimate)
        source = DynamicsPerturbation(
            label="source load",
            kind=PerturbationKind.LOAD,
            affects_model=model,
            coordinate_frame=CoordinateFrame.OBJECT_CENTERED,
        )
        target = DynamicsPerturbation(
            label="target load",
            kind=PerturbationKind.LOAD,
            affects_model=model,
            coordinate_frame=CoordinateFrame.OBJECT_CENTERED,
        )
        transfer = DynamicsTransfer(
            label="unrelated transfer",
            from_perturbation=source,
            to_perturbation=target,
            outcome=TransferOutcome.DIFFERS,
        )

        violations = SUITE.evaluate((source, target, transfer))

        assert [
            violation.check for violation in violations if violation.check == "C56"
        ] == ["C56"]

    def test_object_dynamics_cannot_use_a_perceptual_purpose(self) -> None:
        model = make_object_model(
            make_estimate("object load"),
            purpose=RepresentationPurpose.PERCEPTUAL_JUDGMENT,
        )

        violations = SUITE.evaluate((model,))

        assert [
            violation.check for violation in violations if violation.check == "C57"
        ] == ["C57"]

    def test_representation_must_cover_each_estimated_role(self) -> None:
        estimate = make_estimate("object load")
        model = make_object_model(estimate)
        model.represents_role = [make_role()]

        violations = SUITE.evaluate((model, estimate))

        assert [
            violation.check for violation in violations if violation.check == "C58"
        ] == ["C58"]

    def test_representation_estimate_must_be_declared_by_the_root(self) -> None:
        estimate = make_estimate("undeclared represented estimate")
        model = make_object_model(estimate)
        structure = ObjectDynamicsPhaseStructure(
            label="missing estimate declaration",
            defines_internal_model=[model],
        )

        violations = SUITE.evaluate((structure, model, estimate))

        assert [
            violation.check for violation in violations if violation.check == "C13r"
        ] == ["C13r"]

    def test_update_memory_must_store_the_updated_representation(self) -> None:
        estimate = make_estimate("updated load")
        model = make_object_model(estimate)
        other_model = make_object_model(make_estimate("other load"))
        memory = SensorimotorMemory(
            label="unrelated memory",
            stores_representation=[other_model],
            persistence=MemoryPersistence.WHILE_OBJECT_COUPLED,
        )
        update = RepresentationUpdate(
            label="misplaced update",
            updates_representation=model,
            updates_estimate=[estimate],
            direction=EstimateUpdateDirection.REVISE,
            persists_in=memory,
        )

        violations = SUITE.evaluate((model, other_model, memory, update))

        assert [
            violation.check for violation in violations if violation.check == "C55"
        ] == ["C55"]
