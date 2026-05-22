# SDT — entity index

Complete entity listing for the `semantic_digital_twin` (SDT) package. See [[index]] for Concepts, Bridges, and per-package overview pages.

---

## World model

- [[sdt.world.World]] — entity — SDT world model: rustworkx `PyDAG` of kinematic entities with atomic modification and FK-preserving branch ops.
- [[sdt.world_description.world_entity.KinematicStructureEntity]] — entity — Abstract node base for the kinematic tree; provides FK, CoM, traversal, `_world` backref.
- [[sdt.world_description.world_entity.Body]] — entity — SDT semantic atom: rigid body with visual/collision ShapeCollections and inertial properties.
- [[sdt.world_description.world_entity.Region]] — entity — Named spatial volume with `area` ShapeCollection; used for perception bounding-box queries.
- [[sdt.world_description.world_entity.Connection]] — entity — Kinematic edge (parent→child) with three-transform structure: fixed + variable kinematics + fixed.
- [[sdt.world_description.connections]] — entity — All concrete Connection subclasses: FixedConnection, PrismaticConnection, RevoluteConnection, Connection6DoF, OmniDrive, DifferentialDrive.
- [[sdt.world_description.degree_of_freedom.DegreeOfFreedom]] — entity — Scalar joint coordinate with symbolic CasADi variables; state stored in `world.state`.
- [[sdt.world_description.shape_collection.ShapeCollection]] — entity — Ordered list of shapes anchored to a KSE; provides combined mesh and bounding-box queries.
- [[sdt.world_description.world_state.WorldState]] — entity — `4×N` numpy DOF state bus (pos/vel/acc/jerk); zero-copy memory view for FK compiled functions.

## Spatial computations

- [[sdt.spatial_computations.forward_kinematics]] — entity — `ForwardKinematicsManager`: BFS expression cache + batched CasADi FK compilation; zero-copy evaluation via WorldState memory view.
- [[sdt.spatial_computations.ik_solver]] — entity — `InverseKinematicsSolver`: QP-based IK using `daqp`; iterates velocity steps until convergence or unreachability.
- [[sdt.spatial_computations.raytracer]] — entity — `RayTracer`: trimesh-backed ray casting for segmentation masks and depth maps; lazy update via version counters.

## Robots

- [[sdt.robots.abstract_robot.AbstractRobot]] — entity — Abstract robot base: `SemanticRobotAnnotation` chains (arms, gripper, camera) discovered from World; replaces legacy `pycram.robot_description.RobotDescription`.
- [[sdt.robots.abstract_robot.Manipulator]] — entity — Bundled: `Manipulator` (abstract end-effector), `ParallelGripper`, `HumanoidGripper`, `Finger`; carries `tool_frame` + `front_facing_orientation` consumed by GraspDescription.
- [[sdt.robots.concrete]] — entity — Concrete robot models (20 total): Tiago, Panda, PR2, HSR-B, etc.; two-step world-body wiring pattern.

## Semantic layer

- [[sdt.semantic_annotations.SemanticAnnotations]] — entity — Concrete annotation library: Furniture, Handle, Drawer, Door, Container + mixin traits (HasRootBody, HasSupportingSurface, IsPerceivable, …).
- [[sdt.reasoning.WorldReasoner]] — entity — RDR-based semantic inference: WorldReasoner, CaseReasoner, EQL predicates (summary only; full coverage in `sdt.reasoning.predicates`), query helpers.
- [[sdt.reasoning.predicates]] — entity — Module of `@symbolic_function` predicates: `visible` (segmentation mask, ignores orientation), `contact` (FCL), `is_supported_by` (interval algebra), `reachable`, spatial relations (LeftOf/Above/etc.), `is_place_occupied`, `InsideOf`.
- [[sdt.reasoning.robot_predicates]] — entity — Robot-centric `@symbolic_function` predicates: `is_body_in_gripper` (ray-sampling between fingers), `bodies_in_gripper`, `robot_in_collision`, `robot_holds_body`, `blocking`, `is_pose_free_for_robot`.
- [[sdt.collision_checking]] — entity — `CollisionManager`: collision rule stack (allow/avoid), FCL/pybullet backends, `CollisionConsumer` observer pattern.

## Geometry

- [[sdt.world_description.geometry.BoundingBox]] — entity — Frame-aware AABB; `contains(point)` transforms to origin frame first; `random_events` interval algebra; used by PerceptionQuery and `sdt.reasoning.predicates`.

## Spatial types

- [[sdt.spatial_types.spatial_types]] — entity — Foundational spatial algebra module: `HomogeneousTransformationMatrix`, `RotationMatrix`, `Point3`, `Vector3`, `Quaternion`, `Pose`; all CasADi SX–backed with optional kinematic `reference_frame`.
- [[sdt.spatial_types.Pose]] — entity — Frame-anchored 4×4 HTM backed by CasADi SX; primary pose type throughout SDT and pycram motion designators.

## Datastructures

- [[sdt.datastructures.GripperState]] — entity — `JointStateType` enum: OPEN/CLOSE/MEDIUM; gripper command selector for motion designators.
- [[sdt.datastructures.joint_state.JointState]] — entity — Named joint target configuration (connection → float); `GripperState`/`ArmState`/`TorsoState` are type aliases.

## Adapters

- [[sdt.adapters]] — entity — Format-specific parsers that construct `World` objects from URDF, Xacro, MJCF, USD, FBX, and dataset files (PartNet-Mobility, SAGE-10K, ProcTHOR); includes ROS 2 TF/message adapters.

## Pipeline

- [[sdt.pipeline]] — entity — `Pipeline` + `Step` machinery for asset post-processing: body filtering, mesh-centre normalisation, semantic annotation injection, and convex mesh decomposition (VHACD/CoACD/BoxDecomposer).
