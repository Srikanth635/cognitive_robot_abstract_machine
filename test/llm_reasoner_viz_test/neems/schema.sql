-- The tables of the NEEM database the viewer reads, as the ORM of coraplex generates them (the
-- columns and the foreign keys that say which table extends which), taken from a real database.

CREATE TABLE "PlanMappingDAO" (
	database_id INTEGER NOT NULL, 
	polymorphic_type VARCHAR(255) NOT NULL, 
	_root_id INTEGER, 
	_context_id INTEGER, 
	_initial_world_id INTEGER, 
	PRIMARY KEY (database_id), 
	FOREIGN KEY(_root_id) REFERENCES "PlanNodeDAO" (database_id), 
	FOREIGN KEY(_context_id) REFERENCES "ContextDAO" (database_id), 
	FOREIGN KEY(_initial_world_id) REFERENCES "WorldMappingDAO" (database_id)
);

CREATE TABLE "PlanNodeDAO" (
	database_id INTEGER NOT NULL, 
	start_time DATETIME, 
	end_time DATETIME, 
	status VARCHAR(512) NOT NULL, 
	_reason_id INTEGER, 
	PRIMARY KEY (database_id), 
	FOREIGN KEY(database_id) REFERENCES "PlanEntityDAO" (database_id), 
	FOREIGN KEY(_reason_id) REFERENCES "PlanFailureDAO" (database_id)
);

CREATE TABLE "PlanEdgeDAO" (
	database_id INTEGER NOT NULL, 
	polymorphic_type VARCHAR(255) NOT NULL, 
	_parent_id INTEGER, 
	_child_id INTEGER, 
	PRIMARY KEY (database_id), 
	FOREIGN KEY(_parent_id) REFERENCES "PlanNodeDAO" (database_id), 
	FOREIGN KEY(_child_id) REFERENCES "PlanNodeDAO" (database_id)
);

CREATE TABLE "PlanEntityDAO" (
	database_id INTEGER NOT NULL, 
	polymorphic_type VARCHAR(255) NOT NULL, 
	_plan_id INTEGER, 
	PRIMARY KEY (database_id), 
	FOREIGN KEY(_plan_id) REFERENCES "PlanMappingDAO" (database_id)
);

CREATE TABLE "LanguageNodeDAO" (
	database_id INTEGER NOT NULL, 
	motion_state_chart_template VARCHAR(256) NOT NULL, 
	PRIMARY KEY (database_id), 
	FOREIGN KEY(database_id) REFERENCES "PlanNodeDAO" (database_id)
);

CREATE TABLE "ExecutesSequentiallyDAO" (
	database_id INTEGER NOT NULL, 
	PRIMARY KEY (database_id), 
	FOREIGN KEY(database_id) REFERENCES "LanguageNodeDAO" (database_id)
);

CREATE TABLE "SequentialNodeDAO" (
	database_id INTEGER NOT NULL, 
	PRIMARY KEY (database_id), 
	FOREIGN KEY(database_id) REFERENCES "ExecutesSequentiallyDAO" (database_id)
);

CREATE TABLE "DesignatorNodeDAO" (
	database_id INTEGER NOT NULL, 
	_designator_id INTEGER, 
	PRIMARY KEY (database_id), 
	FOREIGN KEY(database_id) REFERENCES "PlanNodeDAO" (database_id), 
	FOREIGN KEY(_designator_id) REFERENCES "DesignatorDAO" (database_id)
);

CREATE TABLE "ActionNodeDAO" (
	database_id INTEGER NOT NULL, 
	_execution_data_id INTEGER, 
	PRIMARY KEY (database_id), 
	FOREIGN KEY(database_id) REFERENCES "DesignatorNodeDAO" (database_id), 
	FOREIGN KEY(_execution_data_id) REFERENCES "ExecutionDataDAO" (database_id)
);

CREATE TABLE "MotionNodeDAO" (
	database_id INTEGER NOT NULL, 
	PRIMARY KEY (database_id), 
	FOREIGN KEY(database_id) REFERENCES "DesignatorNodeDAO" (database_id)
);

CREATE TABLE "ConditionNodeDAO" (
	database_id INTEGER NOT NULL, 
	pre_condition BOOLEAN NOT NULL, 
	_action_node_id INTEGER, 
	PRIMARY KEY (database_id), 
	FOREIGN KEY(database_id) REFERENCES "PlanNodeDAO" (database_id), 
	FOREIGN KEY(_action_node_id) REFERENCES "ActionNodeDAO" (database_id)
);

CREATE TABLE "DesignatorDAO" (
	database_id INTEGER NOT NULL, 
	polymorphic_type VARCHAR(255) NOT NULL, 
	PRIMARY KEY (database_id)
);

CREATE TABLE "ActionDescriptionDAO" (
	database_id INTEGER NOT NULL, 
	PRIMARY KEY (database_id), 
	FOREIGN KEY(database_id) REFERENCES "DesignatorDAO" (database_id)
);

CREATE TABLE "FullBodyControlledActionDAO" (
	database_id INTEGER NOT NULL, 
	PRIMARY KEY (database_id), 
	FOREIGN KEY(database_id) REFERENCES "ActionDescriptionDAO" (database_id)
);

CREATE TABLE "PouringActionDAO" (
	database_id INTEGER NOT NULL, 
	position_threshold FLOAT, 
	orientation_threshold FLOAT, 
	tilt_angle FLOAT NOT NULL, 
	pour_side_offset FLOAT NOT NULL, 
	pour_approach_offset FLOAT NOT NULL, 
	pour_height FLOAT NOT NULL, 
	arm VARCHAR(512) NOT NULL, 
	pour_side VARCHAR(512), 
	_target_container_id INTEGER, 
	_source_container_id INTEGER, 
	PRIMARY KEY (database_id), 
	FOREIGN KEY(database_id) REFERENCES "FullBodyControlledActionDAO" (database_id), 
	FOREIGN KEY(_target_container_id) REFERENCES "BodyDAO" (database_id), 
	FOREIGN KEY(_source_container_id) REFERENCES "ToolDAO" (database_id)
);

CREATE TABLE "ParkArmsActionDAO" (
	database_id INTEGER NOT NULL, 
	max_joint_velocity FLOAT, 
	arm VARCHAR(512) NOT NULL, 
	PRIMARY KEY (database_id), 
	FOREIGN KEY(database_id) REFERENCES "ActionDescriptionDAO" (database_id)
);

CREATE TABLE "NavigateActionDAO" (
	database_id INTEGER NOT NULL, 
	keep_joint_states BOOLEAN NOT NULL, 
	_target_location_id INTEGER, 
	PRIMARY KEY (database_id), 
	FOREIGN KEY(database_id) REFERENCES "ActionDescriptionDAO" (database_id), 
	FOREIGN KEY(_target_location_id) REFERENCES "PoseMappingDAO" (database_id)
);

CREATE TABLE "BaseMotionDAO" (
	database_id INTEGER NOT NULL, 
	PRIMARY KEY (database_id), 
	FOREIGN KEY(database_id) REFERENCES "DesignatorDAO" (database_id)
);

CREATE TABLE "MoveToolCenterPointMotionDAO" (
	database_id INTEGER NOT NULL, 
	position_threshold FLOAT, 
	orientation_threshold FLOAT, 
	max_linear_velocity FLOAT, 
	max_angular_velocity FLOAT, 
	allow_gripper_collision BOOLEAN, 
	arm VARCHAR(512) NOT NULL, 
	movement_type VARCHAR(512), 
	_target_id INTEGER, 
	PRIMARY KEY (database_id), 
	FOREIGN KEY(database_id) REFERENCES "BaseMotionDAO" (database_id), 
	FOREIGN KEY(_target_id) REFERENCES "PoseMappingDAO" (database_id)
);

CREATE TABLE "ToolDAO" (
	database_id INTEGER NOT NULL, 
	PRIMARY KEY (database_id), 
	FOREIGN KEY(database_id) REFERENCES "HasRootBodyDAO" (database_id)
);

CREATE TABLE "HasRootBodyDAO" (
	database_id INTEGER NOT NULL, 
	PRIMARY KEY (database_id), 
	FOREIGN KEY(database_id) REFERENCES "HasRootKinematicStructureEntityDAO" (database_id)
);

CREATE TABLE "HasRootKinematicStructureEntityDAO" (
	database_id INTEGER NOT NULL, 
	_root_id INTEGER, 
	PRIMARY KEY (database_id), 
	FOREIGN KEY(database_id) REFERENCES "SemanticAnnotationDAO" (database_id), 
	FOREIGN KEY(_root_id) REFERENCES "KinematicStructureEntityDAO" (database_id)
);

CREATE TABLE "SemanticAnnotationDAO" (
	database_id INTEGER NOT NULL, 
	PRIMARY KEY (database_id), 
	FOREIGN KEY(database_id) REFERENCES "WorldEntityWithSimulatorPropertiesDAO" (database_id)
);

CREATE TABLE "WorldEntityWithSimulatorPropertiesDAO" (
	database_id INTEGER NOT NULL, 
	PRIMARY KEY (database_id), 
	FOREIGN KEY(database_id) REFERENCES "WorldEntityWithIDDAO" (database_id)
);

CREATE TABLE "WorldEntityWithIDDAO" (
	database_id INTEGER NOT NULL, 
	id UUID NOT NULL, 
	PRIMARY KEY (database_id), 
	FOREIGN KEY(database_id) REFERENCES "WorldEntityDAO" (database_id)
);

CREATE TABLE "WorldEntityDAO" (
	database_id INTEGER NOT NULL, 
	polymorphic_type VARCHAR(255) NOT NULL, 
	_name_id INTEGER, 
	PRIMARY KEY (database_id), 
	FOREIGN KEY(_name_id) REFERENCES "PrefixedNameDAO" (database_id)
);

CREATE TABLE "BodyDAO" (
	database_id INTEGER NOT NULL, 
	_visual_id INTEGER, 
	_collision_id INTEGER, 
	_inertial_id INTEGER, 
	PRIMARY KEY (database_id), 
	FOREIGN KEY(database_id) REFERENCES "KinematicStructureEntityDAO" (database_id), 
	FOREIGN KEY(_visual_id) REFERENCES "ShapeCollectionDAO" (database_id), 
	FOREIGN KEY(_collision_id) REFERENCES "ShapeCollectionDAO" (database_id), 
	FOREIGN KEY(_inertial_id) REFERENCES "InertialDAO" (database_id)
);

CREATE TABLE "KinematicStructureEntityDAO" (
	database_id INTEGER NOT NULL, 
	PRIMARY KEY (database_id), 
	FOREIGN KEY(database_id) REFERENCES "WorldEntityWithSimulatorPropertiesDAO" (database_id)
);

CREATE TABLE "PrefixedNameDAO" (
	database_id INTEGER NOT NULL, 
	name TEXT NOT NULL, 
	prefix TEXT, 
	polymorphic_type VARCHAR(255) NOT NULL, 
	PRIMARY KEY (database_id)
);

CREATE TABLE "PoseMappingDAO" (
	database_id INTEGER NOT NULL, 
	_position_id INTEGER, 
	_orientation_id INTEGER, 
	PRIMARY KEY (database_id), 
	FOREIGN KEY(database_id) REFERENCES "SpatialTypeDAO" (database_id), 
	FOREIGN KEY(_position_id) REFERENCES "Point3MappingDAO" (database_id), 
	FOREIGN KEY(_orientation_id) REFERENCES "QuaternionMappingDAO" (database_id)
);

CREATE TABLE "SpatialTypeDAO" (
	database_id INTEGER NOT NULL, 
	polymorphic_type VARCHAR(255) NOT NULL, 
	_reference_frame_id INTEGER, 
	PRIMARY KEY (database_id), 
	FOREIGN KEY(_reference_frame_id) REFERENCES "KinematicStructureEntityDAO" (database_id)
);

CREATE TABLE "Point3MappingDAO" (
	database_id INTEGER NOT NULL, 
	x FLOAT NOT NULL, 
	y FLOAT NOT NULL, 
	z FLOAT NOT NULL, 
	PRIMARY KEY (database_id), 
	FOREIGN KEY(database_id) REFERENCES "PointDAO" (database_id)
);
