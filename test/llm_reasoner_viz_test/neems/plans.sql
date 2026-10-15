-- Two plans as the ORM of coraplex writes them, small enough to read: the first the plan of a
-- pour that missed, the second that of a run performed. Every step but the root is left as
-- not started, as in a real recording.

-- the names and the world entities the actions refer to
INSERT INTO "PrefixedNameDAO" VALUES (50, 'PouringCup', NULL, 'PrefixedNameDAO');
INSERT INTO "PrefixedNameDAO" VALUES (51, 'bowl.stl', NULL, 'PrefixedNameDAO');

INSERT INTO "WorldEntityDAO" VALUES (100, 'PouringCupDAO', 50);
INSERT INTO "WorldEntityWithIDDAO" VALUES (100, 'a1d3dd4558f44701ad653e31e327fad4');
INSERT INTO "WorldEntityWithSimulatorPropertiesDAO" VALUES (100);
INSERT INTO "SemanticAnnotationDAO" VALUES (100);
INSERT INTO "HasRootKinematicStructureEntityDAO" VALUES (100, NULL);
INSERT INTO "HasRootBodyDAO" VALUES (100);
INSERT INTO "ToolDAO" VALUES (100);
INSERT INTO "PouringCupDAO" VALUES (100);

INSERT INTO "WorldEntityDAO" VALUES (101, 'BodyDAO', 51);
INSERT INTO "WorldEntityWithIDDAO" VALUES (101, '4f7224f16dcb4546b33efc6b2d225877');
INSERT INTO "WorldEntityWithSimulatorPropertiesDAO" VALUES (101);
INSERT INTO "KinematicStructureEntityDAO" VALUES (101);
INSERT INTO "BodyDAO" VALUES (101, NULL, NULL, NULL);

INSERT INTO "PrefixedNameDAO" VALUES (52, 'map', NULL, 'PrefixedNameDAO');
INSERT INTO "WorldEntityDAO" VALUES (102, 'BodyDAO', 52);
INSERT INTO "WorldEntityWithIDDAO" VALUES (102, 'b2c4d6e8f0a14b3c8d5e7f9a1b3c5d7e');
INSERT INTO "WorldEntityWithSimulatorPropertiesDAO" VALUES (102);
INSERT INTO "KinematicStructureEntityDAO" VALUES (102);
INSERT INTO "BodyDAO" VALUES (102, NULL, NULL, NULL);

-- a pose without an orientation, and one with
INSERT INTO "SpatialTypeDAO" VALUES (300, 'PoseMappingDAO', 102);
INSERT INTO "PoseMappingDAO" VALUES (300, 301, NULL);
INSERT INTO "Point3MappingDAO" VALUES (301, 1.5, 2.25, 0.0);
INSERT INTO "SpatialTypeDAO" VALUES (310, 'PoseMappingDAO', 102);
INSERT INTO "PoseMappingDAO" VALUES (310, 311, 312);
INSERT INTO "Point3MappingDAO" VALUES (311, 0.5, -1.0, 0.0);
INSERT INTO "QuaternionMappingDAO" VALUES (312, 0.0, 0.0, 0.0, 1.0);

-- the designators of the steps
INSERT INTO "DesignatorDAO" VALUES (10, 'ParkArmsActionDAO');
INSERT INTO "ActionDescriptionDAO" VALUES (10);
INSERT INTO "ParkArmsActionDAO" VALUES (10, NULL, 'coraplex.datastructures.enums.Arms.BOTH');

INSERT INTO "DesignatorDAO" VALUES (11, 'PouringActionDAO');
INSERT INTO "ActionDescriptionDAO" VALUES (11);
INSERT INTO "FullBodyControlledActionDAO" VALUES (11);
INSERT INTO "PouringActionDAO" VALUES (11, NULL, NULL, 1.85, 0.0, 0.0, 0.13, 'coraplex.datastructures.enums.Arms.RIGHT', 'coraplex.datastructures.enums.Arms.RIGHT', 101, 100);

INSERT INTO "DesignatorDAO" VALUES (20, 'MoveToolCenterPointMotionDAO');
INSERT INTO "BaseMotionDAO" VALUES (20);
INSERT INTO "MoveToolCenterPointMotionDAO" VALUES (20, NULL, NULL, NULL, NULL, 1, 'coraplex.datastructures.enums.Arms.RIGHT', NULL, 300);

INSERT INTO "DesignatorDAO" VALUES (13, 'ParkArmsActionDAO');
INSERT INTO "ActionDescriptionDAO" VALUES (13);
INSERT INTO "ParkArmsActionDAO" VALUES (13, NULL, 'coraplex.datastructures.enums.Arms.BOTH');

INSERT INTO "DesignatorDAO" VALUES (14, 'PouringActionDAO');
INSERT INTO "ActionDescriptionDAO" VALUES (14);
INSERT INTO "FullBodyControlledActionDAO" VALUES (14);
INSERT INTO "PouringActionDAO" VALUES (14, NULL, NULL, 1.6, -0.188, 0.0, 0.13, 'coraplex.datastructures.enums.Arms.RIGHT', 'coraplex.datastructures.enums.Arms.RIGHT', 101, 100);

INSERT INTO "DesignatorDAO" VALUES (15, 'NavigateActionDAO');
INSERT INTO "ActionDescriptionDAO" VALUES (15);
INSERT INTO "NavigateActionDAO" VALUES (15, 1, 310);

-- plan 1: a sequence of two steps, the second with a condition and a motion below it
INSERT INTO "PlanEntityDAO" VALUES (1, 'SequentialNodeDAO', 1);
INSERT INTO "PlanNodeDAO" VALUES (1, '2026-10-14 16:48:50.000000', '2026-10-14 16:48:52.500000', 'giskardpy.motion_statechart.data_types.LifeCycleValues.SUCCEEDED', NULL);
INSERT INTO "LanguageNodeDAO" VALUES (1, 'giskardpy.motion_statechart.goals.templates.Sequence');
INSERT INTO "ExecutesSequentiallyDAO" VALUES (1);
INSERT INTO "SequentialNodeDAO" VALUES (1);

INSERT INTO "PlanEntityDAO" VALUES (2, 'ActionNodeDAO', 1);
INSERT INTO "PlanNodeDAO" VALUES (2, '2026-10-14 16:48:50.000000', NULL, 'giskardpy.motion_statechart.data_types.LifeCycleValues.NOT_STARTED', NULL);
INSERT INTO "DesignatorNodeDAO" VALUES (2, 10);
INSERT INTO "ActionNodeDAO" VALUES (2, NULL);

INSERT INTO "PlanEntityDAO" VALUES (3, 'ActionNodeDAO', 1);
INSERT INTO "PlanNodeDAO" VALUES (3, '2026-10-14 16:48:50.000000', NULL, 'giskardpy.motion_statechart.data_types.LifeCycleValues.NOT_STARTED', NULL);
INSERT INTO "DesignatorNodeDAO" VALUES (3, 11);
INSERT INTO "ActionNodeDAO" VALUES (3, NULL);

INSERT INTO "PlanEntityDAO" VALUES (4, 'ConditionNodeDAO', 1);
INSERT INTO "PlanNodeDAO" VALUES (4, '2026-10-14 16:48:50.000000', NULL, 'giskardpy.motion_statechart.data_types.LifeCycleValues.NOT_STARTED', NULL);
INSERT INTO "ConditionNodeDAO" VALUES (4, 1, 3);

INSERT INTO "PlanEntityDAO" VALUES (5, 'MotionNodeDAO', 1);
INSERT INTO "PlanNodeDAO" VALUES (5, '2026-10-14 16:48:50.000000', NULL, 'giskardpy.motion_statechart.data_types.LifeCycleValues.NOT_STARTED', NULL);
INSERT INTO "DesignatorNodeDAO" VALUES (5, 20);
INSERT INTO "MotionNodeDAO" VALUES (5);

INSERT INTO "PlanEdgeDAO" VALUES (1, 'PlanEdgeDAO', 1, 2);
INSERT INTO "PlanEdgeDAO" VALUES (2, 'PlanEdgeDAO', 1, 3);
INSERT INTO "PlanEdgeDAO" VALUES (3, 'PlanEdgeDAO', 3, 4);
INSERT INTO "PlanEdgeDAO" VALUES (4, 'PlanEdgeDAO', 3, 5);

INSERT INTO "PlanMappingDAO" VALUES (1, 'PlanMappingDAO', 1, NULL, NULL);

-- plan 2: park the arms, navigate, pour
INSERT INTO "PlanEntityDAO" VALUES (11, 'SequentialNodeDAO', 2);
INSERT INTO "PlanNodeDAO" VALUES (11, '2026-10-14 17:02:10.000000', '2026-10-14 17:02:30.250000', 'giskardpy.motion_statechart.data_types.LifeCycleValues.SUCCEEDED', NULL);
INSERT INTO "LanguageNodeDAO" VALUES (11, 'giskardpy.motion_statechart.goals.templates.Sequence');
INSERT INTO "ExecutesSequentiallyDAO" VALUES (11);
INSERT INTO "SequentialNodeDAO" VALUES (11);

INSERT INTO "PlanEntityDAO" VALUES (12, 'ActionNodeDAO', 2);
INSERT INTO "PlanNodeDAO" VALUES (12, '2026-10-14 17:02:10.000000', NULL, 'giskardpy.motion_statechart.data_types.LifeCycleValues.NOT_STARTED', NULL);
INSERT INTO "DesignatorNodeDAO" VALUES (12, 13);
INSERT INTO "ActionNodeDAO" VALUES (12, NULL);

INSERT INTO "PlanEntityDAO" VALUES (13, 'ActionNodeDAO', 2);
INSERT INTO "PlanNodeDAO" VALUES (13, '2026-10-14 17:02:10.000000', NULL, 'giskardpy.motion_statechart.data_types.LifeCycleValues.NOT_STARTED', NULL);
INSERT INTO "DesignatorNodeDAO" VALUES (13, 15);
INSERT INTO "ActionNodeDAO" VALUES (13, NULL);

INSERT INTO "PlanEntityDAO" VALUES (14, 'ActionNodeDAO', 2);
INSERT INTO "PlanNodeDAO" VALUES (14, '2026-10-14 17:02:10.000000', NULL, 'giskardpy.motion_statechart.data_types.LifeCycleValues.NOT_STARTED', NULL);
INSERT INTO "DesignatorNodeDAO" VALUES (14, 14);
INSERT INTO "ActionNodeDAO" VALUES (14, NULL);

INSERT INTO "PlanEdgeDAO" VALUES (11, 'PlanEdgeDAO', 11, 12);
INSERT INTO "PlanEdgeDAO" VALUES (12, 'PlanEdgeDAO', 11, 13);
INSERT INTO "PlanEdgeDAO" VALUES (13, 'PlanEdgeDAO', 11, 14);

INSERT INTO "PlanMappingDAO" VALUES (2, 'PlanMappingDAO', 11, NULL, NULL);
