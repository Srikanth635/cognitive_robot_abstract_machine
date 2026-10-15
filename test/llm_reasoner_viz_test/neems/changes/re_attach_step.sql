-- A step is added to the first plan that moves a body to another parent, as the plan of a
-- pick-up does (the tables are those of the real database).
CREATE TABLE "ExecutionBoundaryNodeDAO" (
	database_id INTEGER NOT NULL,
	PRIMARY KEY (database_id),
	FOREIGN KEY(database_id) REFERENCES "PlanNodeDAO" (database_id)
);
CREATE TABLE "ReAttachNodeDAO" (
	database_id INTEGER NOT NULL,
	_body_id INTEGER,
	_new_parent_id INTEGER,
	PRIMARY KEY (database_id),
	FOREIGN KEY(database_id) REFERENCES "ExecutionBoundaryNodeDAO" (database_id),
	FOREIGN KEY(_body_id) REFERENCES "KinematicStructureEntityDAO" (database_id),
	FOREIGN KEY(_new_parent_id) REFERENCES "KinematicStructureEntityDAO" (database_id)
);
INSERT INTO "PlanEntityDAO" VALUES (6, 'ReAttachNodeDAO', 1);
INSERT INTO "PlanNodeDAO" VALUES (6, '2026-10-14 16:48:50.000000', NULL, 'giskardpy.motion_statechart.data_types.LifeCycleValues.NOT_STARTED', NULL);
INSERT INTO "ExecutionBoundaryNodeDAO" VALUES (6);
INSERT INTO "ReAttachNodeDAO" VALUES (6, 101, 102);
INSERT INTO "PlanEdgeDAO" VALUES (5, 'PlanEdgeDAO', 1, 6);
