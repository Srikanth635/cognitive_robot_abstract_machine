-- The pour of the first plan becomes a pick-up, which refers to how the object is grasped
-- (the tables are those of the real database).
CREATE TABLE "GraspDescriptionDAO" (
	database_id INTEGER NOT NULL,
	rotate_gripper BOOLEAN NOT NULL,
	manipulation_offset FLOAT NOT NULL,
	approach_direction VARCHAR(512) NOT NULL,
	vertical_alignment VARCHAR(512) NOT NULL,
	polymorphic_type VARCHAR(255) NOT NULL,
	_end_effector_id INTEGER,
	PRIMARY KEY (database_id),
	FOREIGN KEY(_end_effector_id) REFERENCES "EndEffectorDAO" (database_id)
);
CREATE TABLE "PickUpActionDAO" (
	database_id INTEGER NOT NULL,
	_grasp_description_id INTEGER,
	PRIMARY KEY (database_id),
	FOREIGN KEY(database_id) REFERENCES "ActionDescriptionDAO" (database_id),
	FOREIGN KEY(_grasp_description_id) REFERENCES "GraspDescriptionDAO" (database_id)
);
INSERT INTO "GraspDescriptionDAO" VALUES (1, 0, 0.1, 'coraplex.datastructures.enums.ApproachDirection.FRONT', 'coraplex.datastructures.enums.VerticalAlignment.NoAlignment', 'GraspDescriptionDAO', NULL);
INSERT INTO "PickUpActionDAO" VALUES (11, 1);
UPDATE "DesignatorDAO" SET polymorphic_type = 'PickUpActionDAO' WHERE database_id = 11;
