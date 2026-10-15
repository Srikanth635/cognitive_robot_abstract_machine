-- The pour of the first plan becomes an action whose table has a column of bytes.
CREATE TABLE "ScanActionDAO" (
	database_id INTEGER NOT NULL,
	payload BLOB,
	resolution FLOAT,
	PRIMARY KEY (database_id),
	FOREIGN KEY(database_id) REFERENCES "ActionDescriptionDAO" (database_id)
);
INSERT INTO "ScanActionDAO" VALUES (11, x'00ff', 2.5);
UPDATE "DesignatorDAO" SET polymorphic_type = 'ScanActionDAO' WHERE database_id = 11;
