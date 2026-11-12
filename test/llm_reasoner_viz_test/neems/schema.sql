-- The tables of a NEEM database the viewer reads, as the ORM of cramph generates them; the
-- foreign keys to tables the viewer does not read are left out.

CREATE TABLE "StatechartContextDAO" (
	database_id INTEGER NOT NULL,
	tick_duration FLOAT,
	polymorphic_type VARCHAR(255) NOT NULL,
	_world_id INTEGER,
	PRIMARY KEY (database_id)
);

CREATE TABLE "StatechartMappingDAO" (
	database_id INTEGER NOT NULL,
	pass_limit INTEGER NOT NULL,
	passes_per_nesting_level INTEGER NOT NULL,
	structure TEXT NOT NULL,
	polymorphic_type VARCHAR(255) NOT NULL,
	_context_id INTEGER,
	PRIMARY KEY (database_id),
	FOREIGN KEY(_context_id) REFERENCES "StatechartContextDAO" (database_id)
);

CREATE TABLE "NodeStatesAtTickDAO" (
	database_id INTEGER NOT NULL,
	tick_count INTEGER NOT NULL,
	cancelling_node_index INTEGER,
	life_cycle_values JSON NOT NULL,
	observation_values JSON NOT NULL,
	polymorphic_type VARCHAR(255) NOT NULL,
	_cancellation_error_id INTEGER,
	PRIMARY KEY (database_id)
);

CREATE TABLE _41149830878392270947832154666713041506341253427131999726083201 (
	database_id INTEGER NOT NULL,
	_source_statechartmappingdao_id INTEGER NOT NULL,
	_target_nodestatesattickdao_id INTEGER NOT NULL,
	PRIMARY KEY (database_id),
	FOREIGN KEY(_source_statechartmappingdao_id) REFERENCES "StatechartMappingDAO" (database_id),
	FOREIGN KEY(_target_nodestatesattickdao_id) REFERENCES "NodeStatesAtTickDAO" (database_id)
);

CREATE TABLE "WorldEntityDAO" (
	database_id INTEGER NOT NULL,
	polymorphic_type VARCHAR(255) NOT NULL,
	_name_id INTEGER,
	PRIMARY KEY (database_id)
);

CREATE TABLE "WorldEntityWithIDDAO" (
	database_id INTEGER NOT NULL,
	id UUID NOT NULL,
	PRIMARY KEY (database_id),
	FOREIGN KEY(database_id) REFERENCES "WorldEntityDAO" (database_id)
);
