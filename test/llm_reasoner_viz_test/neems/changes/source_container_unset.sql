-- A pour was not given the container it pours from.
UPDATE "StatechartMappingDAO"
SET structure = json_set(structure, '$.nodes[3].source_container', json('null'))
WHERE database_id = 1;
