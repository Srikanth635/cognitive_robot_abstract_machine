SELECT ordinal, payload FROM relationships
WHERE target_identifier IN (SELECT value FROM json_each(:identifiers))
ORDER BY ordinal;
