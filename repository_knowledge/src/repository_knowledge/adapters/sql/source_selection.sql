SELECT payload FROM sources
WHERE relative_path IN (SELECT value FROM json_each(:paths))
ORDER BY relative_path;
