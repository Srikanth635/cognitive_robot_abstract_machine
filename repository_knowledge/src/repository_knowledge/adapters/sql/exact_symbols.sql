SELECT payload FROM symbols
WHERE casefold(symbol_identifier) = casefold(:query)
   OR casefold(json_extract(payload, '$.symbol_name')) = casefold(:query)
ORDER BY symbol_identifier, start_line;
