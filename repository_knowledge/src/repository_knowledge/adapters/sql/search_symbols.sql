SELECT payload FROM symbols
WHERE instr(casefold(symbol_identifier), casefold(:query)) > 0
   OR instr(casefold(json_extract(payload, '$.documentation')), casefold(:query)) > 0
ORDER BY symbol_identifier, start_line;
