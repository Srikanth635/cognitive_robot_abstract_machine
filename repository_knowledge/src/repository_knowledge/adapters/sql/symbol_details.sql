SELECT payload FROM symbols
WHERE symbol_identifier = :symbol_identifier
ORDER BY symbol_identifier, start_line;
