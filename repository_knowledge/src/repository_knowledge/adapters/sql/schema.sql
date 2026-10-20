CREATE TABLE IF NOT EXISTS metadata (payload TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS analyses (
    relative_path TEXT PRIMARY KEY,
    payload BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS sources (
    relative_path TEXT PRIMARY KEY,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS symbols (
    symbol_identifier TEXT NOT NULL,
    relative_path TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY (symbol_identifier, relative_path, start_line)
);
CREATE TABLE IF NOT EXISTS relationships (
    ordinal INTEGER PRIMARY KEY,
    source_identifier TEXT NOT NULL,
    target_identifier TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS diagnostics (
    ordinal INTEGER PRIMARY KEY,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS incoming_references ON relationships(target_identifier);
CREATE INDEX IF NOT EXISTS outgoing_references ON relationships(source_identifier);

