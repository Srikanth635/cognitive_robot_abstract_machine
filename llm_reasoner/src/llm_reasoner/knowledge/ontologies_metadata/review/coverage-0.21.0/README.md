# Current review index

Start with [REVIEW_REPORT.md](REVIEW_REPORT.md) for findings and acceptance criteria.

| Artifact | Scope |
|---|---|
| `original-objectives.csv` / `.json` | Eight objectives, current assessment and next acceptance step |
| `local-competency-coverage.csv` / `.json` | All 30 local questions; distinguishes historical V7 and later contracts |
| `repository-scientific-competencies.csv` / `.json` | All 65 distinct repository scientific questions; semantic review, not passed answer tests |
| `coraplex-action-coverage.csv` / `.json` | 28 static action and 13 motion candidates, mapping limitations and source links |
| `coraplex-static-inventory.json` | AST-extracted classes, locally abstract exclusions, fields, source locations and local call names |
| `module-consolidation.csv` / `.json` | Role and improvement recommendation for every registered ontology |
| `upstream-source-manifest.json` | Pinned commit and 31 inspected files; Git blob hash verification and SHA-256 |
| `ontology-preservation.json` | 87 prior RDF/OWL files unchanged against the cumulative 0.20.0 ZIP |
| `../../reports/current-coverage-review.json` | Current nested queries, record isolation and exact statement coverage checks |
| `../../reports/release-0.21.0.json` | Fresh versus inherited verification and limitations |

The CSV/JSON matrices are curated review decisions. The inventory extractor regenerates static source facts only; it does not automatically establish semantic mappings or execution eligibility. Re-evaluate the curated coverage if the upstream commit changes.

`tools/review_current_coverage.py` binds one existing record and its unique root, then queries a descriptive graph view. Claims and statement reifications remain in the separate original data graph for provenance checks. Queries retrieve declared structure; they do not consume evidence, execute contracts or promote facts into world beliefs.

Current capability retrieval is deliberately bounded to descendant actions and their direct/profile requirement owners and the eight reviewed requirement properties. Other aspect-specific owners still need their existing specialized queries. New requirement properties need a query update; the vocabulary check detects such a schema change.
