# Action knowledge ontologies — current release 0.22.0

Current increment: [primary-action generation and consistency guide](domains/PRIMARY_ACTION_GENERATION_GUIDE.md). Release 0.22.0 strengthens pouring, pickup, placement and navigation within three existing domain ontologies and their shapes. No ontology modules or imports were added; the registry still contains 40 documents. Other missing CoraPlex actions remain deferred. Verification is indexed in `reports/release-0.22.0.json`.

The [0.21.0 coverage review](review/coverage-0.21.0/REVIEW_REPORT.md) remains a dated architectural assessment; its recommendation to add action families is deferred by the user's current four-action scope.

Read [START_HERE.md](START_HERE.md) and [ARCHITECTURE.md](ARCHITECTURE.md) for the current package: seven reusable domains, two compositions, one tool-refinement module, seventeen generic aspect/contract modules and their DUL-based shared core. The ordered recipes and specialized tool profiles introduced in 0.20.0 are documented in [compositions/ACTION_RECIPES.md](compositions/ACTION_RECIPES.md) and [refinements/SPECIALIZED_TOOL_INTERACTIONS.md](refinements/SPECIALIZED_TOOL_INTERACTIONS.md). The transport/carrying composition is documented in [compositions/TRANSPORT_CARRYING.md](compositions/TRANSPORT_CARRYING.md). The articulated-mechanism module is documented in [domains/ARTICULATED_MECHANISMS.md](domains/ARTICULATED_MECHANISMS.md). The tool-interaction/material-transfer modules and populated illustration are documented in [domains/TOOL_INTERACTION_MATERIAL_TRANSFER.md](domains/TOOL_INTERACTION_MATERIAL_TRANSFER.md) and [demo/README.md](demo/README.md).

The remainder of this file is the historical initial AICOR consolidation report. Its original scope and validation statements describe that initial increment, not the entire current release. Current checks and limitations are indexed in `reports/release-0.22.0.json`.

---

> Import architecture updated in release 0.6.0: see `MODULAR_IMPORTS.md` at the package root. Earlier import-chain/open-file instructions below are superseded. Vocabulary descriptions still apply.

# Unified AICOR ontology — initial candidate

This package compares the two supplied AICOR ontologies, preserves their originals, and produces a structurally deduplicated, DUL-aligned candidate. Scope is limited to these inputs; no new frame, phase, causal or action-specific ontology has been built.

## Decision

Use **cram-aicor-ontology.ttl** as the baseline. After normalizing anonymous OWL expressions, it contains every assertion in the supplied aicor-ontology.ttl. Both use the same ontology namespace. A blind union would repeat anonymous restrictions without adding knowledge from the smaller file.

| Input | Named classes | Object properties | Datatype properties | Triples |
|---|---:|---:|---:|---:|
| AICOR | 709 | 578 | 84 | 7,940 |
| CRAM-AICOR | 1,017 | 811 | 155 | 11,596 |
| DUL | 76 | 107 | 5 | 1,751 |

There are 1,376 shared named subjects, 612 additional named subjects in CRAM-AICOR, and no subjects exclusive to AICOR. Eighteen shared subjects have additional assertions in CRAM-AICOR. Counts of named subjects include more than classes.

## Files

- `DUL.owl`: unchanged supplied foundation.
- `sources/`: unchanged supplied domain ontologies.
- `candidates/aicor-unified-preserved.ttl`: structurally deduplicated baseline, retaining original semantics.
- `candidates/aicor-unified-dul.ttl`: recommended working candidate with real DUL references and import.
- `candidates/aicor-unified-dul-offline.owl`: same candidate plus supplied DUL, flattened for offline reasoning. Open this file for a self-contained inspection.
- `reports/`: inventory, mappings, exact shared-term changes, quarantined axioms and validation evidence.
- `tools/`: reproducible build and reasoning scripts.

The DUL candidate has 11,139 asserted triples, excluding imported DUL, and 1,003 locally declared named classes. The reduction of 14 named classes reflects replacement of placeholder DUL classes by actual imported DUL classes.

## Reproduce

Use Python 3 and Java (tested with Java 21). From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python tools/build.py
python tools/consistency.py
python tools/reason.py
```

The last command attempts full classification with a 90-second limit; it is optional and may require a longer limit for this ontology. `catalog-v001.xml` beside the Turtle candidate resolves its DUL import locally in compatible ontology tools.

## Validation status

- Both input Turtle files and supplied DUL parse successfully.
- Structural containment of AICOR in CRAM-AICOR verified.
- Serialized outputs were reparsed and compared structurally.
- Named subclass/disjointness screening: 175 conflicting ancestry pairs before the five documented parent removals, zero after.
- HermiT ontology consistency check passed on the flattened candidate plus DUL.
- Full HermiT classification timed out after 90 seconds. **Satisfiability of every named class is not established.** An ontology may be consistent while containing unsatisfiable classes.

This is a reviewable candidate, not a claim that all semantic redundancies, modelling problems or runtime mappings have been resolved. Read REVIEW.md before extending it.

To use on your computer, extract this folder under `/home/malineni/ontologies/`. Original files outside this package are not modified.
