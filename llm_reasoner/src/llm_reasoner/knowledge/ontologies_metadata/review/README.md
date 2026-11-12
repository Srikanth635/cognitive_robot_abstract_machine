# Running the competency review

Current cross-release review: [coverage-0.21.0/REVIEW_REPORT.md](coverage-0.21.0/REVIEW_REPORT.md). Run `python tools/review_current_coverage.py` from the package root for current nested tool/recipe query checks. The V7 suite below intentionally retains its historical scope.

Read `REVIEW_REPORT.md` for findings and `COMPETENCY_QUESTIONS.md` for the question catalog. Run these commands from the package root with `requirements.txt` installed:

```bash
python tools/review_competency.py
python tools/validate_imports.py
python tools/validate_aspect_completion.py
python tools/reason_imports.py
python tools/check_strategy_integration.py
```

The first command writes `reports/competency-review.json`. The classification and combined consistency commands require Java. Historical reports elsewhere in the package retain the scope of their original release.

Run one descriptive question against one explicit record:

```bash
python tools/review_competency.py \
  --record 'http://aicor.knowledge/ontologies/action-examples.owl#pourRecordV7' \
  --question CQ12
```

Standalone SPARQL files live in `queries/competency/`. Bind `?record` using your SPARQL client's initial bindings or add a `VALUES ?record { <...> }` clause. Load the example graphs and the local generic schema/import closure. These queries do not require a full OWL materialization: CQ12 follows the declared `rdfs:subPropertyOf` hierarchy explicitly. They retrieve declared structure; they do not apply evidence-acceptance policies or infer world facts.

Results preserve full RDF term IRIs, datatypes and language tags. Empty answers have question-specific meanings described in the catalog. For example, no validation evidence means **not supplied in this record**, not **the model is invalid**. No current frame projection means no matching artifact, not that a frame cannot be represented.

To repeat the exact consolidation audit, supply the original 0.9.0 cumulative ZIP:

```bash
python tools/audit_consolidation.py /path/to/gpt_cram_ontologies_aspects_v0.9.0.zip
```

Do not pass the 0.9.0 baseline to the older `validate_imports.py` axiom-preservation option; that option compares the historical 0.5/0.6 restructuring. The dedicated consolidation audit checks this release's intended logical delta.

## Answer interpretation contract

1. Always select a concrete record IRI; never silently substitute a newer version.
2. Treat retrieved action components as descriptions. Inspect CQ22 for included statements and epistemic status.
3. Match provenance to the exact statement. A claim about one component does not endorse all triples reachable from that component.
4. Treat missing evidence, numbers and bindings as unspecified. Do not turn them into false, zero, a default robot, or a successful execution assessment.
5. Keep record membership separate from storage isolation. Shared component IRIs remain mutable unless the storage layer enforces immutability; these query tests do not provide that storage layer.
6. The queries intentionally inspect bounded routes. CQ04 covers direct action parameters; CQ12 covers the owners enumerated in its query. New nested action structures need additional reviewed routes.
