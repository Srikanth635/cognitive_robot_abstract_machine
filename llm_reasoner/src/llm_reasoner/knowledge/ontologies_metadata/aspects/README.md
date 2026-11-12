# Knowledge aspect organization — 0.14.1

The seventeen aspect/contract ontologies are grouped by responsibility. The original fifteen aspect families and two supporting contract modules keep their existing ontology IRIs and axioms. Folder membership is an organizational aid, not an import or inheritance relation.

| Folder | Ontology modules |
|---|---|
| `structure/` | `participants-relations.ttl`, `conditions-outcomes.ttl`, `phases-transitions.ttl` |
| `observation/` | `monitoring-requirements.ttl`, `sensorimotor-events.ttl` |
| `causality/` | `causal-mechanisms.ttl` |
| `physics-control/` | `physical-models.ttl`, `dynamics-control.ttl`, `model-contracts.ttl` |
| `planning/` | `objectives-tradeoffs.ttl`, `capabilities-strategies.ttl`, `failure-diagnosis.ttl` |
| `perspectives/` | `frame-perspective.ttl`, `force-dynamics.ttl`, `image-schemas.ttl` |
| `entities-transfer/` | `entities-materials-affordances.ttl`, `transfer-contracts.ttl` |
| `validation/` | Eight SHACL shape files, including checks that span several groups |

Cross-module semantic bridges remain in `../integrations/`. Reusable action domains remain in `../domains/`. The foundation and provenance/context modules remain in `../core/`. The explanatory documents stay beside this index so their existing links remain usable.

Open `../integrations/all-action-aspects.ttl` for generic aspects, or `../integrations/all-action-knowledge.ttl` for the aspects plus the two implemented domains. Local catalogs are supplied in each aspect subfolder, and `../ontology-registry.json` maps the stable ontology IRIs to their new locations.

## Migration and verification

Extract this release into a fresh directory. Overlaying it onto an older extracted package leaves obsolete flat files behind and can create duplicate file matches.

This is a filesystem refactor. All RDF files, including moved shape files, are byte-identical to release 0.14.0. The ontology registry keys, import IRIs and named concepts remain unchanged. There are still 31 registered ontology documents and two implemented reusable domains.

`../aspect-layout-migration.json` maps all 25 old aspect/shape paths to their new paths. File-based callers must use these new locations or the registry. External tools that hard-code old physical paths need to apply that mapping; ontology IRI-based imports retain their identifiers.

Package scripts now resolve the moved files, scan shape folders recursively and validate nested catalogs. The historical consolidation audit retains baseline path names while resolving current paths through the migration map. Its current-layout output is `reports/consolidation-audit-current-layout.json`; later assembly changes are explicitly excluded from that narrow historical comparison. Validators that load every example now use the full action-knowledge schema so domain-specific relation signatures are available. Historical generated reports retain their original filenames and scope.

To reproduce the refactor audit, with dependencies installed:

```bash
python tools/validate_aspect_layout.py /path/to/gpt_cram_ontologies_object_manipulation_v0.14.0.zip
python tools/validate_imports.py
python tools/validate_object_manipulation.py
python tools/validate_robot_motion.py
python tools/validate_pouring_contracts.py
python tools/review_competency.py
```

Run commands from the package root. The relocation audit verifies exact prior RDF bytes, stable registry keys, unique moved copies, current literal script paths and continued inclusion of both domains. No new reasoning run is claimed for this layout-only release; semantic reasoning results retain the 0.14 scope and the known optional AICOR normalization limitation.
