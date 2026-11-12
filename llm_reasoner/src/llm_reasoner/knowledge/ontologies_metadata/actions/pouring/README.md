# Pouring action ontology and self-contained context — 0.23.0

## Files to use

| Purpose | File |
|---|---|
| Maintained modular pouring entry | `modular/pouring.ttl` |
| One document to give an LLM | `context/pouring-context.md` |
| Standalone ontology for an ontology tool | `context/pouring-complete.ttl` |
| Pouring-only symbolic template | `context/pouring-template.json` |
| Separate generation instructions | `context/GENERATION_INSTRUCTIONS.md` |
| Structural validation constraints | `validation/pouring-shapes.ttl` |
| Imported module versions and hashes | `bundle-manifest.json` |

For LLM use, provide `context/pouring-context.md` plus your instruction, world/robot context and required output serialization. The context document already contains the ontology, template and rules. The separate component files contain the same material for application use; they do not need to be attached again. The full closure can be lengthy; use a model/context strategy capable of consuming it without truncation.

For ontology tools, open the standalone ontology directly, or open the modular entry with its adjacent catalog/registry. The modular folder includes every dependency locally. All shared concept IRIs and source module bytes are preserved. The flattened ontology has one entry header and no imports; dependency header metadata is retained in the modular files and manifest. Do not edit generated copies independently: rebuild them from the maintained modules.

This is an action-specific assembly of existing knowledge representations, reusing material-transfer PouringDescription and Pouring. Its 32-document closure includes DUL, generic aspects/bridges and material-transfer, motion, manipulation and tool-interaction vocabulary. Related definitions inside those modules remain included. Imports do not mandate pickup or contact. Navigation, perception, articulated mechanisms, unrelated recipes/refinements and optional AICOR/CRAM are excluded.

The bundle contains schema and generation aids, not a populated action knowledge base. The default pouring template is partial transfer. Values and world bindings must be supplied or remain unknown. SHACL is provided for structural checks, but not as a complete output schema or replacement for domain-integrity, phase, provenance and evidence-acceptance checks. No executable Python tools, reports, demos, examples, tests or historical handoffs are included. No simulation, LLM provider call or robot execution is performed.
