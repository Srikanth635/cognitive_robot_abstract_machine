# Phase 10: declared identity and source-preserving RDF round trips

Phase 10 adds generic declared-distinctness loading and an RDF exporter. Source ontology names, namespaces, individual counts and class/property names do not appear in implementation branches. AICOR and the small `identity_roundtrip.ttl` ontology are test fixtures.

## Installation and testing

Run from the extracted `krrood-ontology` directory with your KRROOD environment active:

```bash
python -m pip install --no-deps -e .
python -m krrood_ontology --version
python -m pytest tests/test_identity_export.py -q
python -m pytest tests -q
```

Expected version: `0.10.0`. The focused suite has 12 tests; the full suite has 130. For a fresh environment, `python -m pip install -e '.[dev]'` installs development dependencies and the pinned KRROOD source. Python module execution uses the underscore: `krrood_ontology`.

Regenerate models to obtain the new exports; choose fresh output directories:

```bash
python -m krrood_ontology compile examples/identity_roundtrip.ttl \
  --output generated_phase10 --package roundtrip_model \
  --structural-only --binding-policy local-usage --scoped-semantics
python examples/verify_phase10.py generated_phase10

python -m krrood_ontology compile examples/aicor.ttl \
  --output generated_aicor_phase10 --package aicor_model \
  --structural-only --binding-policy local-usage --scoped-semantics
python examples/verify_aicor_phase10.py generated_aicor_phase10
```

Verified generated copies are supplied in `examples/identity_generated` and `examples/aicor_generated`. Pass those directories to the scripts to exercise the included artifacts directly. The full ontology still requires `--structural-only`, since executable profiles cover selected semantics while the RDF source retains the rest. Compilation warnings describe omitted structural semantic mappings; runtime loading and validation check their separately documented profiles.

## Loading identities

```python
from roundtrip_model import load_individuals, model_queries, model_validator

loaded = load_individuals("examples/identity_roundtrip.ttl")
one = loaded["https://example.org/roundtrip#one"]
assert len(loaded.by_iri) == 3
assert loaded.identity_policy == "distinct"
print(loaded.distinct_groups)
report = model_validator.validate(loaded, complete=True)  # OWL identity policy by default
```

New generated `load_individuals` functions default to `identity_policy="distinct"`. This policy:

- Preserves one object per named IRI, without merging aliases or treating different names as proof of different OWL entities.
- Parses named `owl:differentFrom` operands and `owl:AllDifferent`/`owl:distinctMembers` RDF lists. It records distinctness groups and retains every original identity triple and list node in the asserted graph.
- Rejects self-difference, repeated AllDifferent members, malformed/cyclic lists, anonymous identity operands and operands lacking explicit loaded types, before object construction.
- Distinguishes schema declaration/axiom nodes from individual candidates when loading a combined ontology.
- Continues to select a single explicit generated class from inheritance-compatible asserted types. The original asserted type set remains available in `loaded.asserted_types`.

`identity_policy="reject"` retains the old identity-axiom rejection behavior. The shared low-level `IndividualLoader.load` and `IndividualPlanner` still default to `reject`; generated Phase 10 wrappers explicitly select `distinct`. `sameAs` remains rejected under both policies. There is no identity normalization, unique-name assumption or automatic inferred typing. Unrelated asserted types remain an explicit loading error; Phase 10 does not synthesize undeclared Python intersection classes or duplicate an individual's identity into type views.

The validator and loader now share `IdentityConstraints` parsing and the existing checked RDF-list helper. OWL cardinality lower bounds use explicit distinctness evidence from the loader's original asserted graph. Merely creating separate Python objects does not establish OWL distinctness.

## RDF export

```python
from decimal import Decimal
from roundtrip_model import model_exporter

# Copy asserted source, even if object fields have since changed.
source_graph = model_exporter.graph(loaded)
model_exporter.write(loaded, "source-copy.ttl")

# Update mapped object assertions while retaining schema and source metadata.
one.amount[:] = [Decimal("2.50")]
result = model_exporter.write(loaded, "updated.ttl", mode="current")
print(result.added_triples, result.removed_triples, result.omitted_inferred_facts)

# Explicitly project currently present inferred field values as RDF assertions.
with_inference = model_exporter.graph(loaded, mode="current", include_inferred=True)
```

`model_exporter.export(...)` returns `ExportResult(graph, mode, added_triples, removed_triples, omitted_inferred_facts)`. `.graph(...)` returns its RDFLib graph, and `.write(...)` writes a fresh file after successful preflight. Existing files are never overwritten. `format="nt"`, `format="turtle"` and other RDFLib serializer formats can be selected explicitly; Turtle is the default.

**Source mode** copies the parsed asserted RDF graph. It includes the original schema, anonymous expressions, identity axioms, annotations, unmapped metadata and original type assertions. Object-field edits do not affect it. Original file bytes and syntax remain separately available as `loaded.source_bytes` and `loaded.source_format`; compiler source snapshots also retain the input bytes. Exported formatting and blank-node labels need not match those bytes. RDFLib's existing literal parsing/normalization policy applies to the parsed graph.

**Current mode** starts with that source graph and synchronizes only mapped property assertions for the loaded registry. It checks source/type compatibility, references, declared signatures, literals and identity lists through the existing planner without constructing new objects. Unchanged converted literal values retain their original RDF terms, including datatype and language metadata. Newly supplied values use RDFLib conversion and declared datatypes when present; an incompatible conversion fails. Python booleans and numbers remain distinct, so `False` and `0` can coexist without losing an original assertion. Lists/sets have RDF set semantics on export.

Current mode preserves original asserted types, annotations and all schema/unmapped triples. It does not edit annotations via `loaded.annotations`, add/delete individuals, or change their selected classes. Registry membership changes, unregistered reference targets and incompatible class replacement are rejected rather than silently discarded. New untyped identity operands are likewise rejected. A registry lifecycle/type-edit API remains future work.

Native first-insertion edge flags determine inference omission. Inferred edges absent from the original source are omitted by default; original source assertions remain eligible even when their native edge was first inferred during loading. `include_inferred=True` projects all currently present mapped values, which makes those additional facts ordinary assertions when reloaded. A later explicit addition of an already inferred edge cannot be distinguished by KRROOD's flag; use the explicit inclusion policy when that distinction matters. Snapshot flags are not a complete proof history.

Export does not mutate the native graph, enforce all OWL constraints, or introduce a retraction engine. A projected graph can be validated and loaded into a fresh registry. Live native inverse/transitive retraction behavior remains outside the supported additions-only profile. Unsupported converted values, including nonfinite numeric values in current-mode comparisons, fail explicitly.

Relative references are resolved by the existing RDF loader and exported as their resolved IRIs. Moving an export to a different directory therefore preserves individual identities and references. Original-byte retention is separate from this graph-level round-trip contract.

## Reuse and validation evidence

- `identity.py` reuses RDFLib terms and the existing `expression_members` list checks. It introduces no identity-merging engine.
- `export.py` reuses `IndividualPlanner.plan_graph`, generated mappings, native relation snapshots and RDFLib graph serialization.
- `values.py` extracts the existing validator value-comparison policy for shared use; it does not introduce an additional comparison mechanism.
- KRROOD supplies generated Symbol identity, descriptors, monitored collections, implications and EQL. Its native role machinery maintains distinct role objects, so it does not replace the selected one-object-per-IRI loading contract.

Generic tests cover distinctness errors before construction, complete unchanged/edited Turtle round trips, annotation/datatype preservation, inference policy, relative-IRI relocation and rejection without partial output. AICOR acceptance loads four individuals, checks their declared group of four, exercises the enumeration query, validates the loaded scope and round-trips all 11,596 triples. The N-Triples whole-graph check uses RDFLib's parser with the emitted blank labels as a parser context and compares every triple; it avoids costly whole-graph canonicalization. Reloading is additionally checked through the ordinary loader. Small generic Turtle round trips use RDFLib graph isomorphism.

See `phase-10-integration-results.json`, `phase-10-generic-results.json` and `phase-10-test-results.json` for measured evidence. These are fixture acceptance results for the documented profiles, not complete OWL conformance. Broader `sameAs` normalization, unrelated-type views, lifecycle edits, property chains and further OWL translation remain future improvements.
