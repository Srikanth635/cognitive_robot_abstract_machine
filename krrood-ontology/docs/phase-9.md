# Phase 9: positive definitions and native relation additions

Phase 9 adds an opt-in `--scoped-semantics` profile to `local-usage`. Generated objects remain ordinary dataclasses inheriting KRROOD `Symbol`. Queries find positive evidence in a supplied registry; they do not change Python classes or add class assertions. Relation additions run through existing KRROOD descriptors, mixins, monitored sets and graph implications.

## Install and test

From the extracted `krrood-ontology` directory, with your KRROOD environment active:

```bash
python -m pip install --no-deps -e .
python -m krrood_ontology --version
python -m pytest tests/test_scoped_semantics.py -q
```

Expected version: `0.9.0`. Use the underscore in `python -m krrood_ontology`. The focused suite has nine tests; the full suite has 118 (`python -m pytest tests -q`). Install pytest separately if needed. To install dependencies in a fresh environment, use `python -m pip install -e '.[dev]'`; this uses the KRROOD source commit pinned in `pyproject.toml`.

Regenerate your models to obtain the Phase 9 APIs. Every output directory must be new:

```bash
python -m krrood_ontology compile examples/scoped_semantics.ttl \
  --output generated_phase9 --package scoped_model \
  --structural-only --binding-policy local-usage --scoped-semantics
python examples/verify_phase9.py generated_phase9

python -m krrood_ontology compile examples/aicor.ttl \
  --output generated_aicor_phase9 --package aicor_model \
  --structural-only --binding-policy local-usage --scoped-semantics
python examples/verify_aicor_phase9.py generated_aicor_phase9
```

Verified generated copies are included in `examples/scoped_generated` and `examples/aicor_generated`. Test those directly by passing the relevant directory to the verification script. The broader source still requires `--structural-only` because this operational profile covers selected semantics and retains the remaining OWL axioms. Phase 7/8 commands without `--scoped-semantics` retain their previous runtime behavior. The original source snapshots and closure remain available beside every generated package.

## Generated API

```python
from scoped_model import Alpha, Beta, Carrier, model_queries, model_relations

base = "https://example.org/scoped#"
a = Alpha()
b = Beta()
carrier = Carrier()
carrier.link.append(a)
scope = {"urn:a": a, "urn:b": b, "urn:carrier": carrier}

assert model_queries.match(base + "Either", scope) == [a, b]
assert model_queries.match(base + "Linked", scope) == [carrier]
assert type(carrier) is Carrier

model_relations.initialize_reflexivity(scope)  # explicit, repeatable self-edge seeding
assert a in a.connected

a.connected.add(b)
b.connected.add(carrier)
assert carrier in a.connected                 # native transitivity
assert a in carrier.connected                 # native symmetry
facts = model_relations.snapshot(scope)        # both endpoints must be in scope
```

`model_queries.query(...)` returns a native EQL query, and `.match(...)` evaluates it. Both accept a `LoadedIndividuals` registry or a string-IRI-to-object mapping. Optional candidates must belong to the supplied scope by identity. `CLASS_EXPRESSIONS` and `CLASS_ALIASES` expose the translation records.

## Supported definition profile

- Named equivalence aliases, including their asserted ancestor memberships, without Python class merging or reclassification.
- Anonymous union/intersection definitions, including singleton lists; named enumerations; object/literal `hasValue`; object `someValuesFrom` with nested positive expressions.
- Definitions asserted in either direction. An equivalence component supports one anonymous definition; repeated identical RDF nodes are harmless. Multiple distinct definitions and recursive dependencies are rejected before publishing output.
- Ordinary local fields propagate to named aliases, including fields inherited by the aliased class. Definition-local fields use the existing native registration machinery.
- Enumerations and named `hasValue` witnesses match the named objects present in the caller's registry. Missing members supply no positive witness; a missing match does not establish OWL nonmembership.
- Literal witnesses use RDFLib value equality, so Python `False == 0` does not become RDF boolean equality.

Universal restrictions, cardinalities and complements are outside the positive query profile. Phase 8 continues to check supported universal/cardinality constraints. A requested definition outside the profile fails explicitly. Imports in this profile must resolve through supplied local files; unresolved imported definitions cannot silently disappear.

## Relation storage and validation

Properties participating in inverse, symmetry, transitivity, named subproperty, reflexivity or asymmetry declarations receive one shared association on the private `_OntologyEntity` storage base. It is not an ontology class and is absent from `CLASS_BY_IRI`. This association lets native KRROOD implications update different generated types without crossing unrelated wrapped fields. Only these properties are shared; all other local fields keep their existing ownership.

Shared relation fields are monitored sets, populated with `.add(...)` after construction. Original asserted domain/range metadata remains in `MODEL.properties`; operational root bindings have an empty owner IRI and are recorded separately in `MODEL.bindings`. Broader storage does not infer OWL domain/range membership. The individual loader still checks asserted signatures before construction. Scoped validation checks declared domains for nonempty fields and declared ranges for their targets.

`initialize_reflexivity(scope)` uses native `PropertyDescriptorRelation` insertion with `inferred=True`. It seeds only supplied generated objects compatible with any asserted named domain/range, reports `added` and skipped `(individual_iri, property_iri)` pairs, and is idempotent. It does not enumerate the global graph or run automatically during construction. This is an explicit operational policy, not full OWL reflexive-property entailment over every individual.

`model_validator` in this profile uses positive definition membership for constraint applicability and disjointness. It also checks equivalent definitions as necessary conditions and reports asymmetric self-loops/reverse edges. Known conflicts are violations; outside-scope targets or incomplete absence remain unknown. Construction and mutation do not automatically enforce these checks.

`snapshot(scope)` returns immutable records with `subject_iri`, `property_iri`, `object_iri` and `inferred`. The boolean is KRROOD's **first insertion flag**. An inferred edge later added explicitly retains that flag. It is not a proof history or a complete separation of asserted and inferred facts. Asserted RDF remains in the source snapshots or loader's `asserted_graph`. Native implications can involve objects outside a snapshot scope; the scope filters inspection and validation, not KRROOD's global graph operations.

## Reuse and limits

`scoped.py` extends translation using the existing constraint AST/parser, RDFLib traversal and Phase 3 `RelationSemantics`. `queries.py` delegates matching to native EQL type predicates, logic, symbolic functions and existential witnesses. `relation_runtime.py` seeds and reads native relations; it contains no alternative transitive/inverse/subproperty closure algorithm. Existing native restrictions on multiple superproperties/inverses, descriptor cycles and transitive-superproperty over-strengthening still apply.

There is no retraction/recomputation engine, property-chain translation, full OWL classification, identity normalization, or automatic Python type change. Phase 10 remains the next planned work for AICOR's four named individuals, AllDifferent-aware loading, identity policy and source-preserving export/round-trip acceptance. The current enumeration check supplies an explicit registry and does not claim to load that A-Box.

Measured AICOR results are saved in `phase-9-integration-results.json`. The 1,017 named classes and 966 global property definitions are preserved. The binding count changes because five previously local relation properties now share root associations; this is a storage change, not a loss of ontology property definitions.
