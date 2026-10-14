# Phase 7: generic structural compatibility

Phase 7 is implemented as an opt-in structural profile. It produces KRROOD-backed dataclasses for tested named multiple inheritance, supports missing global property signatures through an explicit local-usage field policy, and preserves anonymous class axioms as source-linked RDF records. It does not execute these restrictions or property characteristics. Existing explicit-signature semantic/query compilation remains available unchanged.

## Install and run

From the extracted `krrood-ontology` directory, with KRROOD already installed in your environment:

```bash
python -m pip install --no-deps -e .
python -m krrood_ontology --version
python -m pytest tests -q
```

The module name uses an underscore. Install pytest separately if it is not already present. To install the project's dependencies as well, use `python -m pip install -e '.[dev]'`; this uses the pinned upstream KRROOD dependency.

A small generic case is included:

```bash
python -m krrood_ontology compile tests/fixtures/local_usage.ttl \
  --output generated_phase7 --package local_model \
  --structural-only --binding-policy local-usage
```

Then run from the same directory:

```bash
PYTHONPATH=generated_phase7 python - <<'PY'
from local_model import Combined, Left, Right, MODEL, load_individuals
loaded = load_individuals('tests/fixtures/local_usage.ttl')
a = loaded['https://example.org/structural#a']
b = loaded['https://example.org/structural#b']
assert isinstance(a, Combined) and isinstance(a, Left) and isinstance(a, Right)
assert a.connects[0] is b
assert a.tag == ['ready']
assert MODEL.properties['https://example.org/structural#connects'].domain_iri == ''
print('Phase 7 passed')
PY
```

To regenerate the supplied integration ontology:

```bash
python -m krrood_ontology compile examples/aicor.ttl \
  --output generated_aicor --package aicor_model \
  --structural-only --binding-policy local-usage
python examples/verify_aicor_phase7.py generated_aicor
```

Output directories must be new. A verified generated copy is also included at `examples/aicor_generated`. Run `python examples/verify_aicor_phase7.py examples/aicor_generated` to exercise it directly. The example-specific names occur only in fixtures and verification scripts; the translator contains no AICOR namespace or class-name rules.

## Native machinery and wrapper responsibilities

| Responsibility | Implementation |
| --- | --- |
| Model identity and graph integration | Native KRROOD Symbol |
| Object field registration and collection tracking | Native PropertyDescriptor and its monitored containers |
| One property on multiple classes | Multiple instances of the same native descriptor class, using its domain/range association map |
| Source rendering and formatting | Existing KRROOD CodeGenerator and formatter |
| Literal conversion | Existing RDFLib conversion and shared datatype table |
| Named inheritance validation | Python type construction/C3 and graphlib topological sorting |
| Naming | Existing KRROOD naming helpers plus generic keyword suffix and collision checks |
| Local field projection and axiom records | Thin ontology translation policy in structural.py |

No new inference engine, descriptor implementation, monitored collection or identity registry was introduced.

## Mapping contract

- The default `explicit` profile retains the earlier signature and semantic boundaries. Select `local-usage` explicitly; it currently requires structural-only mode.
- Named parent IRIs are retained in the manifest. Python bases omit redundant ancestor bases and follow a deterministic topological ordering. Cycles and incompatible MROs fail before publication. Anonymous superclass expressions are records, not Python bases.
- Keywords and typing namespace names receive a generic trailing underscore (`Type` becomes `Type_`): an ontology class named `None` maps to `None_`. Distinct IRIs producing the same name still fail; no class name is special-cased.
- A property with a declared named domain receives a field on that domain. Without a domain, traversal of class-owned anonymous subclass/equivalent-class expressions supplies convenient local field owners. Inherited associations are reused; unrelated owners get separate native registrations of the same descriptor class.
- Local ownership is an API policy, not an inferred global OWL domain. Properties without declared domains can logically apply elsewhere. An assertion on a class lacking a projected field currently fails with an unmapped-field diagnostic; no assertion is silently dropped. Properties with no eligible usage remain in `MODEL.properties` and the IRI/name vocabulary without eager fields. The feature report lists them.
- Missing object ranges use Symbol as the operational storage endpoint. This is not an asserted or inferred OWL range. Missing datatype ranges use Any and RDFLib literal conversion. Declared supported literal ranges retain exact-range checks in the loader. Restriction fillers and hasValue literals do not become global range declarations.
- Multiple or anonymous global domain/range signatures remain unsupported with explicit diagnostics. Arbitrary datatype mappings and unrelated individual types are still outside the profile.
- `PROPERTY_BINDINGS[(property_iri, owner_class_iri)]` contains actual native descriptor instances. `PROPERTY_DESCRIPTORS` retains a representative bound instance per property for compatibility. Unbound properties have no descriptor instance. `MODEL.bindings` describes both literal and object field associations; global definitions remain separate in `MODEL.properties`.

## Preserved axioms and reporting

`SCHEMA_AXIOMS` contains class owner, predicate, root node and outgoing RDF triples for anonymous subclass/equivalent-class expressions, plus AllDifferent/AllDisjointClasses groups. Terms use full N3 forms, retaining literal datatypes, list structure and nested blank nodes. These records are RDF structures for future translators, not an already validated OWL expression AST. Other axioms remain in the complete source closure and original byte snapshots.

Blank-node labels in this expanded profile are graph-local and share identity with the written closure file. To avoid expensive whole-graph canonicalization on repetitive ontologies, the expanded profile serializes the original RDFLib graph. Source bytes/hashes and RDF meaning are preserved; byte-identical output across separate parses is not promised. The manifest records this policy. The earlier explicit profile still uses RDFLib canonicalization.

Structural-only compilation does not perform individual preflight. Unsupported identity declarations such as AllDifferent still prevent loading the complete AICOR ontology through the old individual loader; Phase 10 addresses that boundary. Class constructors and native object fields are usable now. Required-value restrictions do not become automatic defaults, and no validators or automatic reclassification are claimed.

## Verified accomplishment

The supplied AICOR input produces 1,017 dataclasses, 966 property definitions, 1,183 field associations, 1,356 class-expression records and 13 global group records. All 1,341 restriction nodes are retained. Package import and descriptor registration pass isolated verification. Representative multiple inheritance, construction, native relation collection updates and literal fields are exercised separately. The existing 96 tests and five new generic structural tests pass.

Next: Phase 8 consumes the preserved class-local axioms to implement explicitly scoped validation. Qualified cardinality, universals, functionality and disjointness remain planned, with open-world limitations kept explicit.
