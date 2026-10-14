# Phase 8: explicitly scoped validation

Phase 8 adds `model_validator` to newly generated packages. It reads the preserved RDF source closure, translates subclass conditions into the shared class-expression records, and checks a caller-supplied registry. It does not mutate objects, change their types, supply defaults, merge identities, or run an OWL inference engine.

## Installation and testing

With the existing KRROOD environment active, run these from the extracted `krrood-ontology` directory:

```bash
python -m pip install --no-deps -e .
python -m krrood_ontology --version
python -m pytest tests/test_validation.py -q
python -m pytest tests -q
```

Expected version: `0.8.0`. The focused suite has eight tests; the full suite has 109. Install pytest separately if needed. `python -m pip install -e '.[dev]'` also installs the project's dependencies using the pinned KRROOD source.

Regenerate a package to obtain its new validation module:

```bash
python -m krrood_ontology compile tests/fixtures/validation.ttl \
  --output generated_phase8 --package validation_model \
  --structural-only --binding-policy local-usage
python examples/verify_phase8.py generated_phase8
```

The verification demonstrates incomplete-data unknowns, a valid complete scope, a universal restriction violation, qualified counting that excludes a different filler type, and a functional conflict.

For your integration ontology:

```bash
python -m krrood_ontology compile examples/aicor.ttl \
  --output generated_aicor_phase8 --package aicor_model \
  --structural-only --binding-policy local-usage
python examples/verify_aicor_phase8.py generated_aicor_phase8
```

A verified generated copy is included at `examples/aicor_generated`. Test it directly with `python examples/verify_aicor_phase8.py examples/aicor_generated`. Each output directory must be new. The expanded profile still uses `--structural-only`: compilation preserves broader OWL semantics and the optional validator checks its declared operational profile separately. No constraint is automatically enforced during construction or mutation.

## API

```python
from validation_model import Child, Part, model_validator

assembly = Child()
part = Part()
assembly.parts.append(part)
assembly.tag.append('ready')
registry = {'urn:assembly': assembly, 'urn:part': part}

report = model_validator.validate(registry)  # incomplete, conservative OWL identity
report = model_validator.validate(
    registry, complete=True, identity_policy='python'
)
print(report.valid)
print(report.to_dict())
print(model_validator.coverage())
```

A `LoadedIndividuals` object can replace the registry when it comes from the supported loader. Scope keys must be strings, values must have registered concrete generated types, and each Python object must have one scope IRI. Do not rely on other objects present in KRROOD's global graph: every object reference needed by a check should be in this registry. An out-of-scope reference produces an unknown rather than being silently pulled into the scope.

`complete=True` declares that mapped field values and explicit structural types in this registry are complete **for the application check**. It is not a statement of OWL entailment or whole-ontology completeness. A returned `valid=True` means all applicable checks in the named profile are satisfied; it does not establish ontology consistency. Empty scopes are vacuous. Diagnostics retain subject IRI, owner class IRI, axiom identifier, property IRI, status, message, and cardinality observations where relevant.

| Status | Interpretation |
| --- | --- |
| satisfied | A positive witness is established, or a complete application scope meets the check |
| violation | A known conflicting fact/distinct count, or the declared complete scope fails the check |
| unknown | Missing witnesses, unresolved membership/identity, additional possible values, or references outside scope |
| unsupported | An applicable expression or representation cannot be evaluated faithfully by this profile |

`report.valid` is False when violations exist, None when only unknown/unsupported outcomes prevent success, and True otherwise. `report.violations`, `report.unknown`, and `report.unsupported` provide filtered checks. `report.omissions` lists known profile exclusions. `coverage()` describes schema parsing separately from runtime applicability.

## Supported checks

- Inherited anonymous subclass conditions; named parents use existing generated inheritance.
- Object and supported datatype existential/universal restrictions.
- Object or datatype hasValue; Boolean false is distinguished from numeric zero.
- Qualified min/max/exact cardinality with class or supported datatype fillers; unqualified cardinality is also supported.
- Union/intersection filler expressions and supported named-individual enumerations within conditions. They do not enable equivalent-class classification.
- Functional object/datatype properties on applicable projected fields.
- Pairwise disjointWith and named AllDisjointClasses groups.
- Declared supported global ranges on applicable projected fields.

Cardinalities count qualifying fillers, not all property values. Repeated references count once. Datatype comparisons use supported converted Python values, keeping Boolean and numeric value spaces separate; this is not a complete OWL datatype/facet implementation. Nonfinite numbers and unrepresentable values are reported unsupported.

An allValuesFrom condition alone does not require a value: an empty complete field satisfies it. In an incomplete scope, failure to observe a class membership is generally unknown, unless a declared disjoint type supplies known negative evidence. Unsupported expressions are reported when applicable; unsupported constraints outside the scope remain visible in coverage.

## Identity policy

The default `identity_policy='owl'` does not assume different individual IRIs or Python objects denote distinct OWL individuals. It uses differentFrom and named AllDifferent groups from the schema and an optional identity graph to establish conservative count lower bounds. These bounds are not maximal entailment results. Identity uncertainty can remain even when `complete=True`.

```python
from rdflib import Graph, URIRef, OWL
identity = Graph()
identity.add((URIRef('urn:p1'), OWL.differentFrom, URIRef('urn:p2')))
report = model_validator.validate(registry, complete=True, identity_graph=identity)
```

With `identity_policy='python'`, distinct Python objects count as distinct application entities. This is an explicit application policy, not an asserted OWL unique-name rule. `sameAs` normalization remains unsupported and is reported. Using AllDifferent as count evidence does not implement identity merging or extend the old individual loader to load arbitrary identity axioms.

## Reuse and boundaries

The implementation imports native KRROOD EQL for explicit type queries, exists/for_all and distinct counts. It reuses the existing ModelQueries type matching, shared ClassExpression records, expression-list validation, generated field mappings, RDFLib parsing/conversion and the shared datatype table. The wrapper translates RDF constraints and reports scope/identity uncertainty; it does not introduce another query or relation engine.

The pinned native ForAll raises on an empty domain, so the adapter supplies the standard vacuous boundary value. Positive ordinal query domains avoid the pinned Exists interpreting ordinal zero as a false binding. Neither adaptation patches or replaces upstream machinery.

Important limits:

- Equivalent-class definitions do not infer types or supply necessary-condition checks in this profile; Phase 9 remains planned.
- Functional checks apply only where a projected field exists. Unbound functional properties are listed in coverage/report omissions. Fields missing on a particular class are outside the mapped-field profile; `complete=True` does not make them checked.
- Complements, data facets, property chains, arbitrary datatypes and unsupported anonymous disjointness operands are outside the profile.
- Existing relation inference behavior remains governed by the chosen compilation profile. The validator does not add transitivity, symmetry, reflexivity, or asymmetry rules.
- Source closure files must remain beside their generated packages. Validation parses them lazily and caches the schema; regenerate rather than editing them after validation begins.
- The complete AICOR individual loader remains limited by the earlier identity/type policy. The examples deliberately construct a registry of generated objects.

## AICOR evidence

The schema parser handles all 1,342 anonymous subclass conditions (the 1,341 restrictions plus one superclass union), 119 functional property declarations and 306 distinct disjointness pairs. There are no unsupported subclass conditions in this input. Eight functional properties have no projected fields and remain explicit omissions; parsing 119 declarations is not equivalent to checking 119 properties on every object.

Runtime examples verify the AccessibilityCritic required value and the two distinct qualified exact-cardinality constraints on AICORReferenceManipulationExecutive. They assert targeted outcomes; they do not label the complete ontology consistent or every generated object fully populated.
