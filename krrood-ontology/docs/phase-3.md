# Phase 3: native relation implications

This document records the Phase 3 milestone. Release 0.4.0 adds the explicit
individual loader documented in [phase-4.md](phase-4.md); relation behavior is
unchanged.

The accomplishment is an importable ontology model whose supported object
relation axioms execute through KRROOD. Adding an assertion can update its
inverse, propagate it to superproperties, or complete a transitive chain.
Existing EQL queries can read these populated relations. This phase adds no
independent inference engine and does not perform automatic class classification.

## Modules to build

| Module | Responsibility | Existing mechanism reused |
| --- | --- | --- |
| `semantics.py` | Validate supported relation axioms and plan descriptor inheritance/mixins | `HasInverseProperty`, `TransitiveProperty`, descriptor inheritance |
| `model.py` | Store descriptor plans and field initialization/container choices | Existing structural intermediate model |
| `translation.py` | Admit supported features; reject others or record explicit omissions | Existing diagnostic pipeline |
| `templates/relations.py.jinja` | Define descriptor classes, then attach instances to model fields | `PropertyDescriptor`, native relation graph and implications |
| `templates/classes.py.jinja` | Use empty sets for fields participating in semantic relations | KRROOD monitored sets supplied by the descriptor |
| `templates/axioms.py.jinja` | Export normalized full-IRI axiom metadata and update policy | Existing code generator |
| `generation.py`, `compiler.py` | Render modules and record the selected runtime profile | KRROOD `CodeGenerator` and formatter |
| `examples/verify_relations.py` | Exercise generated objects and queries | KRROOD EQL `variable`, `entity`, `contains` |

The earlier loader, inspection, naming and structural mapping modules remain.
The new code decides how RDF declarations map to the existing APIs; implication
execution remains in KRROOD.

## Supported relation profile

The Phase 2 structural requirements still apply, including explicit object
property declarations, one named domain and range, and single-parent classes.

| Axiom | Accepted mapping | Boundary |
| --- | --- | --- |
| `owl:inverseOf` | `HasInverseProperty.get_inverse()` | Both properties declared; exactly swapped signatures; one inverse target per property |
| `owl:TransitiveProperty` | `TransitiveProperty` mixin | Domain and range must be the same declared class |
| `owl:SymmetricProperty` | Inverse of the descriptor itself | Domain and range must be the same declared class; incompatible explicit inverse rejected |
| `rdfs:subPropertyOf` | Child descriptor subclasses parent descriptor | One acyclic parent; identical explicit signatures; no transitive ancestor |

Subproperties of transitive properties are rejected because native descriptor
inheritance would also mark the child transitive, strengthening the OWL axiom.
Multiple superproperties, datatype-property semantics, narrower subproperty
signatures and other unsupported combinations fail before generation.

`--structural-only` disables all these semantics and records the omitted axioms.
It generates the earlier list-based structural fields. It still rejects invalid
structural mappings. Functional, inverse-functional, asymmetric, reflexive and
irreflexive properties, property chains, restrictions, class equivalence,
cardinality constraints and ontology imports remain unsupported in strict mode.

## Construct first, then add assertions

```python
from relations_model import Company, Organization, Person

alice = Person(name=["Alice"])
acme = Company()
alice.works_for.add(acme)
assert alice in acme.has_employee

headquarters, branch, office = Organization(), Organization(), Organization()
headquarters.contains_part.add(branch)
branch.contains_part.add(office)
assert office in headquarters.contains_part
assert headquarters in office.part_of
```

Semantic fields have `set[Range]` annotations, independent empty defaults and
`init=False`. KRROOD turns these into its monitored sets. Call `.add()` or
`.update()` after all participating objects have been constructed. Passing a
semantic field in a dataclass constructor is intentionally unsupported: an
inverse implication can otherwise reach a field that has not been initialized.
Ordinary object fields still use the Phase 2 lists and `.append()`; literal
fields remain ordinary collection-valued dataclass fields.

Import all generated model packages **before** constructing a `SymbolGraph` or
any KRROOD `Symbol` instance. KRROOD caches its class diagram; late descriptor
registration is not a supported lifecycle in this release. Generated descriptor
metadata disables its own instance caching using KRROOD's existing
`_cache_instances_` flag, allowing registration to finish before application
objects build that diagram. Application symbols retain their normal caching.
Imports do not clear existing application objects or the global graph.

Updates are **additive only**. Removing, replacing or clearing a collection does
not establish retraction of asserted or inferred graph facts. Do not use those
operations as an ontology update API. The generated package exports
`UPDATE_POLICY = "additions_after_construction"` and
`RETRACTION_SUPPORTED = False`.

## Query inferred relations

```python
from krrood.entity_query_language.factories import contains, entity, variable
from relations_model import Person

people = variable(Person, domain=[alice])
employees = entity(people).where(contains(people.works_for, acme)).evaluate()
assert set(employees) == {alice}
```

The supplied verification script queries employees inferred through
subproperties and containers inferred through transitivity. Its explicit domain
lists make the simple case deterministic. This uses native EQL directly; this
phase does not translate arbitrary OWL class axioms into queries or reclassify
objects based on those axioms.

## Output and testing

```bash
python -m pip install --no-deps -e .
python -m krrood_ontology --version
python -m krrood_ontology compile examples/relations.ttl \
    --output generated_phase3 --package relations_model
PYTHONPATH=generated_phase3 python examples/verify_relations.py
```

Use a fresh output directory. The CLI module name is `krrood_ontology`; the
console command is `krrood-ontology`.

The generated package contains `classes.py`, `relations.py`, `vocabulary.py`,
`axioms.py` and `__init__.py`, alongside `manifest.json`. `RELATION_AXIOMS` stores
normalized full-IRI triples, including both effective inverse directions. It is
metadata, not a second implication executor or lossless source serialization.
The manifest's `inference_applied: false` means compilation does not materialize
source-graph entailments; `relation_implications: true` identifies the selected
native runtime profile. Individual assertions are still deferred to Phase 4.

Verification covers inverse updates from either side, multi-level subproperties,
symmetry, transitive chains in both insertion orders, inverse-side transitive
updates, set uniqueness, inherited fields and EQL over inferred relations.
Negative tests exercise incompatible signatures, unknown or multiple inverses,
cyclic or multiple superproperties and transitive-ancestor rejection. Existing
Phase 1/2 regression tests also remain passing. See `validation.md`.

The next milestone is Phase 4: instantiate ontology individuals, convert
literals, resolve links and establish an identity/role policy using verified
KRROOD APIs.
