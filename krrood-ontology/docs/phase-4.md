# Phase 4: ontology individuals as objects

This document records the Phase 4 milestone. Release 0.5.0 retains this API and
adds compilation-time preflight, snapshots and explicit local import resolution;
see [phase-5.md](phase-5.md).

The accomplishment is a callable loader that creates generated dataclass
instances from an asserted A-Box. References resolve through one IRI registry,
literal conversion uses RDFLib, and relation updates use the descriptors built
in Phase 3. Compilation remains separate from loading: importing a generated
package does not instantiate its source individuals.

## Install and test

Use the existing environment containing the verified KRROOD baseline:

```bash
cd krrood-ontology
python -m pip install --no-deps -e .
python -m krrood_ontology --version
# krrood-ontology 0.4.0

python -m krrood_ontology compile examples/university.ttl \
    --output generated_phase4_university --package university_model
python -m krrood_ontology compile examples/relations.ttl \
    --output generated_phase4_relations --package relations_model

PYTHONPATH=generated_phase4_university:generated_phase4_relations \
    python examples/verify_individuals.py
```

Use fresh output directories. Expected output:

```text
Phase 4 passed: individual loading, IRI identity, inherited types, literals, relations, annotations, and EQL
```

The generated samples are also bundled. After installation:

```bash
PYTHONPATH=examples/generated:examples/relations_generated \
    python examples/verify_individuals.py
```

## Modules, in build order

| Module | Responsibility |
| --- | --- |
| `runtime.py` | `FieldMapping`, `GeneratedModel`, `LoadedIndividuals` and the loading error type |
| `individuals.py` | Preflight asserted types, signatures, references and literals; construct objects; populate fields |
| `templates/runtime.py.jinja` | Export each generated package's mapping and `load_individuals` entry point |
| `generation.py`, package template | Render and export the runtime module |
| `examples/individuals.ttl`, `verify_individuals.py` | Demonstrate separate A-Box loading and loading individuals from a combined ontology |

The loader reuses RDFLib's graph and `Literal.toPython()`. It constructs existing
generated classes, uses native monitored collections and descriptors, and
queries results through KRROOD EQL. It introduces no new implication rules or
symbol graph implementation. The registry maps RDF identity to objects; it does
not replace KRROOD's object identity or global symbol caching.

## API

```python
from university_model import Student, load_individuals

loaded = load_individuals("examples/university.ttl")
alice = loaded["https://example.org/university#Alice"]
course = loaded["https://example.org/university#Course42"]

assert isinstance(alice, Student)
assert alice.name == ["Alice"]
assert alice.takes_course[0] is course
```

The source can be a string path, `Path`, or `OntologyInput` with an explicit RDF
format. It can contain the T-Box and A-Box together or only individual assertions
referring to the generated model's IRIs. Schema declarations in the input do not
recompile or change the generated model. Regenerate the model separately when
changing its schema.

`LoadedIndividuals` exposes:

- `by_iri`: the IRI-to-object registry; `loaded[iri]` provides direct lookup.
- `asserted_types`: the source's explicit class IRIs for each object.
- `annotations`: recognized annotation values retained as RDF terms, preserving
  language tags and datatype information.
- `asserted_graph`: the parsed source graph, without runtime entailments added.
- `source_sha256`: the input bytes' fingerprint.

Import all generated packages before constructing any symbols or the global
`SymbolGraph`, as required by Phase 3's descriptor registration lifecycle.
Generated packages now require this wrapper at runtime, in addition to KRROOD.

## Loading policy

1. Validate all recognized individual assertions before constructing objects.
2. Select the most specific asserted class when all asserted types form a
   generated inheritance chain.
3. Construct one instance per named individual IRI with empty defaults.
4. Resolve all object references from that registry, then add converted values
   to their generated fields. Native KRROOD relation implications execute here.

`owl:NamedIndividual` is optional when an explicit generated class type is
present. A reference target must itself be named and explicitly typed in the same
load. This supports forward, repeated and cyclic references without placeholder
objects. Subclass instances satisfy ancestor domain/range checks. Duplicate
assertions in RDF collapse through RDFLib; list assignment avoids duplicate
values and semantic sets use native uniqueness.

For example, an individual asserted as both `Person` and `Manager` becomes one
`Manager` object. An individual asserted as unrelated `Person` and `Organization`
is rejected. The inspected KRROOD `Role` implementation uses composition and
gives roles distinct Python identities; it does not make the taker an instance
of the role's class. Automatically mapping arbitrary OWL multiple types to
roles would change this loader's identity/type contract, so that mapping is
deferred explicitly.

The loader supports the nine declared Phase 2 XSD datatypes. Field literals
must have the exact declared datatype; plain strings are accepted for
`xsd:string`. RDFLib converts boolean, integer, decimal, floating point, date,
dateTime and time values. Invalid lexical values and language-tagged field
values fail. Language tags remain supported in annotation values. Declared
annotation properties and `rdfs:label`/`rdfs:comment` are retained outside fields.

Unknown asserted classes or individual properties, missing types, wrong
domains/ranges, unresolved references, anonymous individuals, blank-node
annotations and class/property punning receive `IndividualLoadingError` with
`code` and `iri`. `sameAs`, `differentFrom` and ontology imports are rejected.
Unattached annotation resources are not instantiated; only explicitly typed
individuals or subjects of mapped field assertions are individual candidates.

This is an explicit-type application loading profile. Domain/range checks
enforce representability in the chosen Python model; they do not implement OWL
type inference or establish logical inconsistency. There is no automatic class
reclassification, arbitrary multiple-inheritance synthesis, identity merging,
external reference fetching or cross-load object merging.

Each call creates a fresh registry and new objects. KRROOD's global graph can
retain objects from earlier loads; the loader does not clear it. Scope EQL to
the returned objects when isolation matters. Detected input errors fail during
preflight, before construction; unexpected constructor/runtime failures are not
transactional and have no rollback guarantee. Updates remain additive, with
retraction unsupported.

## Manifest and next milestone

Phase 4 manifests report `individual_loader_available: true`,
`individual_type_policy: "explicit_inheritance_chain"` and
`individuals_loaded: false`. The last flag describes compilation, not a later
successful call to the runtime loader.

Phase 5 will harden the compiler product with isolated import verification,
feature/conformance reporting and supplied local import resolution. Broader
multiple-type semantics, class expressions, classification, retraction and
round-trip export remain additional milestones toward near 1-to-1 translation.
