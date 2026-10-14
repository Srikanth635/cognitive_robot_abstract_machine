Phase 14 adds explicit ontology reasoner adapters: existing `owlrl` diagnostics and optional HermiT/OWLAPI consistency checking, with capabilities, source diagnostics and isolated execution. See [docs/phase-14.md](docs/phase-14.md) for installation and tests. The included AICOR input reports unsupported profile/parser findings; no ontology is rewritten or accepted.

Phase 13 adds isolated construction of actual generated dataclasses with KRROOD, shared-reference verification, native relation evidence and an asserted RDF round trip. See [docs/phase-13.md](docs/phase-13.md).

Phase 12 adds revision-bound candidate RDF and ontology-aware preflight through the existing individual planner. See [docs/phase-12.md](docs/phase-12.md).

Phase 11 adds immutable ontology-service contracts and a revision-aware candidate lifecycle. See [docs/phase-11.md](docs/phase-11.md). Execution adapters for candidate construction and reasoning follow in later phases.

Phase 10 adds declared-distinctness loading and source/current RDF export. See [docs/phase-10.md](docs/phase-10.md) for installation, generic round-trip tests, AICOR acceptance and limits.

Phase 9 adds opt-in positive definitions, shared native relation associations, scoped reflexivity, asymmetry checks and relation snapshots. See [docs/phase-9.md](docs/phase-9.md) for installation and testing.

Phase 8 adds explicit scoped validation. See [docs/phase-8.md](docs/phase-8.md)
for the API, installation, tests, native KRROOD reuse and limitations.

Phase 7 adds the opt-in generic `local-usage` structural profile. See
[docs/phase-7.md](docs/phase-7.md) for installation, tests, native reuse and limits.

# KRROOD ontology front end

**Phases 1–6 are delivered for the documented subset.** Generate dataclasses,
load named individuals, use native KRROOD relation implications and query positive
class expressions through existing EQL operators. Compilation verifies imports,
resolves supplied local ontology files and records provenance and feature
treatment. Query matches do not automatically reclassify Python objects.

This is a separate package importing KRROOD at CRAM commit
`236b295a23643477f0a7e9f1c318dd6d3948216d` (KRROOD `26.7.0`). It reuses
RDFLib parsing, KRROOD naming, `CodeGenerator`, formatting, `Symbol`, and
`PropertyDescriptor`, relation mixins, monitored sets and EQL. Ontology-specific
translation and output templates live
here. No upstream code was copied or modified.

## Directory structure

```text
krrood-ontology/
├── pyproject.toml
├── README.md
├── docs/
│   ├── phases.md
│   ├── phase-2.md
│   ├── phase-3.md
│   ├── phase-4.md
│   ├── phase-5.md
│   ├── phase-6.md
│   ├── upstream-api.md
│   └── validation.md
├── src/krrood_ontology/
│   ├── __init__.py
│   ├── __main__.py
│   ├── config.py
│   ├── ontology.py
│   ├── inspection.py
│   ├── backend.py
│   ├── bootstrap.py
│   ├── model.py
│   ├── translation.py
│   ├── semantics.py
│   ├── expressions.py
│   ├── queries.py
│   ├── individuals.py
│   ├── runtime.py
│   ├── imports.py
│   ├── features.py
│   ├── verification.py
│   ├── verification_worker.py
│   ├── generation.py
│   ├── compiler.py
│   ├── cli.py
│   └── templates/
│       ├── package.py.jinja
│       ├── vocabulary.py.jinja
│       ├── classes.py.jinja
│       ├── relations.py.jinja
│       ├── axioms.py.jinja
│       ├── runtime.py.jinja
│       ├── queries.py.jinja
│       └── model_package.py.jinja
├── examples/
│   ├── university.ttl
│   ├── verify_university.py
│   ├── relations.ttl
│   ├── verify_relations.py
│   ├── individuals.ttl
│   ├── verify_individuals.py
│   ├── imports/
│   ├── imports_generated/
│   ├── verify_imports.py
│   ├── expressions.ttl
│   ├── expressions_generated/
│   ├── verify_expressions.py
│   ├── generated/
│   └── relations_generated/
└── tests/
```

## Install

Requires Python 3.11+ and Git; validated on Python 3.12. The runtime imports
KRROOD modules that use `enum.StrEnum`, so Phase 2 raises the Python requirement
from the Phase 1 declaration. KRROOD's upstream dependency set is substantial.

```bash
cd krrood-ontology
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

If you already installed the same KRROOD baseline with all dependencies in the
active environment, update this wrapper without installing KRROOD again:

```bash
python -m pip install --no-deps -e .
```

The Git dependency is pinned to the inspected source commit; its transitive
versions are not locked. To intentionally reuse another KRROOD checkout,
install that checkout first, then use the `--no-deps` command and rerun the tests.

## Compile and test a simple case

Phase 6 compiles supported class expressions into metadata and scoped EQL
queries. Install this release in your existing environment, then regenerate:

```bash
python -m pip install --no-deps -e .
python -m krrood_ontology --version  # krrood-ontology 0.6.0
python -m krrood_ontology compile examples/expressions.ttl \
    --output generated_phase6 --package expressions_model
PYTHONPATH=generated_phase6 python examples/verify_expressions.py
```

Use the generated API with full class IRIs:

```python
from expressions_model import load_individuals, model_queries

loaded = load_individuals("examples/expressions.ttl")
workers = model_queries.match("https://example.org/expressions#WorkingPerson", loaded)
```

Intersections, unions, enumerations, object/datatype value restrictions and
selected object existential restrictions are supported under the precise
profile in [phase-6.md](docs/phase-6.md). Query matches read known objects and
native inferred relations; they do not create new types or class assertions.

Phase 5 adds explicit local import resolution. After installation:

```bash
python -m krrood_ontology --version  # krrood-ontology 0.6.0
python -m krrood_ontology compile examples/imports/university.ttl \
    --import https://example.org/core=examples/imports/core.ttl \
    --output generated_phase5 --package imported_model
PYTHONPATH=generated_phase5 python examples/verify_imports.py generated_phase5
```

This composes two local RDF files into one Python package. It fetches no imports
from the network. Output includes `features.json`, source snapshots and
`source/closure.ttl`, in addition to the model and manifest. Load the closure
through Phase 4 when individuals refer across the imported files. See
[phase-5.md](docs/phase-5.md) for verification and import policies.

For Phase 4, install this release in your existing KRROOD environment and use
fresh output directories:

```bash
python -m pip install --no-deps -e .
python -m krrood_ontology --version  # krrood-ontology 0.6.0
python -m krrood_ontology compile examples/university.ttl \
    --output generated_phase4_university --package university_model
python -m krrood_ontology compile examples/relations.ttl \
    --output generated_phase4_relations --package relations_model
PYTHONPATH=generated_phase4_university:generated_phase4_relations \
    python examples/verify_individuals.py
```

Or test the bundled generated packages:

```bash
PYTHONPATH=examples/generated:examples/relations_generated \
    python examples/verify_individuals.py
```

Use the generated loader directly:

```python
from university_model import load_individuals

loaded = load_individuals("examples/university.ttl")
alice = loaded["https://example.org/university#Alice"]
course = loaded["https://example.org/university#Course42"]
assert alice.name == ["Alice"]
assert alice.takes_course[0] is course
```

Import all generated packages before constructing KRROOD objects. A load creates
one object per named IRI. Multiple explicit types must form an inheritance
chain; unrelated types, identity axioms and unresolved references fail before
object construction. See [phase-4.md](docs/phase-4.md) for the API and limits.

For Phase 3, after installing this release in your active KRROOD environment:

```bash
python -m krrood_ontology --version  # krrood-ontology 0.6.0
python -m krrood_ontology compile examples/relations.ttl \
    --output generated_phase3 --package relations_model
PYTHONPATH=generated_phase3 python examples/verify_relations.py
```

Expected result:

```text
Phase 3 passed: inverse, transitive, symmetric, subproperty implications, and EQL queries
```

The same generated example is included under `examples/relations_generated/`.
Import your generated packages before constructing any KRROOD symbols or
`SymbolGraph`, then populate semantic relations after construction:

```python
from relations_model import Company, Person

alice = Person(name=["Alice"])
acme = Company()
alice.works_for.add(acme)
assert alice in acme.has_employee
```

Relations participating in supported axioms are sets with `init=False`.
Other Phase 2 object fields remain lists. Updates are additive; removal and
retraction are unsupported. See [phase-3.md](docs/phase-3.md) for the exact profile,
module responsibilities, lifecycle constraints and EQL examples.

The earlier structural example still works:

```bash
krrood-ontology compile examples/university.ttl \
    --output generated \
    --package university_model

PYTHONPATH=generated python examples/verify_university.py
```

The verification script prints:

```text
Phase 2 passed: dataclasses, inheritance, typed properties, and KRROOD object relations
```

The example includes Alice and Course42 assertions. Compilation inventories
them; its informational `individuals_deferred` warning means they have not been
loaded as objects during compilation. Call `load_individuals` to instantiate them.

Generated output:

```text
generated/
├── manifest.json
└── university_model/
    ├── __init__.py
    ├── vocabulary.py
    ├── classes.py
    ├── relations.py
    ├── axioms.py
    ├── runtime.py
    └── queries.py
```

The archive also includes this example under `examples/generated/`, so after
installation you can run the supplied output directly:

```bash
PYTHONPATH=examples/generated python examples/verify_university.py
```

Use the models in your application:

```python
from university_model import Person, Student, Course

alice = Student(name=["Alice"])
course = Course()
alice.takes_course.append(course)
assert isinstance(alice, Person)
assert list(alice.takes_course) == [course]
```

Set `PYTHONPATH=generated` when running that application, or add the generated
output directory to your application's import path. Names are collections in
this phase: use `name=["Alice"]`, not a scalar string.

## API

```python
from pathlib import Path
from krrood_ontology import CompilationOptions, OntologyCompiler, OntologyInput

result = OntologyCompiler().compile(
    OntologyInput(Path("ontology.owl")),
    CompilationOptions(Path("generated"), "my_model"),
)
print(result.package_directory)
```

RDF/XML, Turtle, N-Triples and N3 are accepted. `.owl` defaults to RDF/XML;
use `--format turtle` when an OWL file contains Turtle syntax. OWL Functional
Syntax and Manchester Syntax are not supported RDF serializations.

## Supported structural subset

- Explicitly declared named `owl:Class` / `rdfs:Class` terms.
- Acyclic single-parent named subclass relationships; `owl:Thing` as a parent
  becomes a generated root inheriting KRROOD `Symbol`.
- Explicitly declared object/data properties with exactly one named domain and
  one supported range. Object ranges must be declared classes.
- XSD string, boolean, integer, decimal, double, float, date, dateTime and time
  mapped to Python value types. See [phase-2.md](docs/phase-2.md).
- Collection-valued fields with independent defaults. Generated classes use
  `dataclass(eq=False, kw_only=True)` to preserve object identity and avoid
  inheritance ordering issues.
- Object properties attached through KRROOD descriptors and monitored
  containers. Datatype properties are ordinary dataclass fields because the
  existing descriptor requires a `Symbol` range.
- Full IRI mappings through `CLASS_BY_IRI`, `PROPERTY_FIELD_NAMES`,
  `PROPERTY_DESCRIPTORS`, class `__ontology_iri__`, and field metadata.

These are structural application models. Type hints are not runtime type or
OWL constraint validation. No complete OWL reasoner, automatic classification,
identity normalization or automatic network import loading is performed. The Phase 4
loader converts the supported literal datatypes through RDFLib.
Individual instantiation is explicit through the Phase 4 loader. Generated
packages require this wrapper and KRROOD at runtime.

For introspection, reuse `krrood.class_diagrams.utils.get_type_hints_of_object`.
It handles KRROOD's inherited annotations whose types are imported only under
`TYPE_CHECKING`; ordinary `typing.get_type_hints` can fail on those annotations.

## Unsupported mappings

The default compile command rejects detected unsupported OWL semantic features
such as unsupported class expressions, universal/cardinality restrictions and
functional properties. Selected anonymous equivalent-class expressions now
have positive scoped query mappings in Phase 6. Ontology
imports require explicit local mappings in Phase 5. Supported
inverse/transitive/symmetric/subproperty axioms are now
translated to native KRROOD behavior. Unsupported combinations fail with a
diagnostic; see the Phase 3 profile. You can explicitly omit all semantics:

```bash
krrood-ontology compile ontology.owl --output generated --package my_model \
    --structural-only
```

The CLI prints each omission and records it in `manifest.json`. Ambiguous or
invalid structural mappings still fail: multiple parents, anonymous parents,
multiple/missing domains or ranges, undeclared signature entities, unsupported
datatypes, invalid identifiers, and naming collisions. Imported ontologies are
never fetched from the network. Annotation declarations remain in the vocabulary.

Compilation requires a fresh destination. On ordinary generation failures,
temporary output is cleaned up. Individual source assertions and unsupported
axioms are not embedded as Python assignments. Source snapshots and the
asserted runtime closure preserve RDF for later loading. The manifest records
fingerprints, mapping choices, imports, verification and known omissions.
Exporting later object mutations back to RDF remains future work.

## Modules and verification

The modules are documented in [phase-2.md](docs/phase-2.md),
[phase-3.md](docs/phase-3.md), [phase-4.md](docs/phase-4.md) and
[phase-5.md](docs/phase-5.md) and [phase-6.md](docs/phase-6.md). The earlier
loader, inspector, backend, bootstrap, and CLI remain in the same package.

```bash
python -m pytest -q
```

Tests exercise actual generated classes in isolated Python processes, including
KRROOD object relation tracking, inheritance, forward references, all supported
datatypes, relation implications, EQL queries over inferred relations,
individual loading, literal conversion, IRI identity, annotations,
diagnostics, collision detection, reproducible output, and failure
cleanup. See [validation.md](docs/validation.md) for the completed checks.

The optional Phase 1 commands are unchanged:

```bash
krrood-ontology inspect examples/university.ttl
krrood-ontology bootstrap examples/university.ttl --output vocabulary_output \
    --package university_vocabulary
```
