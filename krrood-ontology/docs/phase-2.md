# Phase 2: ontology to dataclass models

This document records the Phase 2 milestone. Release 0.3.0 extends its relation
behavior with the bounded native profile described in [phase-3.md](phase-3.md).

The accomplishment is an importable package containing real domain dataclasses.
The university example now supports `Student(name=["Alice"])` and
`student.takes_course.append(Course())`. Object-property tracking delegates to
KRROOD; this phase does not generate new relation storage or inference logic.

## New modules, in build order

| Module | Responsibility |
| --- | --- |
| `model.py` | Minimal dataclasses describing class names, parent decisions, properties and diagnostics |
| `translation.py` | Read the asserted graph and resolve the supported structural mappings |
| `generation.py` | Feed model decisions to new templates through the existing KRROOD backend |
| `compiler.py` | Load, inspect, translate, render and publish a fresh output directory |

`config.py` adds `CompilationOptions`, which extends the earlier output options.
`backend.py` exposes a shared single-file rendering method. `cli.py` adds
`compile` and reports semantic omissions. The Phase 1 APIs remain available.

New templates:

- `classes.py.jinja`: dataclasses, field metadata, IRI registries.
- `relations.py.jinja`: KRROOD descriptor subclasses and field registration.
- `model_package.py.jinja`: model exports and registration during package import.

## Reused mechanisms

| Need | Existing implementation |
| --- | --- |
| RDF parsing and graph access | RDFLib, already in KRROOD's dependencies |
| Class/field case conversion | `krrood.code_generation.naming` |
| Template rendering | `krrood.code_generation.generator.CodeGenerator` |
| Formatting | `krrood.code_generation.formatting.run_black_on_file` |
| Symbolic instances | `krrood.entity_query_language.predicate.Symbol` |
| Object-property monitoring and graph relations | `krrood.ontomatic.property_descriptor.property_descriptor.PropertyDescriptor` |
| Runtime annotation resolution | `krrood.class_diagrams.utils.get_type_hints_of_object` |
| Inheritance ordering | Python `graphlib.TopologicalSorter` |

The existing property descriptor asserts that the range is a subclass of
`Symbol`. Consequently, attaching it to `list[str]` or `list[int]` would be an
invalid use of the upstream API. Literal properties remain ordinary dataclass
fields. No alternative literal descriptor was invented.

## Structural mapping

| Ontology construct | Generated representation |
| --- | --- |
| Explicit named class | `dataclass(eq=False, kw_only=True)` inheriting `Symbol` |
| One named superclass | Python inheritance |
| `owl:Thing` superclass | Root inheriting `Symbol` |
| Object property with one named domain/range | `list[RangeClass]`, with a KRROOD descriptor |
| Datatype property with one named domain/XSD range | `list[PythonType]` field |
| Class IRI | `__ontology_iri__` and `CLASS_BY_IRI` |
| Property IRI | Field metadata and `PROPERTY_FIELD_NAMES`; object descriptors also appear in `PROPERTY_DESCRIPTORS` |
| Individual assertion | Vocabulary inventory and an explicit deferred diagnostic |

Properties use collections even if a richer ontology might imply functional
behavior. Functional/cardinality semantics are not interpreted in this phase;
detected such features are errors unless the user explicitly chooses
`--structural-only`. Field type annotations do not enforce values at runtime.

Domain/range axioms are used to select the application field owner and type.
The generated field does not implement OWL domain/range inference or retype an
object that receives a value. Missing values do not imply a violated OWL axiom.

| XSD range | Python type |
| --- | --- |
| `string` | `str` |
| `boolean` | `bool` |
| `integer` | `int` |
| `decimal` | `decimal.Decimal` |
| `double`, `float` | `float` |
| `date` | `datetime.date` |
| `dateTime` | `datetime.datetime` |
| `time` | `datetime.time` |

Python numeric/date types are application types, not an implementation of all
XSD lexical forms, precision, timezone rules or value spaces. Other datatype
IRIs are rejected until there is a documented mapping.

## Naming and rejected cases

Names come from IRI fragments or final path components, then KRROOD's naming
helpers. URN final components are supported. Python keyword fields receive a
trailing underscore (`class` becomes `class_`). Full IRIs remain authoritative.
Hyphen normalization is performed by KRROOD itself.

Generation rejects:

- Distinct class IRIs that produce the same exported name.
- Distinct properties that collide within one class or its inheritance chain.
- Public identifiers that remain invalid or begin with an underscore.
- Cyclic or multiple inheritance, anonymous parents, and undeclared named parents.
- Missing or multiple property domains/ranges, unsupported datatypes, and
  undeclared property signatures.
- Object/data property kind conflicts.

`--structural-only` permits recorded semantic omissions, not malformed or
ambiguous structural mappings. Without it, detected unsupported OWL features
and RDFS subproperty axioms fail before generation. The detector covers OWL
predicates/types inventoried by Phase 1 and the specific additional structural
checks; it is not a full OWL profile validator.

## Test the delivered output

```bash
krrood-ontology compile examples/university.ttl --output generated \
    --package university_model
PYTHONPATH=generated python examples/verify_university.py
```

The supplied verifier checks dataclass status, `Student` inheritance, field
types and metadata, independent defaults, descriptor registration, and a graph
relation created by appending a course. Additional company-model tests exercise
three-level inheritance, inherited descriptors, references between classes,
keyword names, and every supported datatype.

Phase 3 adds only verified semantic behavior using the existing inverse and
transitive mixins/EQL. Phase 4 loads individuals. Neither capability is claimed
by this structural milestone.
