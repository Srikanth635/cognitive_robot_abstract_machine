# Phase 6: positive class expressions as EQL queries

Phase 6 preserves supported anonymous class expressions as typed metadata and
builds queries from KRROOD's existing logical operators, type predicates,
membership tests and existential quantifiers. Matching identifies known objects;
it does not change their Python classes or materialize new membership assertions.

## Install, generate and test

Use your existing KRROOD environment and a fresh output directory:

```bash
cd krrood-ontology
python -m pip install --no-deps -e .
python -m krrood_ontology --version
# krrood-ontology 0.6.0

python -m krrood_ontology compile examples/expressions.ttl \
    --output generated_phase6 --package expressions_model
PYTHONPATH=generated_phase6 python examples/verify_expressions.py
```

Expected result:

```text
Phase 6 passed: intersections, unions, enumerations, value restrictions, existentials, nested queries, and scoped identity
```

The freshly generated sample is also included:

```bash
PYTHONPATH=examples/expressions_generated python examples/verify_expressions.py
```

## Modules in build order

| Module | Responsibility | Reused mechanism |
| --- | --- | --- |
| `model.py` additions | Immutable `ClassExpression`, `ExpressionKind`, literal metadata and named definitions | Existing intermediate model and manifest serialization |
| `expressions.py` | Validate expression shape, RDF lists, named references and cycles | RDFLib graph and list traversal; existing diagnostics; Python topological sorting |
| `queries.py` | Assemble scoped EQL conditions and evaluate them | KRROOD `variable`, `entity`, `and_`, `or_`, `contains`, `in_`, `exists`, `flat_variable`, `type_`, `issubclass_`, `symbolic_function` |
| `templates/queries.py.jinja` | Emit immutable expression metadata and a model-specific query entry point | Existing KRROOD code generator and formatter |
| Translator, generator, verifier, feature reporter | Integrate supported features, exports and isolated verification | Existing Phase 5 pipeline |

The only new symbolic adapter is a field reader wrapped in KRROOD's
`symbolic_function`. It returns the mapped field's values when the object belongs
to its structural domain, otherwise an empty tuple. This allows heterogeneous
registry scopes without attribute lookup errors. Logical composition and
existential evaluation remain in KRROOD; no separate rule or query engine was
added.

## Supported syntax and mapping

The structural Phase 2 requirements and additive Phase 3 relation profile still
apply. A declared named class may have one anonymous `owl:equivalentClass`
definition. Either RDF triple direction is accepted, and redundant reciprocal
assertions normalize to one definition. Distinct multiple definitions fail.

| Expression | Accepted shape | Query mapping |
| --- | --- | --- |
| Named class operand | Explicitly declared class IRI | Structural subtype predicate, plus its supported definition if present |
| `owl:intersectionOf` | Well-formed list of at least two supported operands | EQL `and_` |
| `owl:unionOf` | Well-formed list of at least two supported operands | EQL `or_` |
| `owl:oneOf` | Nonempty list of named individual IRIs | Registry object identity and EQL `in_` |
| Object `owl:hasValue` | Explicit `owl:Restriction`, one mapped object property, named individual value | EQL `contains` with the registry object |
| Datatype `owl:hasValue` | Explicit restriction, one mapped datatype property, valid literal with its exact range datatype | RDFLib value conversion and EQL `contains` |
| `owl:someValuesFrom` | Explicit restriction, mapped object property, supported class-expression filler | EQL `flat_variable` and correlated `exists` |

Nested expressions and acyclic references to other defined classes are
supported. Datatype existential restrictions, named-to-named equivalence,
anonymous superclass mappings, negation/complements, universal restrictions,
cardinality, identity merging and recursive definitions remain deferred.

RDF lists must have exactly one `rdf:first` and `rdf:rest` per blank-node cell,
terminate at `rdf:nil`, and be acyclic. Expression nodes must belong to a
compiled definition; malformed, conflicting, unsupported or unattached nodes
fail before generation. Anonymous nodes accept only their supported operator,
appropriate type/onProperty declaration and equivalent-class links; other
anonymous-node predicates currently fail explicitly. All imports must be
resolved through Phase 5 before expression translation.

`--structural-only` omits expression semantics, emits empty `CLASS_EXPRESSIONS`
and records the excluded features. Class dataclasses still exist as structural
types. Ordinary named-class queries remain available against those types.

## Example and API

```turtle
:WorkingPerson a owl:Class ;
    rdfs:subClassOf :Person ;
    owl:equivalentClass [
        owl:intersectionOf (
            :Person
            [ a owl:Restriction ;
              owl:onProperty :worksFor ;
              owl:someValuesFrom :Company ]
        )
    ] .
```

The named subclass statement supplies normal structural inheritance. The
equivalent-class expression supplies a separate query definition; its anonymous
intersection does not automatically become multiple Python bases.

```python
from expressions_model import WorkingPerson, load_individuals, model_queries

base = "https://example.org/expressions#"
loaded = load_individuals("examples/expressions.ttl")
alice = loaded[base + "Alice"]
workers = model_queries.match(base + "WorkingPerson", loaded)

assert alice in workers
assert not isinstance(alice, WorkingPerson)
```

`match(class_iri, loaded, candidates=None)` returns a list of matching objects.
`query(...)` returns a native KRROOD EQL `Query` for evaluation through its normal
API. The generated package exports `CLASS_EXPRESSIONS`, keyed by the full named
class IRI, and `model_queries`. Metadata contains typed immutable trees and
datatype-qualified literal values; source snapshots retain original RDF.

Use `candidates` to narrow the returned objects while keeping the whole supplied
registry available for witnesses:

```python
query = model_queries.query(
    base + "WorkingPerson", loaded,
    candidates=[loaded[base + "Alice"], loaded[base + "Charlie"]],
)
assert set(query.evaluate()) == {loaded[base + "Alice"]}
```

Candidates must belong to that registry. An empty candidate list produces no
matches. All existential witnesses must also occur in the supplied registry;
objects retained in KRROOD's global graph or referenced outside this registry do
not silently enlarge the scope. Named values in `hasValue` and `oneOf` must
resolve in the registry, otherwise query construction raises an actionable
`ValueError`. Separate loads resolve their own objects even when their IRIs are
identical.

## Meaning of a match

For a named class, matching accepts either an object's structural/explicit type
or a positive witness for its supported definition. An explicitly loaded
`WorkingPerson` therefore matches even when no company assertion was loaded.
The query does not invent the existential witness that OWL may imply. Conversely,
an ordinary `Person` with a known company relation can match `WorkingPerson`
without becoming an instance of that Python class.

Queries read both asserted and natively inferred field relations. The example's
employer queries use inverse relationships populated by Phase 3 descriptors.
Boolean/value tests use KRROOD and Python value equality on the supported
converted datatypes. This is a finite positive known-object profile, not complete
OWL datatype or open-world reasoning. No match means no supported positive
witness in the supplied scope; it is not a proof of nonmembership. Validation
of necessary restrictions remains a separate future profile.

Manifest flags record `class_expression_queries`,
`expression_query_policy: "positive_known_objects_explicit_registry"` and
`automatic_reclassification: false`. Feature reports identify supported
expression predicates as query mappings. Phase 5 isolated verification checks
the generated metadata/query registration and still preflights source
individuals without constructing them during compilation.

## Verification and next step

Tests exercise all five expression forms, mixed object domains, nested
existentials, references to other definitions, inferred inverse relations,
explicit type matches, candidate restriction, empty domains, missing value
IRIs, registry isolation and unchanged asserted RDF. Negative cases cover
malformed/cyclic lists, anonymous/named expression cycles, unknown classes or
properties, invalid literals, conflicting operators/definitions, unsupported
restrictions, named equivalence and unattached nodes.

The next improvement is Phase 7: an explicitly scoped validation profile using
KRROOD EQL for universal restrictions, selected cardinalities and disjointness.
It should remain separate from OWL entailment and from automatic classification.
