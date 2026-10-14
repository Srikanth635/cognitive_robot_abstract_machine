# Verified KRROOD baseline

Repository: [cram2/cognitive_robot_abstract_machine](https://github.com/cram2/cognitive_robot_abstract_machine)

Inspected commit: `236b295a23643477f0a7e9f1c318dd6d3948216d`.
Source version: `26.07.0` (`26.7.0` as installed).

The audit used a local checkout, source reads, source searches, and real imports.
It did not infer code availability from a paper or architecture description.
No upstream source was modified or bundled in this deliverable.

## Used by Phase 1

| Import | Verified call / behavior |
| --- | --- |
| `krrood.code_generation.generator.CodeGenerator` | `CodeGenerator(template_directory=str(path))`; `render(template_name, **context)` |
| `krrood.code_generation.formatting.run_black_on_file` | Formats through Black in the active Python interpreter |
| `rdflib.Graph` | `Graph().parse(data=content, format=syntax, publicID=file_uri)`; RDFLib is an upstream dependency, not an Ontomatic importer |

The wrapper uses its own output templates and adds a `repr` filter to the
generator's existing Jinja environment to escape Python string values. It does
not replace KRROOD's Jinja loading, rendering, or formatting implementation.

## Used by Phase 2

Phase 2 additionally imports the naming helpers, `Symbol`, and
`PropertyDescriptor`. The generated object fields use the upstream monitored
containers and relation graph. Literal fields do not receive descriptors,
because the current descriptor requires a `Symbol` range.

The verification scripts also use
`krrood.class_diagrams.utils.get_type_hints_of_object`, the existing resolver for
inherited annotations imported under `TYPE_CHECKING`. This avoids implementing
a separate annotation resolver. Phase 3 now uses inverse/transitive mixins,
descriptor inheritance and the existing EQL factories; `Role` remains for the
A-Box milestone.

Source links pinned to the inspected commit:

- [Generator](https://github.com/cram2/cognitive_robot_abstract_machine/blob/236b295a23643477f0a7e9f1c318dd6d3948216d/krrood/src/krrood/code_generation/generator.py)
- [Formatting](https://github.com/cram2/cognitive_robot_abstract_machine/blob/236b295a23643477f0a7e9f1c318dd6d3948216d/krrood/src/krrood/code_generation/formatting.py)
- [Dependencies and packaged templates](https://github.com/cram2/cognitive_robot_abstract_machine/blob/236b295a23643477f0a7e9f1c318dd6d3948216d/krrood/pyproject.toml)

## Source components relevant to later phases

| Import | Observed purpose | Status in our package |
| --- | --- | --- |
| `krrood.code_generation.naming.to_snake_case`, `to_camel_case` | Identifier case conversion | Used in Phase 2; collision/validity policy remains our responsibility |
| `krrood.entity_query_language.predicate.Symbol` | Base symbolic object abstraction | Used by Phase 2 generated classes |
| `krrood.ontomatic.property_descriptor.property_descriptor.PropertyDescriptor` | Descriptor metadata, storage, introspection, relation integration | Used by Phase 2 object properties; not an OWL parser |
| `krrood.ontomatic.property_descriptor.mixins.HasInverseProperty` | Abstract `get_inverse()` relation contract | Used for Phase 3 inverse and symmetric properties |
| `krrood.ontomatic.property_descriptor.mixins.TransitiveProperty` | Transitive descriptor marker | Used for Phase 3 transitive properties |
| `krrood.entity_query_language.factories.variable`, `entity`, `contains` | Query explicit object domains and collection membership | Used by Phase 3 verification over inferred relations |
| `krrood.patterns.role.Role` | Composed typed views with distinct Python identities | Inspected for Phase 4; automatic OWL multiple-type mapping deferred because identities and membership semantics differ |

Existing examples/tests:

- `test/krrood_test/test_ontomatic/test_property_descriptor.py`
- `test/krrood_test/dataset/role_and_ontology/university_ontology_like_classes.py`
- `test/krrood_test/test_patterns/test_role.py`
- `test/krrood_test/test_code_generation/test_generator.py`

Real imports of `Symbol`, `PropertyDescriptor`, `HasInverseProperty`,
`TransitiveProperty`, and `Role` succeeded in the validation environment. Their
availability alone does not establish support for translating OWL constructs.
Phase 3 integration tests now establish the bounded relation profile described
in `phase-3.md`; other mappings remain future work.

The example source defines dataclasses inheriting `Symbol`, then declares
descriptor subclasses and attaches descriptor instances to fields after the
classes are defined. This is the pattern to study for Phase 2/3 rather than
writing new inverse tracking or monitored collections.

## Gaps established by this checkout

The Ontomatic source directory contains `property_descriptor/`, `failures.py`,
and an empty package initializer. Source searches did not find an OWL loader,
T-Box converter, A-Box converter, or end-to-end ontology compiler in KRROOD's
current source/tests. The dependency list includes RDFLib and OWLRL, but a
dependency declaration is not an implemented conversion API.

`krrood.class_diagrams.module_generation` contains `DataclassRenderer` and
`ModuleRenderer`. The former explicitly notes that fields are not rendered yet;
the latter refers to `../jinja_templates/python_module.py.jinja`. The current
checkout contains that template. Its installation packaging would need checking
because the location is omitted from the explicit template package-data list.
More importantly, the renderer's documented field limitation means it is not
adopted as a complete field-capable ontology backend. The verified
`CodeGenerator` is the reusable rendering core.

Necessary new code is limited to ontology-specific translation decisions,
output templates, orchestration, feature diagnostics, and verification. RDF
parsing, symbolic object behavior, descriptors, roles, and source generation
should stay with their existing implementations where they meet the contract.

## Used by Phase 4

Phase 4 uses RDFLib `Literal.toPython()` and its `ill_typed` flag for conversion
of the nine supported XSD types. It instantiates generated dataclasses and
populates their ordinary lists or KRROOD monitored collections after complete
preflight. The source graph remains asserted RDF; native graph implications do
not modify it. The new IRI registry owns only the RDF-to-object lookup for a
single load.

The inspected `Role` contract specifies distinct role/taker identities and
composition-based membership, rather than Python `isinstance` membership in
the role class. Phase 4 therefore supports multiple explicit types only along
the generated inheritance chain and does not synthesize roles for unrelated
OWL types.

## Used by Phase 5

Phase 5 continues to use the existing RDFLib loader for explicitly mapped local
files. It reuses `rdflib.compare.to_canonical_graph` and RDFLib's N-Triples
serializer for stable closure snapshots. Isolated verification uses Python
subprocess/import/dataclass APIs and the Phase 4 planner; it implements no new
symbolic object or inference behavior. Feature reporting reuses the asserted
graph and existing diagnostic records.

## Used by Phase 6

Positive expression queries directly reuse KRROOD's `and_`, `or_`, `contains`,
`in_`, `exists`, `flat_variable`, `variable`, `entity`, `type_` and `issubclass_`.
The field reader uses the existing `symbolic_function` decorator to expose
dataclass collections safely across heterogeneous candidate domains. No new
logical evaluator or implication engine was introduced. Nested existential
queries and explicit registry isolation are established by generated-model
integration tests. Supported expression metadata is emitted through the same
KRROOD backend.
