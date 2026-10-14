# Phase 6 validation

Validated on Python 3.12.14 against the real KRROOD source checkout at
`236b295a23643477f0a7e9f1c318dd6d3948216d`.

- Wrapper version: `0.6.0`; installed KRROOD version: `26.7.0`.
- Full pytest run: **96 passed** (including all 76 earlier tests).
- Expression/translation checks after final metadata/export changes: **35 passed**.
- Source and test Ruff checks passed; Python files formatted with Black.
- Generated university modules passed Ruff's functional (`F`) checks.
- Wheel build succeeded and included all eight Jinja templates.
- The wheel was installed into a separate target directory, and its CLI
  generated a new relation model. The independent verification script passed.
- Fresh university and relation samples passed their verification scripts.
- The combined Phase 4 script passed for both freshly generated and bundled
  samples, loading university individuals and a separate relation A-Box.
- The wheel's module path and installed version were checked in the separate
  target directory, avoiding an accidental import of the editable source.
- The installed wheel resolved the supplied local-import example, ran isolated
  verification and produced a package whose closure passed runtime loading.
- The installed Phase 6 wheel generated the expression model, verified its
  query registration in isolation and passed the complete expression example.
  Its import origin, version and eight packaged templates were checked.

Generated models were exercised in isolated processes for:

1. The university ontology: dataclasses, inherited literal field, typed object
   field, descriptor registration, independent defaults, field metadata and
   KRROOD graph relation creation.
2. The company ontology: three-level inheritance, inherited descriptors,
   mutually referring classes, keyword field names and every supported XSD
   datatype mapping.
3. A class-only ontology: class construction and empty property registries.
4. The Phase 3 relation ontology: bidirectional inverse updates, multi-level
   subproperties, symmetric implications, transitive chains in both insertion
   orders, inverse-side updates, inherited fields, duplicate avoidance and EQL
   queries over inferred relationships. Native graph edges were also checked
   for their inferred flag.
5. Phase 4 loading: one object per IRI, shared forward references, explicit
   inheritance-chain types, converted literals, RDF annotations, inferred
   relationships, EQL results, unchanged asserted source graph and independent
   registries across repeated loads. All nine supported datatype conversions
   were checked for both value and Python type.
6. Phase 5: generated-artifact import origin, class/field metadata, descriptor
   registration and strict individual preflight in an isolated interpreter;
   local import merging and runtime loading; byte-for-byte source snapshots;
   missing/duplicate/unused import mappings; cyclic import termination;
   structural-only omission; worker failures/timeouts and publication cleanup.
   A closure round-trip additionally verified Unicode line separators and
  embedded newlines in literal values.
7. Phase 6: generated class-expression metadata, intersections, unions,
   enumerations, object/literal hasValue, object existentials, nested
   existentials and references to other definitions. Queries used heterogeneous
   scopes, inferred inverse relations, explicit type membership, narrowed/empty
   candidate domains and isolated registries. Missing named expression values
   and foreign candidates failed explicitly; source RDF and object types stayed
   unchanged. Reciprocal/reverse equivalent-class assertions normalized.

Negative and orchestration checks covered inheritance cycles/multiple parents,
missing/multiple domains or ranges, undeclared signature entities, unsupported
XSD types, invalid identifiers, class/field name collisions, property-kind
conflicts, semantic feature rejection and explicit omission reporting,
reproducible generation in the same environment, preservation of existing
output, cleanup after generation failure, and CLI translation errors.
Phase 3 negative checks additionally covered incompatible inverse signatures,
unknown/multiple inverses, cyclic/multiple superproperties, semantic declarations
on datatype properties, signature mismatches and transitive superproperty
rejection. Structural-only mode explicitly omits the relation axioms.
Phase 4 checks reject missing/unknown/unrelated types, unresolved references,
wrong domains/ranges, invalid or mismatched literals, language-tagged field
values, unmapped individual properties, anonymous individuals, punning and
identity/import axioms. A failing load was tested with constructors instrumented
to establish that preflight completes before object construction.

Phase 6 negatives cover cyclic/malformed RDF lists, anonymous expression cycles,
recursive named definitions, unknown classes/properties, invalid literal kinds,
unsupported datatype existentials, conflicting operators/definitions, empty
enumerations, named equivalence, extra predicates and unattached nodes.

The earlier Phase 1–5 tests still pass. Tests use real KRROOD generators,
formatters and runtime descriptors. Ordinary parser/backend failures remain
exceptions; compilation's ontology translation errors get actionable CLI
messages.

This establishes the documented structural subset and bounded additive native
relation profile. Complete OWL entailment, runtime value validation, retraction,
complete OWL profile conformance, arbitrary multiple-type loading,
round-trip conversion and cross-version KRROOD compatibility are not established
by these tests. Python 3.11+ is declared because the KRROOD runtime uses
`enum.StrEnum`; only Python 3.12 was exercised here.
