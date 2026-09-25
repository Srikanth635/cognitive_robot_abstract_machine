# Integrating `llm_reasoner` into the monorepo

This tree mirrors the repository layout. Copy each path to the matching location in
`cognitive_robot_abstract_machine` and apply the two patches.

```
llm_reasoner/                                     → workspace member, sibling of krrood/
coraplex/src/coraplex/querying/language_model.py  → new module
coraplex/doc/language_model_backend.ipynb         → new notebook
test/llm_reasoner_test/                           → new test package
test/coraplex_test/test_language_model_backend.py → new test module
root_pyproject.patch                              → apply to ./pyproject.toml
coraplex_requirements.patch                       → apply to ./coraplex/requirements.txt
```

Every path under `llm_reasoner/` replaces the file you already have at the same path, except
`requirements.txt`, `_version.py` and `doc/` which are new. `doc/language_model_backend.ipynb`
is your kitchen notebook, relocated from `notebooks/` and updated for the renamed
`Exchange.raw_response` field.

---

## Dependency direction

```
krrood  ←  llm_reasoner  ←  coraplex.querying.language_model  →  semantic_digital_twin
```

`llm_reasoner` imports **only** `krrood`, `typing_extensions` and the standard library, which
keeps it usable by anything in the workspace and testable without a world. Everything that knows
what a robot is lives in `coraplex.querying.language_model` — the same package that already holds
`querying/predicates.py`, which is the existing precedent for coraplex-owned krrood extensions.

The direction matters more than the location. Putting the world-aware seams inside `llm_reasoner`
would make every consumer of a language model backend depend on `semantic_digital_twin`, which is
the coupling the design exists to avoid.

---

## What each piece contributes

| module | what it represents | what it enables |
|---|---|---|
| `llm_reasoner.reflection` | a match as flat, typed, path-named slots, each classified by *what kind of knowledge is missing* | deciding what may be generated versus only selected |
| `llm_reasoner.prompts` | schema, constraints, situation as three separable layers | swapping the world-aware layer without touching the rest |
| `llm_reasoner.backend` | the resolution procedure | generation, decoding, write-back, enforcement, provenance |
| `llm_reasoner.context` | the two injection seams | world knowledge without world imports |
| `coraplex.querying.language_model` | world state as prompt text and candidates as descriptions | choosing between alternatives the schema cannot distinguish |

### Slot kinds encode an epistemic distinction

| kind | what is missing | what is allowed |
|---|---|---|
| `VALUE` | a self-identifying value | generation |
| `REFERENT` | an individual | selection from an extension, never invention |
| `BOUND` | nothing | context for the other slots |

A model can produce the string `"the left gripper"`; it cannot produce an `EndEffector` with a
tool frame and a place in the kinematic structure. `end_effector=...` therefore raises
`ReferentSlotHasNoDomain`, and the fix is `variable(EndEffector, world.semantic_annotations)`.

---

## Verification status

Be clear about what was and was not executed.

**Ran, 20 tests passing.** `test/llm_reasoner_test` against a real `krrood` install: reflection
including nested designators, symbol-graph extensions, prompt assembly, enum and token decoding,
token-to-individual round trip, single-completion batching, constraint enforcement, repair loop,
budget exhaustion, both referent refusals, provenance log.

**Not run: `test/coraplex_test/test_language_model_backend.py`.** `semantic_digital_twin` and
`coraplex` are not installable in the environment this was written in. The module is
syntax-checked and every SDT and coraplex symbol it uses was verified against the repository
source (`World.get_semantic_annotations_by_type`, `HasRootBody.root`, `WorldEntity.name`,
`KinematicStructureEntity.global_pose`, `GripperIsFree`, `NaturalLanguageWithTypeDescription`,
`GraspDescription`). It binds to the existing `pr2_world_copy` fixture in `test/conftest.py`.
Expect fixture-level adjustment on first run — particularly whether that world carries
`NaturalLanguageWithTypeDescription` annotations, which only the SAGE-10k and RoboCasa loaders
populate.

The notebook has a `world = ...` placeholder for the same reason; wire it to your usual setup.

---

## Conventions applied

Rebuilt against `AGENTS.md`: no `try`/`except` outside the JSON parsing boundary (where a
`JSONDecodeError` is converted into `ModelResponseUnparseable`), no `getattr`, absolute global
imports, `from __future__ import annotations`, `# %%` section headers, dataclasses throughout,
RST docstrings with every field documented below it, no abbreviations in identifiers, guard
clauses over nesting, custom exceptions rather than illegal states.

Test mimics follow the krrood convention: `ManipulatorMimic` and `GraspMimic` are named after the
pattern they exercise, not after `EndEffector` and `GraspDescription`, and they import nothing
outside `krrood`.

Run `scripts/format_docstrings.py` on the new files before committing.

---

## Two things to fix upstream

**1. `Match.where` bound through `.expression` silently matches nothing.**

```python
match.where(match.variable.manipulation_offset > 0.0)    # 40 results
match.where(match.expression.manipulation_offset > 0.0)  #  0 results
```

Measured with the shipped `EntityQueryLanguageGenerativeBackend`, so this is not specific to this
package. No exception, no warning; the condition verbalizes correctly and returns an empty set.
`Match.where` on generative matches appears unexercised anywhere in the workspace. Worth a failing
test in `krrood` before a fix.

**2. `krrood.entity_query_language.backends` drags in `probabilistic_model`.**

`backends.py` imports `krrood.parametrization.model_registries`, which imports
`probabilistic_model.probabilistic_circuit.relational.rspn`. Consequences:

- `GenerativeBackend` cannot be imported without a probabilistic-circuit stack, so a language
  model backend needs a dependency it never uses.
- `rspn` exists in no released `probabilistic_model`, so `krrood` is not installable from
  published dependencies at all.

Importing `rspn` lazily inside `RelationalCircuitRegistry`, or moving `ProbabilisticBackend` into
`krrood.parametrization`, would fix both.

Related version constraint, since it cost some time: `random_events_lib.reals` first appears in
`random_events` 4.1.2, but `krrood` needs `compatible_types` from `random_events.variable`, which
only exists from 5.0.1 — and published `probabilistic_model` 7.1.1 cannot work with 5.x because it
imports `SubclassJSONSerializer` from `random_events.utils`, which 5.x dropped.

**Minor:** `prompts.py` and `reflection.py` use `_attribute_docstrings` from
`krrood.entity_query_language.verbalization.rendering.source_documentation`, which is private.
Promoting it to public API would remove the only private-symbol dependency here.

---

## Next

Phase 2 is the referent pipeline: replace `ReferentSlotHasNoDomain` with describe → ground →
select. The model proposes a sortal description, that description is compiled into EQL conditions
and evaluated against the world by a selective backend, and only the survivors are offered for
selection. The model supplies the intensional half of anchoring; EQL supplies the extensional
half.

After that, composition. A `ParameterResolver` seam of shape `Match → Match` would let this
backend resolve the referent and hand the geometry to `ProbabilisticBackend`, which treats the
chosen object as conditioning evidence via `to_dao` and `FeatureExtractor`. The phases inside
`_evaluate` are already split along that line. Note that the rewrite primitive needs care:
`copy.deepcopy` on a `Match` drops `_where_conditions_`, `_quantifier_type_`, `domain` and
`_limit_`, and `Variable` has no `__deepcopy__`, so a domain of live world objects would be cloned.
