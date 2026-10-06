# `llm_reasoner`

`llm_reasoner` resolves underspecified requests and maintains reusable,
description-level knowledge that can guide later resolutions. Its knowledge layer is
deliberately independent of a particular robot, world, or episode: a hypothesis says
what a kind of action normally requires, while the resolver binds that description to
the current world.

## Package architecture

The main responsibilities are kept separate:

| Area | Responsibility |
|---|---|
| `clients` | Language-model client interfaces and recorded exchanges |
| `resolver` | Reflection, prompting, decoding, and request resolution |
| `knowledge/generation` | Capability-selected generation, schema derivation, repair, and admission |
| `knowledge/hypothesis` | Provenance, status, storage, cache migration, and graph scope |
| `knowledge/ontological` | Registration, metadata, and reusable validation primitives |
| `knowledge/microworlds` | Versioned domain vocabularies and their acceptance constraints |
| `knowledge/query.py` | Read-only, provenance-carrying questions over admitted hypotheses |

Dependencies point from applications toward these abstractions. A microworld does not
import viewer or world-model packages, and the generic generation and hypothesis layers
do not contain action-specific concepts.

## The Flanagan microworld

`knowledge/microworlds/flanagan` represents the predictive organization of object
manipulation described by Flanagan, Bowman, and Johansson. The scientific authority,
claim statuses, semantic distinctions, and claim-by-claim coverage are recorded in
[`doc/flanagan_scientific_contract.md`](doc/flanagan_scientific_contract.md).

The modules have distinct owners and one-way dependencies:

| Module | Owned vocabulary |
|---|---|
| `core.py` | Generic phases, roles, quantities, sensory predictions and observations, comparisons, errors, corrections, and transitions |
| `object_dynamics.py` | Object and limb models, efference copies, coordinate frames, memories, transfer, and representation purpose |
| `prehension.py` | Contact, lift, transport, replacement, release, and grip-load control |
| `pouring.py` | Source/target roles, flow phases, material regimes, and transfer variables |
| `gaze.py` | Proactive/reactive gaze, targets, fixation, and multimodal alignment |
| `observation.py` | Actor-observer evidence, visibility conditions, and prediction matching |
| `learning.py` | Exploration, acquisition, refinement, and sensorimotor-map updates |
| `perception.py` | Predicted self-contact attenuation and its unresolved relationship to control |
| `*_constraints.py` | Acceptance rules owned by the corresponding vocabulary module |
| `constraints.py` | Aggregate suite plus action-neutral constraints |
| `canonical.py` | Complete cereal and water pouring reference structures |
| `namespaces.py` | The stable project namespace family |

The dependency spine is `core -> object_dynamics -> prehension -> pouring`. Gaze,
observation, learning, and perception are optional capabilities selected through
`knowledge/microworlds/registry.py`; none introduces a reverse dependency into the
spine. Generic predictive-control types stay in `core.py` because every scientific
capability uses them.

## Scientific meaning

Every source-backed assertion distinguishes its epistemic status:

- `EMPIRICAL_FINDING`: an experimental result reported by the reviewed work.
- `SUPPORTED_INTERPRETATION`: a control organization inferred from converging results.
- `AUTHOR_HYPOTHESIS`: a mechanism proposed but not established.
- `OPEN_QUESTION`: a relationship the paper leaves unresolved.
- `ENGINEERING_EXTENSION`: useful robotic structure not attributed to the paper.

The model also keeps physical events separate from sensory evidence, predictions from
observations, phase boundaries from continuous regulation, immediate corrections from
model updates, and sensorimotor representations from perceptual judgments. Quantities
carry physical dimensions, and cached knowledge is generic rather than occurrence-level.

The following are explicit engineering extensions: role slots, condition requirements,
condition polarity, graph-closed phase transitions, numeric ranges and controller
setpoints, recovery policies, and the pouring-specific flow model. They operationalize
the scientific framework without being presented as findings from the review.

## Namespace and version contract

All Flanagan IRIs use the project-owned root
`http://www.ease-crc.org/ont/llm_reasoner/flanagan/`. Each functional module has a
distinct suffix and can evolve independently. The certified module versions are
`1.0.0`, and the certified generation prompt is `generation/v3`.

Records generated with the former placeholder namespaces, earlier module versions, or
an older prompt are retained for provenance but classified as stale. They are never
served as current knowledge and are not deleted, so old citations remain explainable.
`CachingGenerator` supplies the prompt version automatically; a read-oriented
`KnowledgeService` should be configured with `prompt_version="generation/v3"` when it
must enforce the certified generation contract.

## Using the canonical knowledge

```python
from llm_reasoner.knowledge.hypothesis import HypothesisRecord, Provenance
from llm_reasoner.knowledge.microworlds.flanagan.canonical import (
    build_pouring_phase_structure,
)
from llm_reasoner.knowledge.microworlds.flanagan.constraints import SUITE
from llm_reasoner.knowledge.microworlds.registry import current_versions
from llm_reasoner.knowledge.query import HypothesisView

structure = build_pouring_phase_structure()
record = HypothesisRecord.from_root(
    structure,
    action_type=structure.decomposes_task,
    provenance=Provenance(
        generator="hand-authored-reference",
        prompt_version="generation/v3",
        module_versions=current_versions(),
    ),
)
assert SUITE.evaluate(record.scope.fragments) == []

view = HypothesisView(record)
phases = view.phases()
first_phase_boundaries = view.boundaries_of(phases[0])
```

`HypothesisView` supplies typed, cited questions for central control concepts, including
expected events, comparisons, prediction errors, corrections, model updates, next
phases, dynamics models, grip-load coupling, continuous predictions, gaze targets, and
representation purpose. `domain(Type)` and the cited, read-only `follow()` operation are
available for less common scientific constructs.

Generation uses task profiles rather than a hardcoded pouring root. A task selects the
capabilities and most-specific structure it needs; schema derivation exposes only those
registered types, and the complete scientific constraint suite must accept the result
before it is served.

## Validation

Use the active project interpreter:

```text
python -m pytest test/llm_reasoner_test
scripts/format_docstrings.py <modified Python files>
```

The final certification tests cover stable namespaces, package ownership of canonical
structures, removal of the unused Turtle loader, module and prompt versions, and stale
cache behavior. The canonical cereal and water structures are both required to produce
zero acceptance diagnostics.
