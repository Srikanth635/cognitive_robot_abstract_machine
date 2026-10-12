# `llm_reasoner`

`llm_reasoner` resolves underspecified requests and maintains reusable,
description-level knowledge that can guide later resolutions. Its knowledge is
deliberately independent of a particular robot, world, or episode: a hypothesis says
what a kind of action normally requires, while the resolver binds that description to
the current world.

The knowledge is written in the terms of an OWL ontology. A language model generates it
as instances of Python classes generated from that ontology, and it is admitted only
when it meets the ontology's acceptance rules and, optionally, is consistent with the
ontology under a reasoner.

## Architecture and status

- [`doc/architecture.md`](doc/architecture.md): the components (world injector,
  resolver, knowledge component, clients), what each is for, its inputs and outputs, and
  how they communicate.
- [`doc/status.md`](doc/status.md): what is done, in progress, decided and proposed.

## The predictive-manipulation ontology

`knowledge/ontologies/predictive_manipulation` represents the predictive organization of object
manipulation described by Johansson and Flanagan. It imports CRAM-AICOR and DUL and
links its concepts to both; neither is changed. The scientific authority, claim
statuses and semantic distinctions are recorded in
[`doc/flanagan_scientific_contract.md`](doc/flanagan_scientific_contract.md).

| Module | Owned vocabulary |
|---|---|
| `core.ttl` | Phases, roles, parameters, conditions, events, forces, sensory predictions and observations, comparisons, errors, corrections, transitions |
| `object-dynamics.ttl` | Object and limb dynamics models, efference copies, perturbations, transfer, memories, representation dissociation |
| `prehension.ttl` | Contact, lift, transport, replacement and release phases, and grip-load control |
| `pouring.ttl` | Source and target roles, flow phases and events, material regimes |
| `gaze.ttl` | Gaze fixations, landmarks and visual control points |

Each module has a Python counterpart next to it (`core.py`, ...), generated and checked
in. After changing a module, regenerate them:

```text
python -m llm_reasoner.knowledge.ontology_translation
```

A test fails when the checked-in vocabulary differs from what the ontology generates.

Every source-backed claim carries its evidence status: `EMPIRICAL_FINDING`,
`SUPPORTED_INTERPRETATION`, `AUTHOR_HYPOTHESIS`, `OPEN_QUESTION` or
`ENGINEERING_EXTENSION`.

## Generating and reading knowledge

```python
from pathlib import Path

from krrood.entity_query_language.factories import the

from llm_reasoner.knowledge.storage.store import HypothesisStore
from llm_reasoner.clients import AnthropicClient
from llm_reasoner.generation.backend import HypothesisBackend
from llm_reasoner.hypotheses.service import KnowledgeService
from llm_reasoner.knowledge.ontologies import CRAM_AICOR_NAMESPACE
from llm_reasoner.knowledge.ontologies.predictive_manipulation import pouring
from llm_reasoner.knowledge.ontologies.terms import ConceptReference

service = KnowledgeService(
    backend=HypothesisBackend(
        client=AnthropicClient(),
        store=HypothesisStore(directory=Path("hypotheses")),
    )
)
view = service.view_for(
    "pour milk from the carton into the bowl",
    the(pouring.PouringPhaseStructure)(
        decomposes_task=ConceptReference(iri=CRAM_AICOR_NAMESPACE + "Pouring")
    ),
)
phases = view.phases()
forces = view.force_dynamics()
```

The service answers from the store when it holds a hypothesis for the same instruction,
query and ontology versions, and generates one otherwise. Passing a
`consistency=HermitCheck(...)` to the backend also checks each answer against the
ontology, CRAM-AICOR and DUL with HermiT (the optional `reasoning` extra), and locates
the fragments of a contradicting answer for the repair prompt.

`HypothesisView` answers questions spanning several fragments: the phases in order, the
requirements of a phase by mode, world assumptions, failures, corrections at an event,
the forces each event changes, gaze fixations and grip-load couplings. Every fragment is
cited by the IRI it is stored under.

## Validation

Use the active project interpreter:

```text
python -m pytest test/llm_reasoner_test
scripts/format_docstrings.py <modified Python files>
```

Tests marked `slow` run HermiT; they need Java, owlready2, and the `CRAM_AICOR_ONTOLOGY`
and `DUL_ONTOLOGY` environment variables pointing at those ontologies.
