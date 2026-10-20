"""
The knowledge component as other packages reach it: every name another package may use,
gathered in one module so that the component's own layout can change without them.

Another package, such as cram_viz, imports from this module only.
"""

from llm_reasoner.knowledge.acceptance.derived import described
from llm_reasoner.knowledge.acceptance.restrictions import ClosedWorldMembership
from llm_reasoner.knowledge.acceptance.rules import AcceptanceSuite, Severity
from llm_reasoner.knowledge.acceptance.suite import predictive_manipulation_suite
from llm_reasoner.knowledge.admission.consistency import ConsistencyCheck
from llm_reasoner.knowledge.admission.hermit import HermitCheck
from llm_reasoner.knowledge.generation.backend import HypothesisBackend
from llm_reasoner.knowledge.generation.dialogue import HypothesisRejected
from llm_reasoner.knowledge.generation.wire import CONCEPT_PREFIXES
from llm_reasoner.knowledge.hypotheses.graph import (
    HypothesisGraph,
    HypothesisGraphNode,
    HypothesisGraphNodeKind,
    HypothesisGraphPresentation,
    HypothesisGraphProjector,
    HypothesisGraphRelation,
    SyntheticHypothesisRelation,
)
from llm_reasoner.knowledge.hypotheses.knowledge_access import (
    KNOWLEDGE_ACCESS_VERSION,
    KnowledgeAccess,
    KnowledgeToolName,
)
from llm_reasoner.knowledge.hypotheses.knowledge_graph import KnowledgeGraph
from llm_reasoner.knowledge.hypotheses.service import KnowledgeService
from llm_reasoner.knowledge.hypotheses.view import HypothesisView
from llm_reasoner.knowledge.ontologies import (
    ActionPerspectiveFile,
    CRAM_AICOR_NAMESPACE,
    DUL_NAMESPACE,
    HYPOTHESIS_CONCEPT_NAMESPACE,
    PREDICTIVE_MANIPULATION_NAMESPACE_ROOT,
    LinkedOntologyVariable,
    OntologyModuleFile,
    PredictiveManipulationOntologyFile,
)
from llm_reasoner.knowledge.ontologies.action_perspectives import (
    action_core,
    action_frame,
    failures,
    image_schemas,
)
from llm_reasoner.knowledge.ontologies.described_actions import (
    DescribedAction,
    DescribedActions,
)
from llm_reasoner.knowledge.ontologies.linked import LinkedOntologyTerms, OntologyFiles
from llm_reasoner.knowledge.ontologies.object_kinds import (
    HAS_OPENING,
    OBJECT_KINDS_FILE,
    OBJECT_KINDS_NAMESPACE,
    ObjectKinds,
)
from llm_reasoner.knowledge.ontologies.predictive_manipulation import (
    core,
    gaze,
    object_dynamics,
    pouring,
    prehension,
)
from llm_reasoner.knowledge.ontologies.perspective_roots import (
    PERSPECTIVE_ROOTS,
    PerspectiveRoot,
)
from llm_reasoner.knowledge.ontologies.situation import (
    SituationFeature,
    SituationFeatures,
)
from llm_reasoner.knowledge.ontologies.terms import (
    ConceptReference,
    DescriptionEntity,
    PropertyTerm,
    is_inferred,
)
from llm_reasoner.knowledge.ontologies.vocabulary import Vocabulary
from llm_reasoner.knowledge.representation.rdf import GraphReader
from llm_reasoner.knowledge.storage.citation import Citation, StoredRoot
from llm_reasoner.knowledge.storage.store import HypothesisStore

__all__ = [
    "AcceptanceSuite",
    "action_core",
    "action_frame",
    "ActionPerspectiveFile",
    "Citation",
    "ClosedWorldMembership",
    "ConceptReference",
    "CONCEPT_PREFIXES",
    "ConsistencyCheck",
    "core",
    "CRAM_AICOR_NAMESPACE",
    "described",
    "failures",
    "DescribedAction",
    "DescribedActions",
    "DescriptionEntity",
    "DUL_NAMESPACE",
    "gaze",
    "GraphReader",
    "HAS_OPENING",
    "HermitCheck",
    "HypothesisBackend",
    "HypothesisGraph",
    "HypothesisGraphNode",
    "HypothesisGraphNodeKind",
    "HypothesisGraphPresentation",
    "HypothesisGraphProjector",
    "HypothesisGraphRelation",
    "HypothesisRejected",
    "HypothesisStore",
    "HypothesisView",
    "HYPOTHESIS_CONCEPT_NAMESPACE",
    "image_schemas",
    "is_inferred",
    "KNOWLEDGE_ACCESS_VERSION",
    "KnowledgeAccess",
    "KnowledgeGraph",
    "KnowledgeToolName",
    "KnowledgeService",
    "LinkedOntologyTerms",
    "OBJECT_KINDS_FILE",
    "OBJECT_KINDS_NAMESPACE",
    "ObjectKinds",
    "LinkedOntologyVariable",
    "object_dynamics",
    "OntologyFiles",
    "OntologyModuleFile",
    "PERSPECTIVE_ROOTS",
    "PerspectiveRoot",
    "pouring",
    "PredictiveManipulationOntologyFile",
    "PREDICTIVE_MANIPULATION_NAMESPACE_ROOT",
    "predictive_manipulation_suite",
    "prehension",
    "PropertyTerm",
    "Severity",
    "SituationFeature",
    "SituationFeatures",
    "StoredRoot",
    "SyntheticHypothesisRelation",
    "Vocabulary",
]
