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
from llm_reasoner.knowledge.hypotheses.service import KnowledgeService
from llm_reasoner.knowledge.hypotheses.view import HypothesisView
from llm_reasoner.knowledge.ontologies import (
    CRAM_AICOR_NAMESPACE,
    DUL_NAMESPACE,
    HYPOTHESIS_CONCEPT_NAMESPACE,
    PREDICTIVE_MANIPULATION_NAMESPACE_ROOT,
    LinkedOntologyVariable,
    PredictiveManipulationOntologyFile,
)
from llm_reasoner.knowledge.ontologies.linked import LinkedOntologyTerms, OntologyFiles
from llm_reasoner.knowledge.ontologies.predictive_manipulation import (
    core,
    gaze,
    object_dynamics,
    pouring,
    prehension,
)
from llm_reasoner.knowledge.ontologies.terms import (
    ConceptReference,
    DescriptionEntity,
    PropertyTerm,
    is_inferred,
)
from llm_reasoner.knowledge.ontologies.vocabulary import Vocabulary
from llm_reasoner.knowledge.representation.rdf import GraphReader
from llm_reasoner.knowledge.storage.store import HypothesisStore

__all__ = [
    "AcceptanceSuite",
    "CONCEPT_PREFIXES",
    "CRAM_AICOR_NAMESPACE",
    "ClosedWorldMembership",
    "ConceptReference",
    "ConsistencyCheck",
    "DUL_NAMESPACE",
    "DescriptionEntity",
    "GraphReader",
    "HYPOTHESIS_CONCEPT_NAMESPACE",
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
    "KnowledgeService",
    "LinkedOntologyTerms",
    "LinkedOntologyVariable",
    "OntologyFiles",
    "PREDICTIVE_MANIPULATION_NAMESPACE_ROOT",
    "PredictiveManipulationOntologyFile",
    "PropertyTerm",
    "Severity",
    "SyntheticHypothesisRelation",
    "Vocabulary",
    "core",
    "described",
    "gaze",
    "is_inferred",
    "object_dynamics",
    "pouring",
    "predictive_manipulation_suite",
    "prehension",
]
