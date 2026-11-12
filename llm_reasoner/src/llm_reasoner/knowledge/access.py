"""
The knowledge component as other packages reach it: every name another package may use,
gathered in one module so that the component's own layout can change without them.

Another package, such as cramera, imports from this module only.
"""

from llm_reasoner.knowledge.acceptance.derived import described
from llm_reasoner.knowledge.acceptance.restrictions import ClosedWorldMembership
from llm_reasoner.knowledge.acceptance.rules import AcceptanceSuite, Severity
from llm_reasoner.knowledge.acceptance.suite import ontology_rules_suite
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
from llm_reasoner.knowledge.generation.briefs import GenerationBrief
from llm_reasoner.knowledge.generation.prompts import GenerationPrompt
from llm_reasoner.knowledge.vocabulary.ontology_modules import (
    HYPOTHESIS_CONCEPT_NAMESPACE,
    KNOWLEDGE_ONTOLOGY_MODULES,
    OntologyModuleFile,
)
from llm_reasoner.knowledge.vocabulary.linked import (
    LinkedClassHierarchy,
    LinkedOntologyTerms,
    OntologyFiles,
)
from llm_reasoner.knowledge.vocabulary.situation import (
    SituationFeature,
    SituationFeatures,
)
from llm_reasoner.knowledge.vocabulary.terms import (
    ConceptReference,
    DescriptionEntity,
    PropertyTerm,
    is_inferred,
)
from llm_reasoner.knowledge.vocabulary.lookup import Vocabulary
from llm_reasoner.knowledge.representation.rdf import GraphReader
from llm_reasoner.knowledge.storage.citation import Citation, StoredRoot
from llm_reasoner.knowledge.storage.store import HypothesisStore

__all__ = [
    "AcceptanceSuite",
    "Citation",
    "ClosedWorldMembership",
    "ConceptReference",
    "CONCEPT_PREFIXES",
    "ConsistencyCheck",
    "described",
    "DescriptionEntity",
    "GenerationBrief",
    "GenerationPrompt",
    "GraphReader",
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
    "is_inferred",
    "KNOWLEDGE_ACCESS_VERSION",
    "KNOWLEDGE_ONTOLOGY_MODULES",
    "KnowledgeAccess",
    "KnowledgeGraph",
    "KnowledgeToolName",
    "KnowledgeService",
    "LinkedClassHierarchy",
    "LinkedOntologyTerms",
    "ontology_rules_suite",
    "OntologyFiles",
    "OntologyModuleFile",
    "PropertyTerm",
    "Severity",
    "SituationFeature",
    "SituationFeatures",
    "StoredRoot",
    "SyntheticHypothesisRelation",
    "Vocabulary",
]
