"""Translated relation-axiom metadata; KRROOD descriptors execute the implications."""

RELATION_AXIOMS = (
    (
        "http://aicor.knowledge/l2-ontology.owl#coRefersWith",
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
        "http://www.w3.org/2002/07/owl#SymmetricProperty",
    ),
    (
        "http://aicor.knowledge/l2-ontology.owl#coRefersWith",
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
        "http://www.w3.org/2002/07/owl#TransitiveProperty",
    ),
    (
        "http://aicor.knowledge/l2-ontology.owl#hasPrecondition",
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
        "http://www.w3.org/2002/07/owl#TransitiveProperty",
    ),
    (
        "http://aicor.knowledge/l2-ontology.owl#hasSubaction",
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
        "http://www.w3.org/2002/07/owl#TransitiveProperty",
    ),
    (
        "http://aicor.knowledge/l2-ontology.owl#partOf",
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
        "http://www.w3.org/2002/07/owl#TransitiveProperty",
    ),
)

UPDATE_POLICY = "additions_after_construction"
RETRACTION_SUPPORTED = False
