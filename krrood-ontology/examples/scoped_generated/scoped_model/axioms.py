"""Translated relation-axiom metadata; KRROOD descriptors execute the implications."""

RELATION_AXIOMS = (
    (
        "https://example.org/scoped#connected",
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
        "http://www.w3.org/2002/07/owl#SymmetricProperty",
    ),
    (
        "https://example.org/scoped#connected",
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
        "http://www.w3.org/2002/07/owl#TransitiveProperty",
    ),
)

UPDATE_POLICY = "additions_after_construction"
RETRACTION_SUPPORTED = False
