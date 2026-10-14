"""Translated relation-axiom metadata; KRROOD descriptors execute the implications."""

RELATION_AXIOMS = (
    (
        "https://example.org/expressions#hasEmployee",
        "http://www.w3.org/2002/07/owl#inverseOf",
        "https://example.org/expressions#worksFor",
    ),
    (
        "https://example.org/expressions#worksFor",
        "http://www.w3.org/2002/07/owl#inverseOf",
        "https://example.org/expressions#hasEmployee",
    ),
)

UPDATE_POLICY = "additions_after_construction"
RETRACTION_SUPPORTED = False
