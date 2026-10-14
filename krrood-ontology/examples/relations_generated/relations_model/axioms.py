"""Translated relation-axiom metadata; KRROOD descriptors execute the implications."""

RELATION_AXIOMS = (
    (
        "https://example.org/relations#knows",
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
        "http://www.w3.org/2002/07/owl#SymmetricProperty",
    ),
    (
        "https://example.org/relations#containsPart",
        "http://www.w3.org/2002/07/owl#inverseOf",
        "https://example.org/relations#partOf",
    ),
    (
        "https://example.org/relations#containsPart",
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
        "http://www.w3.org/2002/07/owl#TransitiveProperty",
    ),
    (
        "https://example.org/relations#worksFor",
        "http://www.w3.org/2002/07/owl#inverseOf",
        "https://example.org/relations#hasEmployee",
    ),
    (
        "https://example.org/relations#hasEmployee",
        "http://www.w3.org/2002/07/owl#inverseOf",
        "https://example.org/relations#worksFor",
    ),
    (
        "https://example.org/relations#partOf",
        "http://www.w3.org/2002/07/owl#inverseOf",
        "https://example.org/relations#containsPart",
    ),
    (
        "https://example.org/relations#colleagueOf",
        "http://www.w3.org/2000/01/rdf-schema#subPropertyOf",
        "https://example.org/relations#knows",
    ),
    (
        "https://example.org/relations#employeeOf",
        "http://www.w3.org/2000/01/rdf-schema#subPropertyOf",
        "https://example.org/relations#worksFor",
    ),
    (
        "https://example.org/relations#headsOf",
        "http://www.w3.org/2000/01/rdf-schema#subPropertyOf",
        "https://example.org/relations#employeeOf",
    ),
)

UPDATE_POLICY = "additions_after_construction"
RETRACTION_SUPPORTED = False
