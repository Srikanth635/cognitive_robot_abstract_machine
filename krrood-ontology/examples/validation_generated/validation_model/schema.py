"""Preserved RDF axiom records; preservation does not execute OWL semantics."""

SCHEMA_AXIOMS = (
    {
        "class_iri": "https://example.org/validation#Assembly",
        "predicate_iri": "http://www.w3.org/2000/01/rdf-schema#subClassOf",
        "root": "_:necb9e98bfbd84f18926f6e2a4849fc1fb1",
        "triples": (
            (
                "_:necb9e98bfbd84f18926f6e2a4849fc1fb1",
                "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>",
                "<http://www.w3.org/2002/07/owl#Restriction>",
            ),
            (
                "_:necb9e98bfbd84f18926f6e2a4849fc1fb1",
                "<http://www.w3.org/2002/07/owl#onProperty>",
                "<https://example.org/validation#parts>",
            ),
            (
                "_:necb9e98bfbd84f18926f6e2a4849fc1fb1",
                "<http://www.w3.org/2002/07/owl#someValuesFrom>",
                "<https://example.org/validation#Part>",
            ),
        ),
    },
    {
        "class_iri": "https://example.org/validation#Assembly",
        "predicate_iri": "http://www.w3.org/2000/01/rdf-schema#subClassOf",
        "root": "_:necb9e98bfbd84f18926f6e2a4849fc1fb2",
        "triples": (
            (
                "_:necb9e98bfbd84f18926f6e2a4849fc1fb2",
                "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>",
                "<http://www.w3.org/2002/07/owl#Restriction>",
            ),
            (
                "_:necb9e98bfbd84f18926f6e2a4849fc1fb2",
                "<http://www.w3.org/2002/07/owl#allValuesFrom>",
                "<https://example.org/validation#Part>",
            ),
            (
                "_:necb9e98bfbd84f18926f6e2a4849fc1fb2",
                "<http://www.w3.org/2002/07/owl#onProperty>",
                "<https://example.org/validation#parts>",
            ),
        ),
    },
    {
        "class_iri": "https://example.org/validation#Assembly",
        "predicate_iri": "http://www.w3.org/2000/01/rdf-schema#subClassOf",
        "root": "_:necb9e98bfbd84f18926f6e2a4849fc1fb3",
        "triples": (
            (
                "_:necb9e98bfbd84f18926f6e2a4849fc1fb3",
                "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>",
                "<http://www.w3.org/2002/07/owl#Restriction>",
            ),
            (
                "_:necb9e98bfbd84f18926f6e2a4849fc1fb3",
                "<http://www.w3.org/2002/07/owl#onClass>",
                "<https://example.org/validation#Part>",
            ),
            (
                "_:necb9e98bfbd84f18926f6e2a4849fc1fb3",
                "<http://www.w3.org/2002/07/owl#onProperty>",
                "<https://example.org/validation#parts>",
            ),
            (
                "_:necb9e98bfbd84f18926f6e2a4849fc1fb3",
                "<http://www.w3.org/2002/07/owl#qualifiedCardinality>",
                '"1"^^<http://www.w3.org/2001/XMLSchema#nonNegativeInteger>',
            ),
        ),
    },
    {
        "class_iri": "https://example.org/validation#Assembly",
        "predicate_iri": "http://www.w3.org/2000/01/rdf-schema#subClassOf",
        "root": "_:necb9e98bfbd84f18926f6e2a4849fc1fb4",
        "triples": (
            (
                "_:necb9e98bfbd84f18926f6e2a4849fc1fb4",
                "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>",
                "<http://www.w3.org/2002/07/owl#Restriction>",
            ),
            (
                "_:necb9e98bfbd84f18926f6e2a4849fc1fb4",
                "<http://www.w3.org/2002/07/owl#hasValue>",
                '"ready"',
            ),
            (
                "_:necb9e98bfbd84f18926f6e2a4849fc1fb4",
                "<http://www.w3.org/2002/07/owl#onProperty>",
                "<https://example.org/validation#tag>",
            ),
        ),
    },
    {
        "class_iri": "https://example.org/validation#Bound",
        "predicate_iri": "http://www.w3.org/2000/01/rdf-schema#subClassOf",
        "root": "_:necb9e98bfbd84f18926f6e2a4849fc1fb8",
        "triples": (
            (
                "_:necb9e98bfbd84f18926f6e2a4849fc1fb8",
                "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>",
                "<http://www.w3.org/2002/07/owl#Restriction>",
            ),
            (
                "_:necb9e98bfbd84f18926f6e2a4849fc1fb8",
                "<http://www.w3.org/2002/07/owl#hasValue>",
                "<https://example.org/validation#required>",
            ),
            (
                "_:necb9e98bfbd84f18926f6e2a4849fc1fb8",
                "<http://www.w3.org/2002/07/owl#onProperty>",
                "<https://example.org/validation#parts>",
            ),
        ),
    },
    {
        "class_iri": "https://example.org/validation#Flag",
        "predicate_iri": "http://www.w3.org/2000/01/rdf-schema#subClassOf",
        "root": "_:necb9e98bfbd84f18926f6e2a4849fc1fb7",
        "triples": (
            (
                "_:necb9e98bfbd84f18926f6e2a4849fc1fb7",
                "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>",
                "<http://www.w3.org/2002/07/owl#Restriction>",
            ),
            (
                "_:necb9e98bfbd84f18926f6e2a4849fc1fb7",
                "<http://www.w3.org/2002/07/owl#hasValue>",
                '"false"^^<http://www.w3.org/2001/XMLSchema#boolean>',
            ),
            (
                "_:necb9e98bfbd84f18926f6e2a4849fc1fb7",
                "<http://www.w3.org/2002/07/owl#onProperty>",
                "<https://example.org/validation#flag>",
            ),
        ),
    },
    {
        "class_iri": "https://example.org/validation#Maximum",
        "predicate_iri": "http://www.w3.org/2000/01/rdf-schema#subClassOf",
        "root": "_:necb9e98bfbd84f18926f6e2a4849fc1fb6",
        "triples": (
            (
                "_:necb9e98bfbd84f18926f6e2a4849fc1fb6",
                "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>",
                "<http://www.w3.org/2002/07/owl#Restriction>",
            ),
            (
                "_:necb9e98bfbd84f18926f6e2a4849fc1fb6",
                "<http://www.w3.org/2002/07/owl#maxQualifiedCardinality>",
                '"1"^^<http://www.w3.org/2001/XMLSchema#nonNegativeInteger>',
            ),
            (
                "_:necb9e98bfbd84f18926f6e2a4849fc1fb6",
                "<http://www.w3.org/2002/07/owl#onClass>",
                "<https://example.org/validation#Part>",
            ),
            (
                "_:necb9e98bfbd84f18926f6e2a4849fc1fb6",
                "<http://www.w3.org/2002/07/owl#onProperty>",
                "<https://example.org/validation#parts>",
            ),
        ),
    },
    {
        "class_iri": "https://example.org/validation#Minimum",
        "predicate_iri": "http://www.w3.org/2000/01/rdf-schema#subClassOf",
        "root": "_:necb9e98bfbd84f18926f6e2a4849fc1fb5",
        "triples": (
            (
                "_:necb9e98bfbd84f18926f6e2a4849fc1fb5",
                "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>",
                "<http://www.w3.org/2002/07/owl#Restriction>",
            ),
            (
                "_:necb9e98bfbd84f18926f6e2a4849fc1fb5",
                "<http://www.w3.org/2002/07/owl#minQualifiedCardinality>",
                '"2"^^<http://www.w3.org/2001/XMLSchema#nonNegativeInteger>',
            ),
            (
                "_:necb9e98bfbd84f18926f6e2a4849fc1fb5",
                "<http://www.w3.org/2002/07/owl#onClass>",
                "<https://example.org/validation#Part>",
            ),
            (
                "_:necb9e98bfbd84f18926f6e2a4849fc1fb5",
                "<http://www.w3.org/2002/07/owl#onProperty>",
                "<https://example.org/validation#parts>",
            ),
        ),
    },
    {
        "class_iri": "https://example.org/validation#UnionOwner",
        "predicate_iri": "http://www.w3.org/2000/01/rdf-schema#subClassOf",
        "root": "_:necb9e98bfbd84f18926f6e2a4849fc1fb9",
        "triples": (
            (
                "_:necb9e98bfbd84f18926f6e2a4849fc1fb10",
                "<http://www.w3.org/2002/07/owl#unionOf>",
                "_:necb9e98bfbd84f18926f6e2a4849fc1fb11",
            ),
            (
                "_:necb9e98bfbd84f18926f6e2a4849fc1fb11",
                "<http://www.w3.org/1999/02/22-rdf-syntax-ns#first>",
                "<https://example.org/validation#Part>",
            ),
            (
                "_:necb9e98bfbd84f18926f6e2a4849fc1fb11",
                "<http://www.w3.org/1999/02/22-rdf-syntax-ns#rest>",
                "_:necb9e98bfbd84f18926f6e2a4849fc1fb12",
            ),
            (
                "_:necb9e98bfbd84f18926f6e2a4849fc1fb12",
                "<http://www.w3.org/1999/02/22-rdf-syntax-ns#first>",
                "<https://example.org/validation#Other>",
            ),
            (
                "_:necb9e98bfbd84f18926f6e2a4849fc1fb12",
                "<http://www.w3.org/1999/02/22-rdf-syntax-ns#rest>",
                "<http://www.w3.org/1999/02/22-rdf-syntax-ns#nil>",
            ),
            (
                "_:necb9e98bfbd84f18926f6e2a4849fc1fb9",
                "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>",
                "<http://www.w3.org/2002/07/owl#Restriction>",
            ),
            (
                "_:necb9e98bfbd84f18926f6e2a4849fc1fb9",
                "<http://www.w3.org/2002/07/owl#onProperty>",
                "<https://example.org/validation#parts>",
            ),
            (
                "_:necb9e98bfbd84f18926f6e2a4849fc1fb9",
                "<http://www.w3.org/2002/07/owl#someValuesFrom>",
                "_:necb9e98bfbd84f18926f6e2a4849fc1fb10",
            ),
        ),
    },
    {
        "class_iri": None,
        "predicate_iri": "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
        "root": "_:necb9e98bfbd84f18926f6e2a4849fc1fb13",
        "triples": (
            (
                "_:necb9e98bfbd84f18926f6e2a4849fc1fb13",
                "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>",
                "<http://www.w3.org/2002/07/owl#AllDisjointClasses>",
            ),
            (
                "_:necb9e98bfbd84f18926f6e2a4849fc1fb13",
                "<http://www.w3.org/2002/07/owl#members>",
                "_:necb9e98bfbd84f18926f6e2a4849fc1fb14",
            ),
            (
                "_:necb9e98bfbd84f18926f6e2a4849fc1fb14",
                "<http://www.w3.org/1999/02/22-rdf-syntax-ns#first>",
                "<https://example.org/validation#Part>",
            ),
            (
                "_:necb9e98bfbd84f18926f6e2a4849fc1fb14",
                "<http://www.w3.org/1999/02/22-rdf-syntax-ns#rest>",
                "_:necb9e98bfbd84f18926f6e2a4849fc1fb15",
            ),
            (
                "_:necb9e98bfbd84f18926f6e2a4849fc1fb15",
                "<http://www.w3.org/1999/02/22-rdf-syntax-ns#first>",
                "<https://example.org/validation#Other>",
            ),
            (
                "_:necb9e98bfbd84f18926f6e2a4849fc1fb15",
                "<http://www.w3.org/1999/02/22-rdf-syntax-ns#rest>",
                "<http://www.w3.org/1999/02/22-rdf-syntax-ns#nil>",
            ),
        ),
    },
)
