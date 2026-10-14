"""Preserved RDF axiom records; preservation does not execute OWL semantics."""

SCHEMA_AXIOMS = (
    {
        "class_iri": "https://example.org/scoped#Carrier",
        "predicate_iri": "http://www.w3.org/2000/01/rdf-schema#subClassOf",
        "root": "_:n09cdb4b97eb24fe6a108d8ad7c2b073eb12",
        "triples": (
            (
                "_:n09cdb4b97eb24fe6a108d8ad7c2b073eb12",
                "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>",
                "<http://www.w3.org/2002/07/owl#Restriction>",
            ),
            (
                "_:n09cdb4b97eb24fe6a108d8ad7c2b073eb12",
                "<http://www.w3.org/2002/07/owl#onProperty>",
                "<https://example.org/scoped#link>",
            ),
            (
                "_:n09cdb4b97eb24fe6a108d8ad7c2b073eb12",
                "<http://www.w3.org/2002/07/owl#someValuesFrom>",
                "<http://www.w3.org/2002/07/owl#Thing>",
            ),
        ),
    },
    {
        "class_iri": "https://example.org/scoped#Choice",
        "predicate_iri": "http://www.w3.org/2002/07/owl#equivalentClass",
        "root": "_:n09cdb4b97eb24fe6a108d8ad7c2b073eb9",
        "triples": (
            (
                "_:n09cdb4b97eb24fe6a108d8ad7c2b073eb10",
                "<http://www.w3.org/1999/02/22-rdf-syntax-ns#first>",
                "<https://example.org/scoped#one>",
            ),
            (
                "_:n09cdb4b97eb24fe6a108d8ad7c2b073eb10",
                "<http://www.w3.org/1999/02/22-rdf-syntax-ns#rest>",
                "_:n09cdb4b97eb24fe6a108d8ad7c2b073eb11",
            ),
            (
                "_:n09cdb4b97eb24fe6a108d8ad7c2b073eb11",
                "<http://www.w3.org/1999/02/22-rdf-syntax-ns#first>",
                "<https://example.org/scoped#two>",
            ),
            (
                "_:n09cdb4b97eb24fe6a108d8ad7c2b073eb11",
                "<http://www.w3.org/1999/02/22-rdf-syntax-ns#rest>",
                "<http://www.w3.org/1999/02/22-rdf-syntax-ns#nil>",
            ),
            (
                "_:n09cdb4b97eb24fe6a108d8ad7c2b073eb9",
                "<http://www.w3.org/2002/07/owl#oneOf>",
                "_:n09cdb4b97eb24fe6a108d8ad7c2b073eb10",
            ),
        ),
    },
    {
        "class_iri": "https://example.org/scoped#Either",
        "predicate_iri": "http://www.w3.org/2002/07/owl#equivalentClass",
        "root": "_:n09cdb4b97eb24fe6a108d8ad7c2b073eb1",
        "triples": (
            (
                "_:n09cdb4b97eb24fe6a108d8ad7c2b073eb1",
                "<http://www.w3.org/2002/07/owl#unionOf>",
                "_:n09cdb4b97eb24fe6a108d8ad7c2b073eb2",
            ),
            (
                "_:n09cdb4b97eb24fe6a108d8ad7c2b073eb2",
                "<http://www.w3.org/1999/02/22-rdf-syntax-ns#first>",
                "<https://example.org/scoped#Alpha>",
            ),
            (
                "_:n09cdb4b97eb24fe6a108d8ad7c2b073eb2",
                "<http://www.w3.org/1999/02/22-rdf-syntax-ns#rest>",
                "_:n09cdb4b97eb24fe6a108d8ad7c2b073eb3",
            ),
            (
                "_:n09cdb4b97eb24fe6a108d8ad7c2b073eb3",
                "<http://www.w3.org/1999/02/22-rdf-syntax-ns#first>",
                "<https://example.org/scoped#Beta>",
            ),
            (
                "_:n09cdb4b97eb24fe6a108d8ad7c2b073eb3",
                "<http://www.w3.org/1999/02/22-rdf-syntax-ns#rest>",
                "<http://www.w3.org/1999/02/22-rdf-syntax-ns#nil>",
            ),
        ),
    },
    {
        "class_iri": "https://example.org/scoped#FlagCarrier",
        "predicate_iri": "http://www.w3.org/2000/01/rdf-schema#subClassOf",
        "root": "_:n09cdb4b97eb24fe6a108d8ad7c2b073eb13",
        "triples": (
            (
                "_:n09cdb4b97eb24fe6a108d8ad7c2b073eb13",
                "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>",
                "<http://www.w3.org/2002/07/owl#Restriction>",
            ),
            (
                "_:n09cdb4b97eb24fe6a108d8ad7c2b073eb13",
                "<http://www.w3.org/2002/07/owl#hasValue>",
                '"false"^^<http://www.w3.org/2001/XMLSchema#boolean>',
            ),
            (
                "_:n09cdb4b97eb24fe6a108d8ad7c2b073eb13",
                "<http://www.w3.org/2002/07/owl#onProperty>",
                "<https://example.org/scoped#flag>",
            ),
        ),
    },
    {
        "class_iri": "https://example.org/scoped#Linked",
        "predicate_iri": "http://www.w3.org/2002/07/owl#equivalentClass",
        "root": "_:n09cdb4b97eb24fe6a108d8ad7c2b073eb4",
        "triples": (
            (
                "_:n09cdb4b97eb24fe6a108d8ad7c2b073eb4",
                "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>",
                "<http://www.w3.org/2002/07/owl#Restriction>",
            ),
            (
                "_:n09cdb4b97eb24fe6a108d8ad7c2b073eb4",
                "<http://www.w3.org/2002/07/owl#onProperty>",
                "<https://example.org/scoped#link>",
            ),
            (
                "_:n09cdb4b97eb24fe6a108d8ad7c2b073eb4",
                "<http://www.w3.org/2002/07/owl#someValuesFrom>",
                "_:n09cdb4b97eb24fe6a108d8ad7c2b073eb5",
            ),
            (
                "_:n09cdb4b97eb24fe6a108d8ad7c2b073eb5",
                "<http://www.w3.org/2002/07/owl#unionOf>",
                "_:n09cdb4b97eb24fe6a108d8ad7c2b073eb6",
            ),
            (
                "_:n09cdb4b97eb24fe6a108d8ad7c2b073eb6",
                "<http://www.w3.org/1999/02/22-rdf-syntax-ns#first>",
                "<https://example.org/scoped#Alpha>",
            ),
            (
                "_:n09cdb4b97eb24fe6a108d8ad7c2b073eb6",
                "<http://www.w3.org/1999/02/22-rdf-syntax-ns#rest>",
                "_:n09cdb4b97eb24fe6a108d8ad7c2b073eb7",
            ),
            (
                "_:n09cdb4b97eb24fe6a108d8ad7c2b073eb7",
                "<http://www.w3.org/1999/02/22-rdf-syntax-ns#first>",
                "<https://example.org/scoped#Beta>",
            ),
            (
                "_:n09cdb4b97eb24fe6a108d8ad7c2b073eb7",
                "<http://www.w3.org/1999/02/22-rdf-syntax-ns#rest>",
                "<http://www.w3.org/1999/02/22-rdf-syntax-ns#nil>",
            ),
        ),
    },
    {
        "class_iri": "https://example.org/scoped#Marked",
        "predicate_iri": "http://www.w3.org/2002/07/owl#equivalentClass",
        "root": "_:n09cdb4b97eb24fe6a108d8ad7c2b073eb8",
        "triples": (
            (
                "_:n09cdb4b97eb24fe6a108d8ad7c2b073eb8",
                "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>",
                "<http://www.w3.org/2002/07/owl#Restriction>",
            ),
            (
                "_:n09cdb4b97eb24fe6a108d8ad7c2b073eb8",
                "<http://www.w3.org/2002/07/owl#hasValue>",
                '"false"^^<http://www.w3.org/2001/XMLSchema#boolean>',
            ),
            (
                "_:n09cdb4b97eb24fe6a108d8ad7c2b073eb8",
                "<http://www.w3.org/2002/07/owl#onProperty>",
                "<https://example.org/scoped#flag>",
            ),
        ),
    },
    {
        "class_iri": "https://example.org/scoped#Nested",
        "predicate_iri": "http://www.w3.org/2002/07/owl#equivalentClass",
        "root": "_:n09cdb4b97eb24fe6a108d8ad7c2b073eb14",
        "triples": (
            (
                "_:n09cdb4b97eb24fe6a108d8ad7c2b073eb14",
                "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>",
                "<http://www.w3.org/2002/07/owl#Restriction>",
            ),
            (
                "_:n09cdb4b97eb24fe6a108d8ad7c2b073eb14",
                "<http://www.w3.org/2002/07/owl#onProperty>",
                "<https://example.org/scoped#link>",
            ),
            (
                "_:n09cdb4b97eb24fe6a108d8ad7c2b073eb14",
                "<http://www.w3.org/2002/07/owl#someValuesFrom>",
                "_:n09cdb4b97eb24fe6a108d8ad7c2b073eb15",
            ),
            (
                "_:n09cdb4b97eb24fe6a108d8ad7c2b073eb15",
                "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>",
                "<http://www.w3.org/2002/07/owl#Restriction>",
            ),
            (
                "_:n09cdb4b97eb24fe6a108d8ad7c2b073eb15",
                "<http://www.w3.org/2002/07/owl#onProperty>",
                "<https://example.org/scoped#link>",
            ),
            (
                "_:n09cdb4b97eb24fe6a108d8ad7c2b073eb15",
                "<http://www.w3.org/2002/07/owl#someValuesFrom>",
                "<https://example.org/scoped#Alpha>",
            ),
        ),
    },
)
