"""Preserved RDF axiom records; preservation does not execute OWL semantics."""

SCHEMA_AXIOMS = (
    {
        "class_iri": "https://example.org/roundtrip#Pick",
        "predicate_iri": "http://www.w3.org/2002/07/owl#equivalentClass",
        "root": "_:n26f6b58cb8404c84882e5cfd00138c74b1",
        "triples": (
            (
                "_:n26f6b58cb8404c84882e5cfd00138c74b1",
                "<http://www.w3.org/2002/07/owl#oneOf>",
                "_:n26f6b58cb8404c84882e5cfd00138c74b2",
            ),
            (
                "_:n26f6b58cb8404c84882e5cfd00138c74b2",
                "<http://www.w3.org/1999/02/22-rdf-syntax-ns#first>",
                "<https://example.org/roundtrip#one>",
            ),
            (
                "_:n26f6b58cb8404c84882e5cfd00138c74b2",
                "<http://www.w3.org/1999/02/22-rdf-syntax-ns#rest>",
                "_:n26f6b58cb8404c84882e5cfd00138c74b3",
            ),
            (
                "_:n26f6b58cb8404c84882e5cfd00138c74b3",
                "<http://www.w3.org/1999/02/22-rdf-syntax-ns#first>",
                "<https://example.org/roundtrip#two>",
            ),
            (
                "_:n26f6b58cb8404c84882e5cfd00138c74b3",
                "<http://www.w3.org/1999/02/22-rdf-syntax-ns#rest>",
                "_:n26f6b58cb8404c84882e5cfd00138c74b4",
            ),
            (
                "_:n26f6b58cb8404c84882e5cfd00138c74b4",
                "<http://www.w3.org/1999/02/22-rdf-syntax-ns#first>",
                "<https://example.org/roundtrip#three>",
            ),
            (
                "_:n26f6b58cb8404c84882e5cfd00138c74b4",
                "<http://www.w3.org/1999/02/22-rdf-syntax-ns#rest>",
                "<http://www.w3.org/1999/02/22-rdf-syntax-ns#nil>",
            ),
        ),
    },
    {
        "class_iri": None,
        "predicate_iri": "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
        "root": "_:n26f6b58cb8404c84882e5cfd00138c74b5",
        "triples": (
            (
                "_:n26f6b58cb8404c84882e5cfd00138c74b5",
                "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>",
                "<http://www.w3.org/2002/07/owl#AllDifferent>",
            ),
            (
                "_:n26f6b58cb8404c84882e5cfd00138c74b5",
                "<http://www.w3.org/2002/07/owl#distinctMembers>",
                "_:n26f6b58cb8404c84882e5cfd00138c74b6",
            ),
            (
                "_:n26f6b58cb8404c84882e5cfd00138c74b6",
                "<http://www.w3.org/1999/02/22-rdf-syntax-ns#first>",
                "<https://example.org/roundtrip#one>",
            ),
            (
                "_:n26f6b58cb8404c84882e5cfd00138c74b6",
                "<http://www.w3.org/1999/02/22-rdf-syntax-ns#rest>",
                "_:n26f6b58cb8404c84882e5cfd00138c74b7",
            ),
            (
                "_:n26f6b58cb8404c84882e5cfd00138c74b7",
                "<http://www.w3.org/1999/02/22-rdf-syntax-ns#first>",
                "<https://example.org/roundtrip#two>",
            ),
            (
                "_:n26f6b58cb8404c84882e5cfd00138c74b7",
                "<http://www.w3.org/1999/02/22-rdf-syntax-ns#rest>",
                "_:n26f6b58cb8404c84882e5cfd00138c74b8",
            ),
            (
                "_:n26f6b58cb8404c84882e5cfd00138c74b8",
                "<http://www.w3.org/1999/02/22-rdf-syntax-ns#first>",
                "<https://example.org/roundtrip#three>",
            ),
            (
                "_:n26f6b58cb8404c84882e5cfd00138c74b8",
                "<http://www.w3.org/1999/02/22-rdf-syntax-ns#rest>",
                "<http://www.w3.org/1999/02/22-rdf-syntax-ns#nil>",
            ),
        ),
    },
)
