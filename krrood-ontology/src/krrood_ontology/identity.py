"""Shared parsing of explicit RDF distinctness, without identity normalization."""

from dataclasses import dataclass

from rdflib import OWL, RDF, URIRef

from krrood_ontology.expressions import expression_members


@dataclass(frozen=True)
class IdentityConstraints:
    """Declared distinctness is evidence; different IRIs alone are not."""

    groups: tuple[frozenset[str], ...] = ()

    @classmethod
    def from_graph(cls, graph, fail, *, registered=None):
        groups = set()
        for a, b in graph.subject_objects(OWL.differentFrom):
            if not isinstance(a, URIRef) or not isinstance(b, URIRef):
                fail("identity_members", "differentFrom requires named operands")
            if a == b:
                fail(
                    "identity_conflict",
                    "An individual cannot differ from itself",
                    str(a),
                )
            groups.add(frozenset((str(a), str(b))))
        for group in graph.subjects(RDF.type, OWL.AllDifferent):
            heads = tuple(graph.objects(group, OWL.distinctMembers))
            if len(heads) != 1:
                fail(
                    "identity_list",
                    "AllDifferent requires one distinctMembers list",
                    str(group),
                )
            members = expression_members(graph, heads[0], fail)
            if len(members) < 2 or any(not isinstance(v, URIRef) for v in members):
                fail(
                    "identity_members",
                    "AllDifferent requires at least two named operands",
                    str(group),
                )
            if len(set(members)) != len(members):
                fail(
                    "identity_conflict",
                    "AllDifferent repeats an individual",
                    str(group),
                )
            groups.add(frozenset(map(str, members)))
        if registered is not None:
            missing = set().union(*groups) - set(registered) if groups else set()
            if missing:
                fail(
                    "unresolved_identity",
                    "Distinctness operands must be explicitly typed individuals in this load",
                    sorted(missing)[0],
                )
        return cls(tuple(sorted(groups, key=lambda g: tuple(sorted(g)))))
