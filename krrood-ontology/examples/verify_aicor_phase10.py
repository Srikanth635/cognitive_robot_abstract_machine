"""AICOR acceptance fixture for generic identity loading and RDF export."""

import json
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from rdflib import BNode, Graph

sys.path.insert(0, str(Path(sys.argv[1]).resolve()))
import aicor_model as m

source = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).parent / "aicor.ttl"
start = time.monotonic()
loaded = m.load_individuals(source)
assert len(loaded.by_iri) == 4
assert len(loaded.distinct_groups) == 1 and len(loaded.distinct_groups[0]) == 4
assert len({id(v) for v in loaded.by_iri.values()}) == 4
for iri, node in m.CLASS_EXPRESSIONS.items():
    if node.kind.value == "enumeration":
        assert {id(v) for v in m.model_queries.match(iri, loaded)} == {
            id(loaded[v]) for v in node.individuals
        }
result = m.model_exporter.export(loaded, mode="current")
assert result.added_triples == 0 and result.removed_triples == 0
assert set(result.graph) == set(loaded.asserted_graph)
with TemporaryDirectory() as d:
    path = Path(d) / "roundtrip.nt"
    m.model_exporter.write(loaded, path, mode="current", format="nt")
    # RDFLib's native N-Triples parser can preserve explicitly emitted blank labels.
    # This checks every triple without expensive whole-graph canonicalization.
    context = {str(v): v for v in result.graph.all_nodes() if isinstance(v, BNode)}
    parsed = Graph().parse(path, format="nt", bnode_context=context)
    assert set(parsed) == set(result.graph)
    restored = m.load_individuals(path)
    assert set(restored.by_iri) == set(loaded.by_iri)
    assert restored.asserted_types == loaded.asserted_types
    assert restored.distinct_groups == loaded.distinct_groups
    assert len(restored.asserted_graph) == len(loaded.asserted_graph)
report = m.model_validator.validate(loaded, complete=True)
assert not any(
    c.kind == "identity" and c.status == "unsupported" for c in report.checks
)
print(
    json.dumps(
        {
            "classes": len(m.CLASS_BY_IRI),
            "properties": len(m.MODEL.properties),
            "loaded_individuals": len(loaded.by_iri),
            "distinct_groups": len(loaded.distinct_groups),
            "distinct_members": len(loaded.distinct_groups[0]),
            "source_triples": len(loaded.asserted_graph),
            "roundtrip_all_triples_equal": True,
            "roundtrip_types_and_identity_equal": True,
            "source_bytes_retained": loaded.source_bytes == source.read_bytes(),
            "owl_validation_summary": report.to_dict()["summary"],
            "runtime_check_seconds": round(time.monotonic() - start, 3),
        },
        indent=2,
    )
)
