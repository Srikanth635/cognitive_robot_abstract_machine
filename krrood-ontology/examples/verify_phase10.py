"""Generic identity and editable-field RDF round-trip example."""

import json
import sys
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory

from rdflib import Graph
from rdflib.compare import isomorphic

sys.path.insert(0, str(Path(sys.argv[1]).resolve()))
import roundtrip_model as m

source = (
    Path(sys.argv[2])
    if len(sys.argv) > 2
    else Path(__file__).parent / "identity_roundtrip.ttl"
)
base = "https://example.org/roundtrip#"
loaded = m.load_individuals(source)
assert len(loaded.by_iri) == 3
assert len(loaded.distinct_groups) == 2
assert set(m.model_queries.match(base + "Pick", loaded)) == set(loaded.by_iri.values())
initial = m.model_exporter.export(loaded, mode="current")
assert isomorphic(initial.graph, loaded.asserted_graph)
assert initial.omitted_inferred_facts > 0
loaded[base + "one"].amount[:] = [Decimal("2.50")]
with TemporaryDirectory() as d:
    path = Path(d) / "current.ttl"
    edited = m.model_exporter.write(loaded, path, mode="current")
    restored = m.load_individuals(path)
    assert restored[base + "one"].amount == [Decimal("2.50")]
    assert restored.distinct_groups == loaded.distinct_groups
    assert isomorphic(Graph().parse(path), edited.graph)
    assert isomorphic(m.model_exporter.graph(restored, mode="current"), edited.graph)
print(
    json.dumps(
        {
            "loaded_individuals": len(loaded.by_iri),
            "distinct_groups": len(loaded.distinct_groups),
            "unchanged_current_isomorphic": True,
            "omitted_inferred_facts": initial.omitted_inferred_facts,
            "edited_value_roundtrip": "2.50",
            "added_triples": edited.added_triples,
            "removed_triples": edited.removed_triples,
            "identity_and_annotations_preserved": restored.annotations
            == loaded.annotations,
        },
        indent=2,
    )
)
