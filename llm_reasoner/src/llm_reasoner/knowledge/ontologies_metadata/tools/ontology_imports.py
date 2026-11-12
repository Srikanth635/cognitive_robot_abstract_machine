"""Strict offline import resolution from the package's ontology registry."""
from pathlib import Path
import json
from rdflib import Graph,URIRef,OWL,RDF
ROOT=Path(__file__).resolve().parents[1]
REGISTRY=json.loads((ROOT/'ontology-registry.json').read_text())
def closure(iri):
 seen=set();todo=[str(iri)];g=Graph()
 while todo:
  current=todo.pop()
  if current in seen:continue
  if current not in REGISTRY:raise ValueError('Unresolved local ontology import: '+current)
  part=Graph().parse(ROOT/REGISTRY[current])
  if (URIRef(current),RDF.type,OWL.Ontology) not in part:raise ValueError('Registry ontology mismatch: '+current)
  seen.add(current);g+=part;todo.extend(str(o) for o in part.objects(URIRef(current),OWL.imports))
 return g,sorted(seen)


def aspect_file(filename):
 """Resolve one uniquely named aspect ontology or validation shape locally."""
 candidates=list((ROOT/'aspects').rglob(filename))
 if len(candidates)!=1:raise ValueError('Expected one aspect file: '+filename)
 return candidates[0]
