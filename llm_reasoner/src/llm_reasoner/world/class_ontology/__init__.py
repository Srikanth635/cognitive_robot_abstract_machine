"""
The ontology of a code base: the classes a caller selects, written as OWL with what each
says of itself, its documentation, the classes it specialises, and its fields with their
documentation, types and defaults.

It reads live classes, as the world injector does, and names no package itself: a
selection file names the packages to import, the classes whose subclasses are described,
the packages whose data structures are described in full, and the bases whose fields are
bookkeeping. Write the ontology a selection asks for with ``python -m
llm_reasoner.world.class_ontology <selection.json> <ontology.ttl>``.
"""
