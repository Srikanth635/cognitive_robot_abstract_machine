"""
Activate the sample hypothesis and query it the way the viewer's server process does,
outside pytest, whose fixtures would otherwise build the symbol graph for it.

Prints the number of phases the query finds.
"""

from cramera.resolution import hypothesis_knowledge, kb

kb.build_symbol_graph()
hypothesis_knowledge.activate_sample()
kb.reset_kb()
print(kb.run_query("an(entity(pouring_phase))")["count"])
