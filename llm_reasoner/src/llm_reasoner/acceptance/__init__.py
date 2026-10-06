"""
Acceptance rules: what makes a generated hypothesis admissible.

The rules read a hypothesis as a closed world, where a missing value is an error rather
than an open question, which is what a language model that generates too little needs to
be told. :mod:`~llm_reasoner.acceptance.derived` checks the restrictions the ontology
states; the remaining rules state relationships between several fragments that a
restriction cannot. Open-world contradictions with CRAM-AICOR and DUL are left to the
reasoner.
"""
