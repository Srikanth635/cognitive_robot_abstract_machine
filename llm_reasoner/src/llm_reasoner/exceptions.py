"""Root exception hierarchy.

Only the common base lives here. Each half raises its own subclasses, in its own
module, because the failure vocabularies have nothing to do with one another:
resolution fails over unresolvable slots and missing referents, generation fails
over hypotheses that violate acceptance constraints. A generation failure
inheriting from ``LanguageModelBackendError`` would claim a kinship that does
not exist and would be caught by handlers meant for something else.
"""


class LLMReasonerError(Exception):
    """Base for everything this package raises."""