# Reusable compositions — 0.20.0

Compositions combine existing domain requirements while preserving shared entity identities, component contexts and evidence boundaries. They are distinct from primitive reusable action domains and from executable plans. There are seven existing domains and two maintained composition ontologies in this release.

`transport-carrying.ttl` is the first composition: maintained holding plus mobility over an explicit common evaluation window, with load support, loaded-envelope and load-arrival requirements. Four synthetic records exercise destination/route carrying with and without an orientation constraint. Read **TRANSPORT_CARRYING.md** before using the reference inspector.

The ordered recipes now add pickup before carrying and placement afterwards. Supported/cart transport and more complex arrangements remain further extensions. Those require explicit semantics for their sequence, intervals and evidence. They must not inherit grasped carrying's requirements merely because all are informally called transport. The generic action core, shared aspects and individual domains remain reusable independently.

Open `../integrations/all-action-knowledge.ttl` for the combined assembly, or load this module alone through the local catalog. `../integrations/all-action-aspects.ttl` remains independent of domains and compositions.

`action-recipes.ttl` adds explicit ordered stage occurrences, separate stage contexts and qualified handoffs. Initial pickup–carry–place and acquire–use–return cycles are documented in **ACTION_RECIPES.md**. These are bounded recipe descriptions and inspectors, not executable plans.
