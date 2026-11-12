# Module architecture — 0.22.0

Current increment: [primary-action generation and consistency guide](domains/PRIMARY_ACTION_GENERATION_GUIDE.md). Release 0.22.0 strengthens pouring, pickup, placement and navigation within three existing domain ontologies and their shapes. No ontology modules or imports were added; the registry still contains 40 documents. Other missing CoraPlex actions remain deferred. Verification is indexed in `reports/release-0.22.0.json`.

The [0.21.0 coverage review](review/coverage-0.21.0/REVIEW_REPORT.md) remains a dated architectural assessment; its recommendation to add action families is deferred by the user's current four-action scope.

| Layer | Registered ontology documents |
|---|---:|
| DUL foundation | 1 |
| Action-description core and provenance/context | 2 |
| Generic aspect/contract modules | 17 |
| Optional semantic/composition bridges | 6 |
| Reusable action domains | 7 |
| Reusable composition modules | 2 |
| Domain refinement modules | 1 |
| Generic-aspect and combined-knowledge assemblies | 2 |
| Optional unified AICOR and CRAM bridge | 2 |
| Total | 40 |

Aspect modules remain in seven responsibility folders with shared SHACL validation files; see **aspects/README.md**. Shapes, examples, catalogs, reports and preserved/offline convenience copies are not additional maintained ontology modules.

| Domain | Direct imports |
|---|---|
| Robot motion | Conditions/outcomes, monitoring requirements, model contracts |
| Object manipulation | Robot motion, phases/transitions, monitoring requirements, knowledge context |
| Mobility/navigation | Robot motion, phases/transitions, monitoring requirements, knowledge context |
| Perception/attention | Robot motion, sensorimotor events, phases/transitions, monitoring requirements, knowledge context |
| Tool interaction | Robot motion, monitoring requirements, knowledge context |
| Material transfer | Transfer contracts, monitoring requirements, knowledge context |
| Articulated mechanisms | Object manipulation, monitoring requirements, knowledge context |

DUL and the core are imported transitively. Material transfer does not require tool interaction: **integrations/pouring-tool-bridge.ttl** adds an optional link between a pouring description and a tool-use description. This composition neither equates all tools with transfer actions nor requires every transfer to use a tool. The comprehensive demo shares its bottle and destination identities across that link.

**all-action-aspects.ttl** assembles the seventeen aspects/contracts, five aspect bridges and context. **all-action-knowledge.ttl** adds seven reusable domains, transport/carrying, ordered recipes, specialized tool refinements and the pouring/tool bridge. No domain imports CoraPlex, SDT, ROS, a robot model or live world. The optional CRAM/AICOR bridge remains separate.

Fresh 0.22.0 classification passed for object manipulation, material transfer, mobility/navigation and the combined assembly, with no unsatisfiable named classes. Combined generic examples and the demo passed consistency. See **reports/primary-action-reasoning.json**. Optional AICOR reasoning was not rerun; its earlier normalization error and timeout remain unresolved.

The comprehensive record is a connected description with statement provenance. Its JSON is a one-off readable view, not a general executable/frame/physics projection system. Working projections remain subsequent work.

Articulated mechanisms reuse joint-coordinate/frame/parameter structures, condition expressions and phase monitors. The domain supports bounded revolute/prismatic configuration, opening/closing and explicit latch release/securing profiles; it adds no framework dependency. See **domains/ARTICULATED_MECHANISMS.md**.

**compositions/transport-carrying.ttl** imports object manipulation, mobility/navigation and context. It composes separately scoped holding/mobility requirements under a common evidence interval. It is the first composition module, not an eighth primitive domain or a new scheduler. See **compositions/TRANSPORT_CARRYING.md**.

**compositions/action-recipes.ttl** imports transport/carrying and **refinements/specialized-tool-interactions.ttl**. The refinement imports generic tool interaction and context; it has no recipe dependency. Recipes reference qualified stage conditions and handoffs rather than flattening historical states or executing stages. See the two module documentation files for the initial bounded profiles.
