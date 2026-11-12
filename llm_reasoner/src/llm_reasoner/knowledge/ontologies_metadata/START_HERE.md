# Action knowledge ontologies — release 0.22.0

Current increment: [primary-action generation and consistency guide](domains/PRIMARY_ACTION_GENERATION_GUIDE.md). Release 0.22.0 strengthens pouring, pickup, placement and navigation within three existing domain ontologies and their shapes. No ontology modules or imports were added; the registry still contains 40 documents. Other missing CoraPlex actions remain deferred. Verification is indexed in `reports/release-0.22.0.json`.

The [0.21.0 coverage review](review/coverage-0.21.0/REVIEW_REPORT.md) remains a dated architectural assessment; its recommendation to add action families is deferred by the user's current four-action scope.

Seven reusable domains are available: robot motion, object manipulation, mobility/navigation, perception/attention, tool interaction, material transfer and articulated mechanisms.

Introduced in 0.20.0: **compositions/ACTION_RECIPES.md** describes ordered pickup–carry–place and acquire–use–return tool cycles. **refinements/SPECIALIZED_TOOL_INTERACTIONS.md** describes cutting, wiping, maintained pressing and insertion-to-seat. There are two composition modules and one tool-refinement module alongside the seven domains.

The first reusable composition is transport/carrying: **compositions/TRANSPORT_CARRYING.md** describes four variants built from maintained holding and navigation, with explicit coextensive evidence windows, loaded-envelope and load-arrival requirements.

Read **domains/ARTICULATED_MECHANISMS.md** for the seventh domain and its eight synthetic configuration/opening/closing/latch profiles. See **domains/TOOL_INTERACTION_MATERIAL_TRANSFER.md** for the preceding two domains and their seven authoring profiles. **demo/README.md** introduces the comprehensive fictional pouring record; **demo/comprehensive-pouring.json** is its readable authoring view and **demo/comprehensive-pouring.ttl** contains the linked RDF knowledge and per-statement provenance.

Open **integrations/all-action-knowledge.ttl** for all aspects, seven domains, both compositions, specialized tool refinements and the optional pouring/tool bridge. Individual domains can be loaded independently. **integrations/all-action-aspects.ttl** remains the generic aspect assembly. Local catalogs resolve imports offline. There are 40 registered ontology documents; aspect folders remain unchanged.

The supplied domain/recipe records and numerical examples are explicitly synthetic Proposed knowledge. They establish no measured outcome, safe operating limit, calibrated controller or executable action. The comprehensive example uses shared entity identities across its linked knowledge aspects; it does not introduce a working projection engine.

The three changed domains and combined assembly were freshly classified in 0.22.0 with no unsatisfiable named classes; combined generic knowledge, examples and the fictional demo passed consistency. The optional AICOR/CRAM check was not rerun: its earlier timeout and SceneNode normalization issue remain unresolved. Original AICOR axioms are unchanged.

Install **requirements.txt**; Java is required for HermiT. Current validation is indexed in **reports/release-0.22.0.json**. Extract the cumulative archive into a fresh directory.
