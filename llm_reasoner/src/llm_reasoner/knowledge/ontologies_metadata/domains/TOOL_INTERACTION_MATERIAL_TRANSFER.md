# Tool interaction and material transfer — 0.17.0

> Updated in 0.22.0: see [PRIMARY_ACTION_GENERATION_GUIDE.md](PRIMARY_ACTION_GENERATION_GUIDE.md) and `reports/release-0.22.0.json`. The dated development history and counts below retain their original scope.

These two world-independent domain modules extend the shared action-description core. They describe requirements and proposed knowledge rather than concrete robot events. DUL is inherited through the core; no execution-framework or live-world dependency is introduced.

## Tool interaction

`tool-interaction.ttl` specializes an abstract tool interaction into four bounded profiles:

| Profile | Required success facts in PostState |
|---|---|
| Positioning | Working part aligned with intended target |
| Engagement | Operator holds the tool; working part contacts the target |
| Functional use | Operator holds the tool; tool is functional; intended effect is observed at the target |
| Disengagement | Tool clear of target; working part no longer contacts target |

The description identifies operator, tool, working part, target, intended effect and reference frame, plus a context scoped to that action description. The working part must belong to the same tool. Required/forbidden/unspecified contact policies are explicit. Functional use is not inferred from contact alone. The current engagement/disengagement fixtures are contact-oriented specializations; other interaction modes can be added without forcing all tool use into contact.

An optional force-limit parameter uses newtons. The supplied demo's 8 N is an invented number, not an operating recommendation, measured tolerance or certified safe limit.

The four canonical profile kinds are ActionKind individuals, while the description specializations are OWL classes. Entity specifications denote descriptive roles, not physically instantiated world objects. Abstract capabilities remain to be matched to injected robot resources.

## Material transfer

`material-transfer.ttl` reuses existing transfer contracts rather than defining duplicate source/destination/amount properties. Profiles are partial transfer, complete transfer and pouring. The contract identifies source, destination, material specification, transferred portion, mode, delivered/target amounts and spillage/tolerance amounts.

All profiles require three positive success conjuncts: delivered amount >= target amount, spillage <= tolerance, and the transferred portion contained in the intended destination. Complete transfer additionally requires residual source amount <= an explicit residual tolerance. Partial transfer does not require an empty source. Pouring uses the partial profile in this release's bounded checker; complete pouring can be added as an explicit composition/profile when needed.

Quantity basis is explicitly volume or mass. The bounded reference checker supports canonical cubic metres or kilograms with exact matching units. It performs no unit conversion. These are authoring/evaluation constraints in SHACL and Python; OWL alone does not perform inequality arithmetic or assume missing facts are false.

This first material-transfer domain describes container-to-container quantities. It does not yet specify all modes of transfer such as scraping a surface, thermal transfer, chemical reaction, compressible gas transfer or mixing with conversion. Conservation/model assumptions remain separately stated. Generalizing those cases requires additional profiles, not treating this fixture as a universal physics model.

## Composition and comprehensive record

`integrations/pouring-tool-bridge.ttl` optionally links a pouring description to a ToolUseDescription. The bounded checker requires that the tool identity equal the source and the tool target equal the transfer destination. Material transfer itself remains independent of tool interaction. A free-pour example does not require bottle/glass contact.

`examples/tool-material-transfer-profiles.ttl` supplies seven synthetic Proposed records. Parameters remain unspecified there. `demo/comprehensive-pouring.ttl` adds a populated fictional record, with one root and connected descriptions for entities, material transfer, tool use, motion parameters, phases, monitoring, causality, physical models, control, objectives, capabilities, strategy, diagnosis, recovery and qualitative perspectives.

All linked descriptive non-type/non-label statements have explicit Proposed claims and synthetic evidence. Child tool-use and root pouring evaluation contexts are separately scoped; no accepted assessment, actual observation, runtime binding or controller is created. The JSON is a readable illustration of the linked record, not a new general projection implementation.

## Validation

Run from the package root after installing requirements:

```bash
python tools/build_comprehensive_demo.py
python tools/validate_tool_material_transfer.py
python tools/reason_tool_material_transfer.py
python tools/validate_imports.py
python tools/review_competency.py
```

The shape/integrity checks cover identity/context bindings, success expression requirements, contact policy, explicit units, finite nonnegative quantities, residual-source semantics, three-valued missing evidence, negative cases and exact statement coverage. The demo's hypothetical accounting is checked separately from transfer success. Neither test proves real physical feasibility or execution readiness. Reasoning reports distinguish generic classification/consistency from the optional AICOR/CRAM check.
