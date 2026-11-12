# Domain refinements — 0.20.0

Refinements add bounded technique-specific requirements to an existing domain. They do not replace its vocabulary, introduce a world dependency or count as additional primitive domains.

`specialized-tool-interactions.ttl` specializes ToolUseDescription for cutting, wiping, maintained pressing and insertion-to-seat. See **SPECIALIZED_TOOL_INTERACTIONS.md** for the exact supported profiles and their limitations. General positioning, engagement, generic functional use and disengagement remain owned by `../domains/tool-interaction.ttl`.

Recipes in `../compositions/action-recipes.ttl` may refer to these refined descriptions. A technique's outcome is not inferred from merely traversing a recipe stage or making contact.
