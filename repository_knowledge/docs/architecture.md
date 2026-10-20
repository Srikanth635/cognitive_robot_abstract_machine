# Architecture decisions

The implementation follows [the phased plan](plan.md), based on main at `7b27424de7768798dc561f447e11c2a8eb86c307`.

1. Keep models and abstract contracts independent of concrete Git, AST, SQLite, JSON and command-line implementations. Operations use the contracts. Composition selects adapters.
2. Reuse KRROOD's dataclass serializer, exception base and compatible source-extraction facilities. Its dependencies remain explicit; source models do not inherit runtime symbol registration.
3. The initial tool reads a specified committed revision. It never treats a branch name or working-tree timestamp as evidence of freshness.
4. Track inventory exclusions, parser failures and unresolved references. A missing result is not proof that no caller exists.
5. Store the wiki and implementation under this parent directory. Development records are outside the wiki. Cache and environment directories are ignored.
6. The user approved the new package's existing CRAM dependencies `krrood` and `typing-extensions`.
7. Static indexing compiles/parses source without importing indexed modules. Use the standard-library compiler's reported result for syntax validation; no local exception-handler exemption is required.
8. Normal tests run in an isolated package suite with an explicit `--orm-build never` contract. They reuse pytest fixtures and repository-local datasets. No ORM build or live external service is needed.
9. Root UV configuration is rejected by the installed UV version. The focused environment uses Python 3.12 and pip editable installs; unrelated root configuration remains outside this change.
10. Keep this developer tool separately installable, like basstler. The existing CI robotics environment supplies approved CRAM dependencies; its dedicated matrix entry installs the tool without dependency resolution and runs the isolated package suite.
11. KRROOD remains unchanged. The JSON adapter uses explicit subclass envelopes for nested records and delegates field serialization to the existing dataclass and enum serializers. Only an imported whitelist of evidence types is accepted on decoding.
12. A CLI query captures one completed index before checking its requested commit. Freshness validation and navigation share that captured graph.
13. Source exclusions are versioned policy. Format/extractor version and the configured Python version contribute to snapshot identity; changing the policy requires a version bump and cache rebuild. Phase 4 introduces format 2; the earlier symbol-only format remains identifiable and is refused for refresh.
14. Persistence can return a coherent evidence selection or the complete graph. Operations retain query ordering, pagination and evidence rules. SQLite narrows records before decoding through parameterized statements; its JSON functions and Unicode case-folding preserve the existing search contract.
15. A working-tree overlay binds checkout identity, actual HEAD, Git index, selected contents and untracked inclusion policy. It is stored separately from committed evidence. Current-source queries validate that identity before and after navigation.
16. Original file analysis is retained independently of resolved relationships and compressed using the standard library at the storage boundary. Refresh reuses matching contents/import context and resolves all targets conservatively. No new dependencies or KRROOD changes are required.
17. Scope validation and replacement share the SQLite write transaction. Failed publication retains the previous completed graph. No-change refresh avoids both analysis decoding and publication; reconciliation deliberately rebuilds and republishes.
18. Curated concepts and their evidence checks are distinct from extracted source facts. OKF encoding lives in an adapter; no YAML dependency enters models or operations. The user approved the existing CRAM dependency PyYAML for this boundary.
19. Format findings, local-policy findings and source consistency are reported separately. Portable verification events never authenticate themselves. The drafting boundary cannot promote stable concepts; independent review/publication remains phase 7.
20. The wiki interface reads a source/declaration projection from one completed SQLite snapshot and uses the same freshness operation as structural queries. Supported concept links propagate invalidation conservatively. Export and lifecycle proposals write only to a new directory.
21. Nested metadata dictionaries use typed mapping entries at the KRROOD boundary. Owned field serialization stays delegated to DataclassJSONSerializer, arbitrary serialized classes remain refused, and KRROOD is not modified.
22. Wiki content comparison reports an explicit policy. Uniform UTF-8 LF/CRLF equivalents are accepted only when both actual byte digests match reconstructed representations of the same text; affected paths are reported. Exact-byte comparison remains selectable through the operation. Source-index snapshot identity and freshness remain byte-exact.
23. Task evidence records live in an inner domain module. Test discovery depends on the abstract `TestAnalyzer`; pytest interpretation remains an adapter. Context and change operations consume captured graphs and domain values without importing Git, SQLite, YAML or CLI implementations.
24. Task queries pin one resolved graph projection without reusable raw analyses. Working-source freshness is checked around discovery and assembly. Change assessment additionally pins an explicitly selected committed baseline from the same repository history.
25. Context budgets count text characters and preserve whole evidence items, exact citations, partial-excerpt counts and omissions. Confidence and package relevance precede lexical test ordering. Static collection, proposed test coverage, curated-prose accuracy and authenticated review remain separate claims.


26. Publication records and gates depend on explicit source-identity, review, captured-knowledge and digest abstractions. GitHub, HTTPS, archive and policy parsing stay in adapters. The default fork policy requires machine validation without human approval; optional independent review remains distinct.
27. Canonical admission uses a clean externally pinned checker and its own KRROOD source, two fresh extractions, and captured committed wiki documents. Proposal code, caches and policy cannot supply executable authority. The selected runtime image is separately digest-pinned.
28. Consumers bind an explicit artifact ID, successful pinned publisher run, archive digest, policy, source and complete bundle. Refusal returns source-only context; historical consumption selects an entire code-evidence/wiki pair. Phase 8 will connect that result to agent clients.
