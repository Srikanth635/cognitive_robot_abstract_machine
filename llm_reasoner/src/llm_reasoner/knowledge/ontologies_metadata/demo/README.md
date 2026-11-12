# Comprehensive pouring illustration

`comprehensive-pouring.lisp` is an abbreviated Lisp-style reading aid (not executable syntax). `comprehensive-pouring.ttl` is the linked RDF knowledge record; `comprehensive-pouring.json` is its one-off readable authoring view. JSON summaries abbreviate the RDF graph; they are not a lossless serialization of every axiom or claim. Rebuild both using `python tools/build_comprehensive_demo.py` from the package root. The builder clones a prior rich pouring fixture and composes the new material-transfer and tool-use descriptions while preserving shared entity identities. Original examples are unchanged.

All values and assumed outcomes are deliberately fictional. Every descriptive claim is Proposed with CuratedExample evidence. Outcome quantities are hypothetical inputs, not observations. No actual robot, world frame, detected object, actuator binding or accepted assessment exists.

| Parameter | Fictional value |
|---|---:|
| Initial source water | 600 mL |
| Requested delivered volume | 250 mL |
| Assumed delivered volume | 252 mL |
| Assumed spilled volume | 1 mL |
| Spillage tolerance | 5 mL |
| Assumed source remainder | 347 mL |
| Destination capacity | 400 mL |
| Bottle dry mass | 0.42 kg |
| Density | 998 kg/m³ |
| Viscosity | 0.0011 Pa·s |
| Pour position in an unbound demo frame | (0.62, -0.18, 0.94) m |
| Tilt | 1.15 rad |
| Angular speed bound | 0.35 rad/s |
| Acceleration bound | 0.4 m/s² |
| Hypothetical force limit | 8 N |
| Sampling period | 0.02 s |
| Retry budget | 2 |

RDF uses cubic metres for volume. Displayed mL values are exact conversions. Toy accounting balances: 600 = 347 + 252 + 1 mL. This depends on the explicit incompressible, fully accounted losses approximation and is not a fluid simulation or empirical validation. Transfer success also requires the assumed correct-destination containment fact; arithmetic alone is insufficient.

The bottle is both the transfer source and the tool; its mouth is the working part. The destination glass is also the tool target. The material specification and transferred portion are separate referents. Parent pouring and child tool-use have distinct scoped contexts. Future contextual acceptance needs appropriate per-context records/assessments; it is not implied by sharing a root snapshot.

The linked record includes phase structure, monitoring and unknown-evidence policy, causal/mechanistic hypotheses, model contracts and volume accounting, partial dynamics/control descriptions, objectives, abstract capabilities/strategy, diagnostic candidates/recovery, frame schema references, qualitative force dynamics and image schemas. The JSON summarizes those links without pretending every referenced model is complete.

State evolution equations, output equations, calibrated actuator models and resource matches remain missing. Causal probabilities and force-strength estimates remain unknown. No controller is synthesized and the CoraPlex executable projection is null. A hypothetical 8 N force bound establishes no safety claim. Frame slots in JSON are an illustrative summary, not a generated working frame projection.
