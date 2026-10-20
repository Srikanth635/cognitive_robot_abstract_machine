---
generated:
  by: repository-knowledge-pilot/0.1.0
  at: '2026-10-20T05:56:43.055902+00:00'
sources:
- id: spatial
  resource: https://github.com/cram2/cognitive_robot_abstract_machine/blob/7b27424de7768798dc561f447e11c2a8eb86c307/semantic_digital_twin/doc/style_guide.md#L1-L158
type: Convention
title: Spatial notation
description: Frame names, homogeneous coordinates and transformation composition.
status: draft
cram:
  source_commit: 7b27424de7768798dc561f447e11c2a8eb86c307
  depends_on: []
  replacement: ''
  evidence:
  - source_id: spatial
    path: semantic_digital_twin/doc/style_guide.md
    start_line: 1
    end_line: 158
    content_digest: 35b526dce0ebf7b545706b067ed96060ab93c4fd072a2cfeadb8643ebd070cf8
    symbol: ''
---
# Spatial notation

The spatial style guide names transformations `root_T_tip`, rotations `root_R_tip`, points
`root_P_tip` and vectors `root_V_tip`. Points use homogeneous coordinate one and vectors use zero.
Composition matches inner frames: `a_T_c = a_T_b @ b_T_c`.[^spatial]

The guide distinguishes inversion of a full transformation from transposition: use `inverse_frame`
for the former; transposition is an inverse for rotation matrices. It also requires matching
reference frames when subtracting points.[^spatial]

Consult the current spatial guide before generating spatial types or connection code. This short
entry is a route to the source convention, rather than a complete geometry reference.

[^spatial]: Spatial style guide
