// Finished slide bodies, one function per slide id, drawn with the layouts' helpers as
// native, editable shapes. deck.js calls the body registered for a slide instead of the
// skeleton. Each body takes the deck (for its shape types and scheme colours), the slide
// and the outline entry. Every figure drawn here is traced to the record in outline.js.

const L = require("./layouts.js");
const K = L.K, TINT = L.TINT, FONT = L.FONT, SIZE = L.SIZE, W = L.W;
const COMPONENT = L.theme.components;
const GRADE = L.theme.evidence_grades;

function text(slide, C, content, x, y, w, h, options, objectName) {
  slide.addText(content, Object.assign({ x, y, w, h, isTextBox: true, margin: 0, fontSize: 11, color: C.text1, valign: "top", objectName }, options || {}));
}

// %% slide 3: CRAM's packages, and where llm_reasoner sits
function knowledgeExecutive(pres, C, s) {
  const x = 0.5, w = W - 1.0;
  // plans
  L.card(pres, s, x, 1.45, w, 0.72, K.navy, K.background, "plans card");
  text(s, C, [{ text: "Plans: coraplex  ", options: { bold: true, color: K.navy, fontFace: FONT.heading, fontSize: 14 } }, { text: "interprets and executes high-level action plans in the CRAM plan language; when a plan resolves the parameters of an action it asks the Knowledge Executive", options: { color: C.text1 } }], x + 0.25, 1.45, w - 0.5, 0.72, { valign: "middle" }, "plans text");
  L.arrow(pres, s, x + 1.6, 2.17, x + 1.6, 2.55, K.navy, "plans to krrood");
  L.pill(pres, s, "resolving parameters", x + 1.75, 2.2, 1.7, 0.3, K.navy, "resolving pill");

  // the krrood layer with its backends
  const ky = 2.5, kh = 2.7;
  s.addShape(pres.ShapeType.roundRect, { x, y: ky, w, h: kh, rectRadius: 0.15, fill: { color: TINT.blue }, line: { color: K.navy, width: 1.5, dashType: "dash" }, objectName: "krrood layer" });
  L.pill(pres, s, "krrood layer", x + 0.25, ky - 0.15, 1.2, 0.3, K.navy, "krrood pill");
  text(s, C, "Symbolic knowledge representation: object-oriented classes, EQL queries and rule-based reasoning; the symbol graph is the live world", x + 1.6, ky + 0.12, w - 1.85, 0.3, { fontSize: 10.5, color: C.text2, italic: true }, "krrood text");
  const backends = [
    { title: "Probabilistic models", kicker: "probabilistic_model, random_events", colour: K.muted, lines: ["Resolve parameters from probabilistic models learned from data.", "Answer inside what the data covered; a value without its reasons."] },
    { title: "Backends per action", kicker: "resolution code written for one action", colour: K.muted, lines: ["Resolve what somebody wrote code for.", "A new object, world or action means new code."] },
    { title: "llm_reasoner", kicker: "generative backend, this talk", colour: K.navy, fill: K.background, lines: ["A language model's general understanding of the action, bound to this world's facts.", "Every answer with a why and checked grounds; the ontology and physics keep it honest."] },
  ];
  L.columns(pres, s, C, backends, x + 0.25, ky + 0.5, w - 0.5, kh - 0.7, "backends");

  // the world, motion and the robot
  const by = 5.35, bh = 0.72;
  const bw = (w - 0.6) / 3;
  const below = [
    { title: "Semantic Digital Twin", text: "the world: sensor data, robot models, semantic annotations", colour: K.teal },
    { title: "Giskardpy", text: "whole-body motion control", colour: K.teal },
    { title: "Robot or simulation", text: "PR2 in the apartment; MuJoCo, Isaac Sim", colour: K.teal },
  ];
  below.forEach((b, i) => {
    const bx = x + i * (bw + 0.3);
    L.card(pres, s, bx, by, bw, bh, b.colour, K.background, `below ${i} card`);
    text(s, C, [{ text: b.title + "  ", options: { bold: true, color: b.colour, fontFace: FONT.heading, fontSize: 13 } }, { text: b.text, options: { color: C.text2, fontSize: 10.5 } }], bx + 0.2, by, bw - 0.4, bh, { valign: "middle" }, `below ${i} text`);
  });
  L.arrow(pres, s, x + bw / 2, by, x + bw / 2, ky + kh, K.teal, "world to krrood");
  L.pill(pres, s, "symbol graph", x + bw / 2 + 0.1, ky + kh + 0.0, 1.1, 0.26, K.teal, "symbol graph pill");
  L.band(pres, s, "Placement", "llm_reasoner belongs to the krrood layer: it imports krrood and never a world model, a robot or a plan package.", x, 6.2, w, 0.5, "placement band");
}

// %% slide 9: one resolution, step by step
function flow(pres, C, s) {
  const items = [
    { title: "The query", colour: K.ink, text: "The caller passes a krrood query: the slots it fixes, the slots it leaves open, and optionally a task in words." },
    { title: "Knowledge first", colour: COMPONENT.knowledge, text: "Before any request to its model, the resolver consults the knowledge component: the hypothesis for this action and situation, served from the store or generated once." },
    { title: "The request", colour: COMPONENT.resolver, text: "The action's source and docstrings, each open slot with its meaning and candidates, the given values, the task and the answer's schema." },
    { title: "Model with tools", colour: COMPONENT.language_model, text: "The model reads the world through eight world tools and the knowledge through three knowledge tools, then answers JSON: every slot, a why, its grounds." },
    { title: "Checks", colour: COMPONENT.resolver, text: "Every ground is checked against its source; an answer breaking a where-condition is repaired with the broken conditions named." },
    { title: "The trial", colour: COMPONENT.simulation, text: "Before a pouring resolution is accepted, the pour is tried in the simulation; a miss goes back as structured data, up to three trials." },
  ];
  const x = 0.5, y = 1.55, w = W - 1.0, h = 2.6;
  const cw = L.steps(pres, s, C, items, x, y, w, h, "step");
  const gap = (w - cw * items.length) / (items.length - 1);
  const centre = (i) => x + i * (cw + gap) + cw / 2;

  // the repair edge: checks back to the model
  const r1 = y + h + 0.35;
  s.addShape(pres.ShapeType.line, { x: centre(4), y: y + h, w: 0, h: 0.35, line: { color: COMPONENT.resolver, width: 1.25 }, objectName: "repair down" });
  s.addShape(pres.ShapeType.line, { x: centre(3), y: r1, w: centre(4) - centre(3), h: 0, line: { color: COMPONENT.resolver, width: 1.25 }, objectName: "repair across" });
  L.arrow(pres, s, centre(3), r1, centre(3), y + h + 0.02, COMPONENT.resolver, "repair up");
  L.pill(pres, s, "repair: the broken conditions named", centre(3) + 0.15, r1 - 0.15, 2.3, 0.3, COMPONENT.resolver, "repair pill");

  // the retrial edge: the trial back to the request
  const r2 = y + h + 0.95;
  s.addShape(pres.ShapeType.line, { x: centre(5), y: y + h, w: 0, h: 0.95, line: { color: COMPONENT.simulation, width: 1.25 }, objectName: "retrial down" });
  s.addShape(pres.ShapeType.line, { x: centre(2), y: r2, w: centre(5) - centre(2), h: 0, line: { color: COMPONENT.simulation, width: 1.25 }, objectName: "retrial across" });
  L.arrow(pres, s, centre(2), r2, centre(2), y + h + 0.02, COMPONENT.simulation, "retrial up");
  L.pill(pres, s, "a pour that missed goes back as structured data, up to three trials", centre(2) + 0.15, r2 - 0.15, 4.4, 0.3, COMPONENT.simulation, "retrial pill");

  // the legend of component colours
  const legend = [["resolver", "Resolver"], ["knowledge", "Knowledge component"], ["language_model", "Language model"], ["simulation", "Simulation"]];
  legend.forEach(([key, name], i) => {
    const lx = x + i * 2.2;
    s.addShape(pres.ShapeType.ellipse, { x: lx, y: 5.62, w: 0.18, h: 0.18, fill: { color: COMPONENT[key] }, line: { color: COMPONENT[key], width: 0 }, objectName: `legend ${key} dot` });
    text(s, C, name, lx + 0.26, 5.56, 1.9, 0.3, { fontSize: 10.5, valign: "middle" }, `legend ${key}`);
  });
  L.band(pres, s, "The answer", "a value for every slot, a one-sentence why, and grounds: a world handle or a knowledge fragment, each checked.", x, 6.05, w, 0.55, "answer band");
}

// %% slide 13: the ontology in two forms
function ontology(pres, C, s) {
  const lx = 0.5, lw = 6.6;
  const layers = [
    { y: 5.3, h: 0.6, title: "DUL", text: "the foundational ontology", colour: K.muted, fill: TINT.grey },
    { y: 4.55, h: 0.6, title: "CRAM-AICOR", text: "cram-aicor-ontology.ttl: 1,069 classes, 811 object properties; roles, frames, force dynamics", colour: K.muted, fill: TINT.grey },
  ];
  layers.forEach((l, i) => {
    L.card(pres, s, lx, l.y, lw, l.h, l.colour, l.fill, `layer ${i}`);
    text(s, C, [{ text: l.title + "  ", options: { bold: true, color: K.navy, fontFace: FONT.heading, fontSize: 13 } }, { text: l.text, options: { color: C.text2, fontSize: 10.5 } }], lx + 0.25, l.y, lw - 0.5, l.h, { valign: "middle" }, `layer ${i} text`);
  });
  // our modules: two families side by side
  const my = 2.75, mh = 1.65;
  s.addShape(pres.ShapeType.roundRect, { x: lx, y: my, w: lw, h: mh, rectRadius: 0.12, fill: { color: TINT.blue }, line: { color: K.navy, width: 1.5, dashType: "dash" }, objectName: "modules frame" });
  L.pill(pres, s, "our modules", lx + 0.2, my - 0.15, 1.1, 0.3, K.navy, "modules pill");
  const fam = [
    { title: "Action perspectives", lines: ["action-core, action-frame,", "image-schemas, failures"], colour: K.blue },
    { title: "Predictive manipulation", lines: ["core, object-dynamics,", "prehension, pouring, gaze"], colour: K.purple },
  ];
  fam.forEach((f, i) => {
    const fx = lx + 0.2 + i * (lw / 2);
    const fw = lw / 2 - 0.3;
    L.card(pres, s, fx, my + 0.35, fw, mh - 0.55, f.colour, K.background, `family ${i}`);
    text(s, C, [{ text: f.title, options: { bold: true, color: f.colour, fontFace: FONT.heading, fontSize: 13, breakLine: true } }, { text: f.lines.join("\n"), options: { color: C.text1, fontSize: 10.5, fontFace: FONT.mono } }], fx + 0.2, my + 0.45, fw - 0.4, mh - 0.75, {}, `family ${i} text`);
  });
  // the hypothesis on top
  L.card(pres, s, lx, 1.5, lw, 1.05, K.gold, TINT.gold, "hypothesis card");
  text(s, C, [{ text: "A generated hypothesis  ", options: { bold: true, color: K.gold, fontFace: FONT.heading, fontSize: 13 } }, { text: "instances of the modules' classes: one root structure and the fragments making it up, each stored under an IRI; graded Generated until physics reaches it", options: { color: C.text1, fontSize: 10.5 } }], lx + 0.25, 1.5, lw - 0.5, 1.05, { valign: "middle" }, "hypothesis text");
  text(s, C, "imports", lx + lw + 0.05, 4.95, 0.7, 0.3, { fontSize: 9, italic: true, color: C.text2 }, "imports note");

  // the two forms on the right
  const rx = lx + lw + 0.9, rw = W - 0.5 - rx;
  const forms = [
    { y: 1.5, title: "Working form", colour: K.blue, fill: TINT.blue, kicker: "EQL, CLOSED WORLD", lines: ["Python classes generated from the modules with krrood's code generator and checked in; a test fails when they differ from the ontology.", "Queried with EQL over the knowledge component's own graph."] },
    { y: 3.75, title: "Reasoning form", colour: K.purple, fill: TINT.purple, kicker: "HERMIT, OPEN WORLD", lines: ["The same knowledge written as RDF in the ontology's terms.", "Checked for consistency against the ontology, CRAM-AICOR and DUL; a contradiction is located to the fragments causing it."] },
  ];
  forms.forEach((f, i) => {
    L.card(pres, s, rx, f.y, rw, 2.1, f.colour, f.fill, `form ${i}`);
    text(s, C, f.title, rx + 0.25, f.y + 0.15, rw - 0.5, 0.4, { bold: true, color: f.colour, fontFace: FONT.heading, fontSize: 15, valign: "middle" }, `form ${i} title`);
    text(s, C, f.kicker, rx + 0.25, f.y + 0.55, rw - 0.5, 0.28, { bold: true, color: C.text2, fontSize: 9.5 }, `form ${i} kicker`);
    text(s, C, f.lines.map((t, j) => ({ text: t, options: { breakLine: j < f.lines.length - 1, paraSpaceAfter: 5 } })), rx + 0.25, f.y + 0.88, rw - 0.5, 1.15, { fontSize: 10.5 }, `form ${i} lines`);
    L.arrow(pres, s, lx + lw, my + 0.5 + i * 0.6, rx, f.y + 1.05, f.colour, `modules to form ${i}`);
  });
  L.band(pres, s, "The code's schema too", "cram-classes.ttl, written from the live designator classes: 392 classes, 13 enumerations, in place of a 9.8 MB hand-made file.", 0.5, 6.1, W - 1.0, 0.6, "code schema band");
}

// %% slide 14: generation, acceptance, admission, store, reading
function admission(pres, C, s) {
  const items = [
    { title: "Generation", colour: COMPONENT.language_model, text: "A structured completion shaped by every vocabulary class reachable from the queried type: a flat list of fragments referring to each other by identifier." },
    { title: "Acceptance", colour: K.blue, text: "Closed world: restrictions_hold, ranges_hold, shapes_hold (SHACL across fragments), components_declared. What is missing is an error." },
    { title: "Admission", colour: K.purple, text: "Open world, optional: the hypothesis as RDF checked with HermiT against the ontology, CRAM-AICOR and DUL; the contradicting fragments located." },
    { title: "Store", colour: K.teal, text: "One record per hypothesis: instruction, root type, fixed values, every exchange, warnings, verdict, module versions. Keyed by root type, fixed values and versions." },
    { title: "Reading", colour: K.teal, text: "Served from the store from then on; HypothesisView answers questions across fragments, every fragment cited by its IRI." },
  ];
  const x = 0.5, y = 1.55, w = W - 1.0, h = 2.55;
  const cw = L.steps(pres, s, C, items, x, y, w, h, "stage");
  const gap = (w - cw * items.length) / (items.length - 1);
  const centre = (i) => x + i * (cw + gap) + cw / 2;
  const ry = y + h + 0.4;
  [1, 2].forEach((i) => s.addShape(pres.ShapeType.line, { x: centre(i), y: y + h, w: 0, h: 0.4, line: { color: K.red, width: 1.25 }, objectName: `problems down ${i}` }));
  s.addShape(pres.ShapeType.line, { x: centre(0), y: ry, w: centre(2) - centre(0), h: 0, line: { color: K.red, width: 1.25 }, objectName: "problems across" });
  L.arrow(pres, s, centre(0), ry, centre(0), y + h + 0.02, K.red, "patch up");
  L.pill(pres, s, "problems shown to the model, which answers a patch: up to 2 repairs", centre(0) + 0.15, ry - 0.15, 4.2, 0.3, K.red, "repair pill");

  // the answer format
  L.card(pres, s, x, 5.0, 7.3, 1.1, K.line_light, TINT.grey, "answer format card");
  text(s, C, '{"fragments": [\n  {"id": "pouring", "type": "PouringPhaseStructure", "decomposes_task": "aicor:Pouring", "defines_component": ["tilting", "flow_onset", ...]},\n  {"id": "tilting", "type": "ActionPhase", "initiated_by": "flow_onset", "is_optional": false, ...}]}', x + 0.2, 5.1, 6.9, 0.95, { fontSize: 9, fontFace: FONT.mono, color: K.navy }, "answer format");
  L.card(pres, s, x + 7.6, 5.0, w - 7.6, 1.1, K.line_light, K.background, "served card");
  text(s, C, [{ text: "Once per situation.  ", options: { bold: true, color: K.teal, fontFace: FONT.heading, fontSize: 12.5 } }, { text: "A request with the same root type and fixed values under the current module versions is answered from the store, whatever instruction it was generated for: in the live cases 9 s and $0.04 against about 3 minutes and $0.30–0.41 to generate.", options: { fontSize: 10.5 } }], x + 7.8, 5.0, w - 8.0, 1.1, { valign: "middle" }, "served text");
  L.band(pres, s, "Admitted means", "every fragment meets its classes' restrictions and ranges, the shapes hold across fragments, and, with HermiT, nothing contradicts the ontology.", x, 6.2, w, 0.5, "admitted band");
}

// %% slide 16: Flanagan's phases, their boundary events and forces
function flanagan(pres, C, s) {
  const phases = [
    ["reach", "ReachPhase"], ["contact", "ContactEstablishmentPhase"], ["load", "ForceLoadingPhase"], ["lift", "LiftPhase"], ["transport", "TransportPhase"],
    ["align", "AlignmentPhase"], ["tilt", "TiltToFlowPhase"], ["regulate", "FlowRegulationPhase"], ["terminate", "FlowTerminationPhase"], ["upright", "ReturnUprightPhase"],
    ["transport back", "TransportPhase"], ["replace", "ReplacePhase"], ["unload", "UnloadPhase"], ["release", "ContactReleasePhase"], ["withdraw", "WithdrawPhase"],
  ];
  const events = ["reach done", "grasp contact", "liftoff", "clearance reached", "above bowl", "lip aligned", "flow onset", "volume reached", "flow cessation", "upright restored", "above counter", "touchdown", "forces unloaded", "grasp release", "withdrawal done"];
  const forces = { 1: "starts grip force", 2: "ends support", 6: "flow vs containment", 8: "flow vs containment", 11: "starts support", 12: "ends lifting load", 13: "ends grip force" };
  const groups = [["Prehension (prehension.ttl)", 0, 5, K.teal], ["Pouring (pouring.ttl)", 5, 10, K.blue], ["Replacing (prehension.ttl)", 10, 15, K.teal]];

  const x = 0.5, w = W - 1.0, n = phases.length;
  const pw = (w - 0.05 * (n - 1)) / n;
  const sy = 2.65, sh = 0.62;
  groups.forEach(([name, a, b, colour], i) => {
    const gx = x + a * (pw + 0.05), gw = (b - a) * (pw + 0.05) - 0.05;
    text(s, C, name, gx, 1.5, gw, 0.3, { bold: true, color: colour, fontFace: FONT.heading, fontSize: 12, align: "center", valign: "middle" }, `group ${i}`);
    s.addShape(pres.ShapeType.line, { x: gx, y: 1.82, w: gw, h: 0, line: { color: colour, width: 1.5 }, objectName: `group ${i} rule` });
  });
  phases.forEach(([name], i) => {
    const colour = i < 5 || i >= 10 ? K.teal : K.blue;
    const px = x + i * (pw + 0.05);
    s.addShape(pres.ShapeType.roundRect, { x: px, y: sy, w: pw, h: sh, rectRadius: 0.08, fill: { color: i < 5 || i >= 10 ? TINT.teal : TINT.blue }, line: { color: colour, width: 1.25 }, objectName: `phase ${i}` });
    text(s, C, name, px + 0.03, sy, pw - 0.06, sh, { fontSize: 9.5, bold: true, color: colour, align: "center", valign: "middle" }, `phase ${i} name`);
    // the event ending the phase, as a marker on the right edge, its label above or below
    const ex = px + pw + 0.025 - 0.08;
    const ey = sy + sh / 2 - 0.08;
    s.addShape(pres.ShapeType.ellipse, { x: ex, y: ey, w: 0.16, h: 0.16, fill: { color: K.navy }, line: { color: K.background, width: 1 }, objectName: `event ${i} marker` });
    const above = i % 2 === 0;
    const labelX = Math.min(ex - 0.5, W - 0.5 - 1.16);
    text(s, C, events[i], labelX, above ? sy - 0.6 : sy + sh + 0.08, 1.16, 0.5, { fontSize: 9, color: K.navy, align: "center", valign: above ? "bottom" : "top" }, `event ${i} label`);
    if (forces[i]) {
      const fy = i % 2 ? 3.95 : 4.3;
      L.pill(pres, s, forces[i], ex - 0.6, fy, 1.36, 0.28, K.gold, `force ${i}`, TINT.gold);
      s.addShape(pres.ShapeType.line, { x: ex + 0.08, y: sy + sh + 0.02, w: 0, h: fy - sy - sh - 0.02, line: { color: K.gold, width: 0.75, dashType: "dash" }, objectName: `force ${i} tie` });
    }
  });
  text(s, C, "A boundary event ends one phase and starts the next; the gold pills name the forces an event starts, ends or opposes", x, 4.63, w, 0.3, { fontSize: 10.5, italic: true, color: C.text2, align: "center" }, "strip caption");

  // the contract's distinctions
  const dist = [
    ["Mechanical event versus sensory event", "they are never one individual because their timing coincides"],
    ["Prediction versus observation", "an expectation before evidence, and the evidence; they may disagree"],
    ["Phase boundary versus within-phase regulation", "a phase ends at a discrete event while its forces are regulated continuously"],
    ["Corrective action versus model update", "one protects this action, the other changes future predictions"],
  ];
  const dw = (w - 0.9) / 4;
  dist.forEach(([head, body], i) => {
    const dx = x + i * (dw + 0.3);
    L.card(pres, s, dx, 5.0, dw, 1.05, K.line_light, K.background, `distinction ${i}`);
    text(s, C, [{ text: head, options: { bold: true, color: K.navy, fontSize: 10.5, breakLine: true } }, { text: body, options: { color: C.text2, fontSize: 9.5 } }], dx + 0.15, 5.08, dw - 0.3, 0.95, {}, `distinction ${i} text`);
  });
  L.band(pres, s, "Evidence status", "every claim is EMPIRICAL_FINDING, SUPPORTED_INTERPRETATION, AUTHOR_HYPOTHESIS, OPEN_QUESTION or ENGINEERING_EXTENSION; the pouring machinery is an engineering extension.", x, 6.15, w, 0.6, "status band");
}

// %% slide 18: evidence grades and a worked example
function grades(pres, C, s) {
  const rows = [
    ["Grade", "Meaning", "Where it lives today"],
    [{ text: "Measured", options: { bold: true, color: GRADE.measured } }, "Observed or measured in a physics simulation, with its setup: a quantity, an event's occurrence, the order of events, a limit", "PhysicsSimulationSetup records (physical_measurements/flow-onset.ttl)"],
    [{ text: "Derived", options: { bold: true, color: GRADE.derived } }, "Computed from geometry or another physical model, with its setup", "GeometricDerivationSetup records"],
    [{ text: "Sourced", options: { bold: true, color: GRADE.sourced } }, "Stated by a source, with the source's own support", "core:SourceClaim, core:claimedIn, core:evidenceStatus"],
    [{ text: "Defined", options: { bold: true, color: GRADE.defined } }, "Follows from the ontology's axioms: a class, a restriction, a CRAM-AICOR or DUL term", "The ontology modules"],
    [{ text: "Generated", options: { bold: true, color: GRADE.generated } }, "Stated by a language model and admitted by the acceptance checks: a hypothesis", "Hypothesis fragments; the record names the model"],
  ];
  L.table(pres, s, C, rows, 0.5, 1.5, 7.6, [1.1, 3.6, 2.9], "grades table", { rowHeight: 0.5 });

  // the worked example: a measured onset against two generated ranges
  const ex = 8.5, ew = W - 0.5 - ex;
  L.card(pres, s, ex, 1.5, ew, 3.0, K.line_light, K.background, "example card");
  text(s, C, "Evidence can contradict a hypothesis", ex + 0.25, 1.62, ew - 0.5, 0.35, { bold: true, color: K.navy, fontFace: FONT.heading, fontSize: 13, valign: "middle" }, "example title");
  const ax = ex + 0.45, aw = ew - 0.9, ay = 3.35;
  const px = (deg) => ax + (deg / 120) * aw;
  s.addShape(pres.ShapeType.line, { x: ax, y: ay, w: aw, h: 0, line: { color: K.line, width: 1.25 }, objectName: "axis" });
  [0, 30, 60, 90, 120].forEach((d) => {
    s.addShape(pres.ShapeType.line, { x: px(d), y: ay - 0.05, w: 0, h: 0.1, line: { color: K.line, width: 1 }, objectName: `tick ${d}` });
    text(s, C, `${d}°`, px(d) - 0.3, ay + 0.08, 0.6, 0.25, { fontSize: 9, color: C.text2, align: "center" }, `tick label ${d}`);
  });
  s.addShape(pres.ShapeType.rect, { x: px(30), y: ay - 0.62, w: px(110) - px(30), h: 0.22, fill: { color: K.teal }, line: { color: K.teal, width: 0 }, objectName: "range supported" });
  text(s, C, "generated 30–110°: supported", px(30), ay - 0.9, px(110) - px(30), 0.25, { fontSize: 9.5, bold: true, color: K.teal }, "supported label");
  s.addShape(pres.ShapeType.rect, { x: px(70), y: ay - 0.32, w: px(110) - px(70), h: 0.22, fill: { color: K.red }, line: { color: K.red, width: 0 }, objectName: "range refuted" });
  text(s, C, "70–110°: refuted", px(70) - 1.4, ay - 0.32, 1.35, 0.22, { fontSize: 9.5, bold: true, color: K.red, align: "right", valign: "middle" }, "refuted label");
  s.addShape(pres.ShapeType.line, { x: px(55.2), y: ay - 0.75, w: 0, h: 1.0, line: { color: K.navy, width: 2 }, objectName: "measured line" });
  text(s, C, "measured 55.2°", px(55.2) - 0.7, ay + 0.3, 1.4, 0.25, { fontSize: 9.5, bold: true, color: K.navy, align: "center" }, "measured label");
  text(s, C, "The pitcher's simulated flow onset at fill 0.5 against two generated flow-onset ranges", ex + 0.25, 3.95, ew - 0.5, 0.5, { fontSize: 9.5, italic: true, color: C.text2 }, "example caption");

  L.card(pres, s, 0.5, 4.7, 7.6, 1.25, K.line_light, TINT.grey, "rule card");
  text(s, C, [
    { text: "A part of an answer is graded by the strongest evidence that bears on it; a summary grade is the weakest part's. ", options: {} },
    { text: "Already today a resolution's grounds, world handles and knowledge fragment IRIs, are checked against their source, and a ground naming nothing is sent back for repair (UnknownGround).", options: { color: C.text2 } },
  ], 0.75, 4.8, 7.1, 1.05, { fontSize: 10.5, valign: "middle" }, "rule text");
  L.card(pres, s, ex, 4.7, ew, 1.25, K.line_light, K.background, "legend card");
  text(s, C, "The grade colours used on every later slide", ex + 0.25, 4.8, ew - 0.5, 0.3, { bold: true, color: K.navy, fontFace: FONT.heading, fontSize: 11.5 }, "legend title");
  L.gradeLegend(pres, s, C, ex + 0.25, 5.2, "legend", 3, 0.32);
  L.band(pres, s, "Provenance", "the questions about where an answer comes from are not a level of their own: every answer answers them, through its grades.", 0.5, 6.15, W - 1.0, 0.5, "provenance band");
}


// %% slide 19: the flow-onset tilt, measured, with the simulation as a clip
function flowOnset(pres, C, s) {
  const rows = [
    ["Container", "Settled fill", "Geometry", "Simulation", "Difference"],
    ["jeroen_cup.stl", "0.244", "77.0\u00b0", "80.2\u00b0", "+3.2\u00b0"],
    ["jeroen_cup.stl", "0.505", "67.9\u00b0", "69.0\u00b0", "+1.1\u00b0"],
    ["jeroen_cup.stl", "0.566", "64.9\u00b0", "65.7\u00b0", "+0.8\u00b0"],
    ["Static_MilkPitcher.stl", "0.292", "63.8\u00b0", "67.6\u00b0", "+3.8\u00b0"],
    ["Static_MilkPitcher.stl", "0.496", "52.2\u00b0", "55.2\u00b0", "+3.0\u00b0"],
    ["Static_MilkPitcher.stl", "0.597", "46.6\u00b0", "49.3\u00b0", "+2.7\u00b0"],
  ];
  L.table(pres, s, C, rows, 0.5, 1.5, 7.0, [2.2, 1.2, 1.2, 1.2, 1.2], "onset table", { rowHeight: 0.36 });
  text(s, C, "Six cases, three runs each, about 11 minutes per container; three runs of a case are identical, the GPU solver being deterministic from the same start.", 0.5, 4.1, 7.0, 0.5, { fontSize: 10.5, italic: true, color: C.text2 }, "table note");
  L.media(pres, s, C, "clip_a.mp4", 7.85, 1.5, 4.98, 2.8, "Isaac Sim: the pitcher at fill 0.5, tilted at 5\u00b0/s until the particle water leaves it", "clip A: the pitcher tilting in Isaac Sim until the water leaves", "clip a");
  const cards = [
    { title: "Why measure", colour: K.gold, fill: TINT.gold, lines: ["21 generated copies of the same liquid pouring structure stated 7 different flow-onset ranges: 60\u2013110\u00b0 five times, 30\u2013110\u00b0 five, 40\u2013110\u00b0 four, and others."] },
    { title: "The setup, kept with the value", colour: K.teal, fill: TINT.teal, lines: ["Isaac Sim 6.0.0.1, PhysX PBD water on the GPU; the exact mesh as collider; 2 mm particles; steps of 1/600 s; tilt at 5\u00b0/s; onset when 10 particles are outside the hull."] },
    { title: "Against the generated ranges", colour: K.blue, fill: TINT.blue, lines: ["The simulation pours 0.8\u20133.8\u00b0 after the geometry in every case, as expected of a criterion that waits for particles outside the hull; against 0\u2013120\u00b0 and 40\u2013120\u00b0, both are narrow and agree."] },
  ];
  L.columns(pres, s, C, cards, 0.5, 4.6, W - 1.0, 1.5, "onset cards");
  L.band(pres, s, "In the pouring scene", "the check reads the pitcher's simulated 55.2\u00b0: the trial's 105.8\u00b0 pours, a tilt of 11.5\u00b0 is refused, each citing the record.", 0.5, 6.2, W - 1.0, 0.5, "onset band");
}

// %% slide 25: the viewers, with the viewer following a run as a clip
function viewers(pres, C, s) {
  L.media(pres, s, C, "clip_c.mp4", 0.5, 1.5, 7.1, 4.0, "llm_reasoner_viz replaying run 77fc9006: the components, the timeline, the attempts and where the pours ended", "clip C: the viewer following a run", "clip c");
  L.media(pres, s, C, "cram_viz_designator.png", 7.85, 1.5, 4.98, 2.8, "cram_viz: the resolved designator with its why, what it rests on, and the hypothesis consulted", "screenshot: cram_viz's Designator tab", "designator shot");
  L.card(pres, s, 7.85, 4.55, 4.98, 1.45, K.line_light, TINT.grey, "log card");
  text(s, C, [
    { text: "Everything comes from one log per run. ", options: { bold: true, color: K.navy } },
    { text: "A replayed resolution writes about 36 events, 210 KB, the prompts being most of it; a real Execute of 12 s wrote 65 plan events, 15 KB. Every pour tried and every performed run is kept as a NEEM through coraplex's ORM, in one database the viewer opens read-only.", options: {} },
  ], 8.05, 4.65, 4.58, 1.25, { fontSize: 10.5, valign: "middle" }, "log text");
  L.band(pres, s, "Two tools", "cram_viz shows the scene and the knowledge and performs the run; llm_reasoner_viz shows the system doing its job, live and replayed.", 0.5, 6.2, W - 1.0, 0.5, "viewers band");
}

// %% slide 20: the pour prediction in three numbers, with the landings a run recorded
function pourPrediction(pres, C, s) {
  const cards = [
    { big: "0.188 m", colour: K.gold, fill: TINT.gold, head: "the default pour's miss", text: "coraplex's defaults put the mouth 0.188 m beside the opening; only the simulated pour told the resolver so." },
    { big: "3 of 3", colour: K.teal, fill: TINT.teal, head: "first pours land, against 0 of 3", text: "with the prediction offered as a tool before answering; one pour tried per run instead of two." },
    { big: "< 5 mm", colour: K.blue, fill: TINT.blue, head: "between prediction and simulation", text: "every predicted landing was within 5 mm of where the simulated pour measured it." },
  ];
  const gap = 0.35, pad = 0.25;
  const cw = (W - 1.0 - 2 * gap) / 3;
  cards.forEach((c, i) => {
    const x = 0.5 + i * (cw + gap), y = 1.45, h = 2.05;
    L.card(pres, s, x, y, cw, h, c.colour, c.fill, "card " + i);
    text(s, C, c.big, x + pad, y + 0.15, cw - 2 * pad, 0.75, { fontSize: 40, color: c.colour, fontFace: FONT.heading, valign: "middle" }, "big " + i);
    text(s, C, c.head, x + pad, y + 0.92, cw - 2 * pad, 0.35, { fontSize: 13, bold: true, fontFace: FONT.heading }, "head " + i);
    text(s, C, c.text, x + pad, y + 1.3, cw - 2 * pad, 0.7, { fontSize: 10.5, color: C.text2 }, "text " + i);
  });

  // the landings of one recorded run, drawn as the viewer draws them: seen from above
  const lx = 0.5, ly = 3.7, lw = 4.9, lh = 2.3;
  L.card(pres, s, lx, ly, lw, lh, K.line_light, K.background, "landing card");
  text(s, C, "Where the pours ended, seen from above (run 77fc9006, simple_pouring)", lx + 0.2, ly + 0.1, lw - 0.4, 0.3, { bold: true, color: K.navy, fontFace: FONT.heading, fontSize: 11 }, "landing title");
  const scale = 8; // inches per metre
  const cx = lx + 0.95, cy = ly + 1.25, r = 0.069 * scale;
  s.addShape(pres.ShapeType.ellipse, { x: cx - r, y: cy - r, w: 2 * r, h: 2 * r, fill: { color: K.background }, line: { color: K.navy, width: 1.25, dashType: "dash" }, objectName: "opening" });
  text(s, C, "the bowl's opening, r = 0.069 m", cx - 1.0, cy + r + 0.04, 2.0, 0.25, { fontSize: 9, color: C.text2, align: "center" }, "opening label");
  const pours = [
    { ahead: -0.036, right: 0.184, landed: false, label: "1: the defaults, 0.187 m off" },
    { ahead: 0.001, right: 0.0, landed: true, label: "2: pour_side_offset −0.188 m, 1 mm off" },
  ];
  pours.forEach((p, i) => {
    const qx = cx + p.right * scale, qy = cy - p.ahead * scale;
    const colour = p.landed ? K.teal : K.gold;
    s.addShape(pres.ShapeType.ellipse, { x: qx - 0.13, y: qy - 0.13, w: 0.26, h: 0.26, fill: { color: colour }, line: { color: K.background, width: 1 }, objectName: `pour ${i}` });
    text(s, C, String(i + 1), qx - 0.13, qy - 0.13, 0.26, 0.26, { fontSize: 9.5, bold: true, color: K.background, align: "center", valign: "middle" }, `pour ${i} number`);
    text(s, C, p.label, lx + 2.95, ly + 0.85 + i * 0.45, lw - 3.15, 0.4, { fontSize: 9.5, color: colour, bold: true, valign: "middle" }, `pour ${i} label`);
  });
  text(s, C, "up the page is ahead of the robot, to the right is its right; distances to scale", lx + 2.95, ly + 1.8, lw - 3.15, 0.45, { fontSize: 8.5, italic: true, color: C.text2 }, "axes note");

  // the landing, performed in the simulation
  const tx = lx + lw + 0.35;
  L.media(pres, s, C, "clip_b_land.mp4", tx, ly, 4.1, 2.0, "Execute of run 77fc9006: the PR2 pours with pour_side_offset \u22120.188 m and lands", "clip B: the corrected pour landing", "clip b");
  const nx = tx + 4.25, nw = W - 0.5 - nx;
  text(s, C, [
    { text: "What the prediction composes", options: { bold: true, color: K.navy, fontFace: FONT.heading, fontSize: 10.5, breakLine: true } },
    { text: "coraplex's own pour poses for the values, in milliseconds;", options: { breakLine: true, paraSpaceBefore: 3 } },
    { text: "the hold, measured once per scene and arm;", options: { breakLine: true } },
    { text: "the mouth in the source's frame, as the outcome measures it.", options: {} },
  ], nx, ly, nw, lh, { fontSize: 10 }, "prediction text");
  L.band(pres, s, "Cost per run", "$0.083–0.133 without the prediction, $0.046–0.091 with it. Three runs of pour_geometry_open each, Sonnet 5.5, 2026-10-16.", 0.5, 6.15, W - 1.0, 0.5, "cost note");
}

// %% slide 21: the levels of competence as a staircase
function competency(pres, C, s) {
  const levels = [
    { name: "Remember", colour: GRADE.defined, question: "What is the action? Which roles? Which material regimes?", backed: "Defined; the frame's content Generated", cq: "CQ-001, CQ-055" },
    { name: "Understand", colour: GRADE.generated, question: "Which phases, in which order? What does each pursue, which event ends it?", backed: "Generated, Sourced where a paper says it; Measured once episodes are observed", cq: "CQ-002–004, 006–008, 059" },
    { name: "Apply", colour: GRADE.measured, question: "At which tilt does this pitcher, half full, start to pour? Is this hypothesis valid here?", backed: "Measured or Derived", cq: "CQ-054, CQ-056, CQ-065" },
    { name: "Analyze", colour: GRADE.measured, question: "Which force-dynamic events occur, and what opposes what? Which contacts does an event make or break?", backed: "Measured where an episode shows the event", cq: "CQ-005, 009–011, 031–035, 060" },
    { name: "Evaluate", colour: GRADE.measured, question: "Which constraints keep this pour safe? Does this designator break one? Did the events match their predictions?", backed: "Measured", cq: "CQ-012–016, CQ-061" },
    { name: "Create", colour: K.navy, question: "An executable designator for this situation; the phases a new initial state needs.", backed: "All of the above, cited", cq: "the resolver" },
  ];
  const x = 0.5, w = W - 1.0, n = levels.length, gap = 0.18;
  const cw = (w - gap * (n - 1)) / n;
  const top = 1.5, bottom = 5.75, rise = 0.38;
  levels.forEach((l, i) => {
    const cx = x + i * (cw + gap);
    const cy = top + (n - 1 - i) * rise;
    const ch = bottom - cy;
    L.card(pres, s, cx, cy, cw, ch, l.colour, K.background, `level ${i} card`);
    s.addShape(pres.ShapeType.rect, { x: cx + 0.02, y: cy + 0.02, w: cw - 0.04, h: 0.08, fill: { color: l.colour }, line: { color: l.colour, width: 0 }, objectName: `level ${i} top` });
    text(s, C, l.name, cx + 0.15, cy + 0.15, cw - 0.3, 0.35, { bold: true, color: l.colour, fontFace: FONT.heading, fontSize: 14, valign: "middle" }, `level ${i} name`);
    text(s, C, l.question, cx + 0.15, cy + 0.55, cw - 0.3, 1.5, { fontSize: 10, color: C.text1 }, `level ${i} question`);
    text(s, C, [{ text: "Backed by  ", options: { bold: true, color: C.text2 } }, { text: l.backed, options: { color: l.colour, bold: true } }], cx + 0.15, bottom - 1.25, cw - 0.3, 0.8, { fontSize: 9.5 }, `level ${i} backed`);
    text(s, C, l.cq, cx + 0.15, bottom - 0.4, cw - 0.3, 0.3, { fontSize: 9, fontFace: FONT.mono, color: C.text2 }, `level ${i} cq`);
  });
  L.gradeLegend(pres, s, C, x, 5.92, "legend");
  L.band(pres, s, "Build order", "C1 verdicts, C2 grades on answers, C3 frozen knowledge, C4 the catalogue, C5 observed episodes, C6 Analyze and Evaluate, C7 Apply and Create; agreed 2026-10-25, nothing of it built yet.", x, 6.25, w, 0.5, "build order band");
}

module.exports = {
  ...require("./bodies_content.js"),
  ...require("./bodies_appendix.js"),
  "knowledge-executive": knowledgeExecutive,
  architecture: require("./architecture_body.js"),
  flow,
  ontology,
  admission,
  flanagan,
  grades,
  "pour-prediction": pourPrediction,
  competency,
  "flow-onset": flowOnset,
  viewers,
};
