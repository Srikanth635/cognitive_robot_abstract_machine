// The bodies of the five-slide deck on the research idea, drawn as pictures: the gap
// between instruction and execution, the ring of the idea, the perspectives and the
// levels, the trap case, and the causal chain physics backs.

const L = require("./layouts.js");
const K = L.K, TINT = L.TINT, FONT = L.FONT, W = L.W;
const GRADE = L.theme.evidence_grades;

function text(slide, C, content, x, y, w, h, options, objectName) {
  slide.addText(content, Object.assign({ x, y, w, h, isTextBox: true, margin: 0, fontSize: 11, color: C.text1, valign: "top", objectName }, options || {}));
}

function heading(slide, C, content, x, y, w, colour, objectName, size) {
  text(slide, C, content, x, y, w, 0.4, { bold: true, color: colour || K.navy, fontFace: FONT.heading, fontSize: size || 14, valign: "middle" }, objectName);
}

// %% slide 1: the research problem
function problem(pres, C, s) {
  // the instruction
  const y = 1.5, h = 2.6;
  L.card(pres, s, 0.5, y, 3.3, h, K.blue, TINT.blue, "instruction card");
  heading(s, C, "What is said", 0.75, y + 0.15, 2.8, K.blue, "said heading");
  s.addShape(pres.ShapeType.roundRect, { x: 0.75, y: y + 0.7, w: 2.8, h: 0.9, rectRadius: 0.15, fill: { color: K.background }, line: { color: K.blue, width: 1.25 }, objectName: "bubble" });
  text(s, C, "“Pour the milk into the bowl.”", 0.85, y + 0.7, 2.6, 0.9, { fontSize: 15, bold: true, color: K.navy, fontFace: FONT.heading, align: "center", valign: "middle" }, "instruction");
  text(s, C, "A goal, not a motion. Which arm, which tilt, from which side, how far beside the goal, how high: nothing in the sentence says.", 0.75, y + 1.7, 2.8, 0.85, { fontSize: 10.5, color: C.text2, italic: true }, "said note");
  L.arrow(pres, s, 3.85, y + h / 2, 4.3, y + h / 2, K.line, "to gap");

  // the gap: the open slots
  L.card(pres, s, 4.35, y, 4.3, h, K.red, TINT.red, "gap card");
  heading(s, C, "What is left open", 4.6, y + 0.15, 3.8, K.red, "gap heading");
  const slots = ["which arm?", "which tilt?", "from which side?", "how far beside?", "how high?", "which container?", "approach from where?", "the mouth over the opening?"];
  slots.forEach((t, i) => {
    const col = i % 2, row = Math.floor(i / 2);
    L.pill(pres, s, t, 4.6 + col * 1.95, y + 0.65 + row * 0.4, 1.85, 0.32, K.red, `slot ${i}`, K.background);
  });
  text(s, C, "Eight slots of coraplex's PouringAction; a default binds a slot too, silently.", 4.6, y + 2.25, 3.8, 0.3, { fontSize: 10, color: C.text2, italic: true }, "gap note");
  L.arrow(pres, s, 8.7, y + h / 2, 9.15, y + h / 2, K.line, "to execution");

  // execution
  L.card(pres, s, 9.2, y, 3.63, h, K.navy, K.background, "execution card");
  heading(s, C, "What execution demands", 9.45, y + 0.15, 3.2, K.navy, "execution heading");
  L.media(pres, s, C, "clip_b_land.jpg", 9.45, y + 0.62, 3.13, 1.4, "", "", "execution still");
  text(s, C, "Every value bound before a motion is generated; with the defaults the mouth ends 0.188 m beside the bowl.", 9.45, y + 2.0, 3.2, 0.55, { fontSize: 10, color: C.text2, italic: true }, "execution note");

  // what CRAM holds today
  const cy = 4.35, ch = 1.3;
  const cw = (W - 1.0 - 0.3) / 2;
  L.card(pres, s, 0.5, cy, cw, ch, K.line, TINT.grey, "handwritten card");
  text(s, C, [{ text: "A backend written by hand  ", options: { bold: true, color: K.navy, fontSize: 13, fontFace: FONT.heading } }, { text: "answers only what somebody thought to write; a new object, world or action means new code.", options: { fontSize: 11 } }], 0.75, cy + 0.15, cw - 0.5, ch - 0.3, { valign: "middle" }, "handwritten text");
  L.card(pres, s, 0.8 + cw, cy, cw, ch, K.line, TINT.grey, "learned card");
  text(s, C, [{ text: "A model learned from data  ", options: { bold: true, color: K.navy, fontSize: 13, fontFace: FONT.heading } }, { text: "answers only inside the data it saw; outside it, a default or silence, and in both cases a value without its reasons.", options: { fontSize: 11 } }], 1.05 + cw, cy + 0.15, cw - 0.5, ch - 0.3, { valign: "middle" }, "learned text");
  L.band(pres, s, "Objective", "turn any underspecified instruction into an executable action designator, every choice explained and checked.", 0.5, 5.9, W - 1.0, 0.6, "objective band");
}

// %% slide 2: the idea as a ring
function ring(pres, C, s) {
  const cx = 4.0, cy = 4.1;
  const polar = (r, degrees) => ({ x: cx + r * Math.cos(degrees * Math.PI / 180), y: cy + r * Math.sin(degrees * Math.PI / 180) });
  const box = (r) => ({ x: cx - r, y: cy - r, w: 2 * r, h: 2 * r });

  // the outer ring: the perspectives
  const perspectives = ["FrameNet roles", "Minsky frames", "Image schemas", "Flanagan phases", "Failures, corrections"];
  const outerR = 2.15;
  perspectives.forEach((name, i) => {
    const start = -90 - 36 + i * 72 + 2, end = start + 68;
    s.addShape(pres.ShapeType.blockArc, Object.assign(box(outerR), { angleRange: [start, end], arcThicknessRatio: 0.17, fill: { color: TINT.blue }, line: { color: K.blue, width: 1 }, objectName: `perspective arc ${i}` }));
    const mid = (start + end) / 2;
    const p = polar(outerR + 0.36, mid);
    const left = Math.cos(mid * Math.PI / 180) < -0.2, right = Math.cos(mid * Math.PI / 180) > 0.2;
    text(s, C, name, left ? Math.max(0.5, p.x - 1.4) : right ? p.x : p.x - 0.85, p.y - 0.16, right ? 1.4 : left ? Math.min(1.4, p.x - 0.5) : 1.7, 0.32, { fontSize: 10, bold: true, color: K.blue, align: left ? "right" : right ? "left" : "center", valign: "middle" }, `perspective label ${i}`);
  });

  // the four sources of truth
  const quadrants = [
    { name: "Physics", does: "measured numbers", colour: K.gold, start: -90 },
    { name: "Live world", does: "supplies the facts", colour: K.teal, start: 0 },
    { name: "Ontology", does: "keeps it consistent", colour: K.blue, start: 90 },
    { name: "Generative model", does: "hypothesises", colour: K.purple, start: 180 },
  ];
  const r1 = 1.72;
  quadrants.forEach((q, i) => {
    s.addShape(pres.ShapeType.blockArc, Object.assign(box(r1), { angleRange: [q.start + 2, q.start + 88], arcThicknessRatio: 0.5, fill: { color: q.colour }, line: { color: K.background, width: 1 }, objectName: `quadrant ${q.name}` }));
    const p = polar(r1 * 0.74, q.start + 45);
    text(s, C, [{ text: q.name, options: { bold: true, breakLine: true } }, { text: q.does, options: {} }], p.x - 0.62, p.y - 0.36, 1.24, 0.72, { fontSize: 9, color: K.background, align: "center", valign: "middle" }, `quadrant label ${q.name}`);
  });

  // the centre
  const r0 = 0.84;
  s.addShape(pres.ShapeType.ellipse, Object.assign(box(r0), { fill: { color: K.navy }, line: { color: K.background, width: 2 }, objectName: "centre" }));
  text(s, C, [{ text: "Executable action", options: { bold: true, breakLine: true, fontSize: 10 } }, { text: "every slot bound, every choice explained", options: { fontSize: 8 } }], cx - 0.75, cy - 0.42, 1.5, 0.84, { color: K.background, align: "center", valign: "middle" }, "centre label");

  // the legend
  const lx = 7.9, lw = W - 0.5 - lx;
  heading(s, C, "What each part does", lx, 1.5, lw, K.navy, "legend heading");
  const legend = [
    { colour: K.purple, head: "The generative model hypothesises", body: "what the action normally involves: roles, phases, the events that end them, conditions, forces, failures; always as a hypothesis, generated once per situation." },
    { colour: K.blue, head: "The ontology keeps it honest in form", body: "the hypothesis is admitted only when it meets the ontology's rules and, under HermiT, contradicts nothing in CRAM-AICOR and DUL; it is queried as krrood objects." },
    { colour: K.teal, head: "The live world supplies the facts", body: "what is in this world now, read through krrood: kinds, individuals, who holds what, which way an opening faces." },
    { colour: K.gold, head: "Physics supplies the numbers", body: "the quantities that decide a parameter come from geometry and simulation, never from the model; a measurement supports or refutes what was hypothesised." },
  ];
  legend.forEach((l, i) => {
    const y = 2.0 + i * 0.9;
    s.addShape(pres.ShapeType.ellipse, { x: lx, y: y + 0.05, w: 0.22, h: 0.22, fill: { color: l.colour }, line: { color: l.colour, width: 0 }, objectName: `legend dot ${i}` });
    text(s, C, [{ text: l.head, options: { bold: true, color: l.colour, fontSize: 12, breakLine: true } }, { text: l.body, options: { fontSize: 10 } }], lx + 0.35, y, lw - 0.35, 0.85, {}, `legend ${i}`);
  });
  text(s, C, [{ text: "Outer ring  ", options: { bold: true, color: K.blue } }, { text: "the stereotypical knowledge, one perspective per ontology module, all in CRAM-AICOR terms.", options: {} }], lx, 5.6, lw, 0.45, { fontSize: 10 }, "outer ring note");
  L.band(pres, s, "The resolver", "decides here, and says what each choice rests on.", lx, 6.15, lw, 0.45, "resolver band");
}

// %% slide 3: perspectives and levels
function knowledge(pres, C, s) {
  const rows = [
    { name: "Action frame", who: "FrameNet, Minsky", colour: K.blue, says: "roles, typical fillers, defaults, preconditions, what the action brings about" },
    { name: "Script", who: "Schank and Abelson", colour: K.muted, says: "the expected sequence of events, preparation included" },
    { name: "Image schemas", who: "Johnson, Lakoff, Talmy", colour: K.teal, says: "the substance goes along a path from source to goal and into it; the source tips; the goal holds" },
    { name: "Predictive control", who: "Johansson and Flanagan", colour: K.purple, says: "phases, the events that end them, sensory predictions, errors and corrections" },
    { name: "Failures", who: "what goes wrong", colour: K.red, says: "the failures of each phase and what corrects them" },
  ];
  heading(s, C, "Stereotypical knowledge, five perspectives", 0.5, 1.45, 6.0, K.navy, "perspectives heading");
  rows.forEach((r, i) => {
    const y = 1.95 + i * 0.72;
    L.card(pres, s, 0.5, y, 6.0, 0.62, r.colour, K.background, `perspective ${i}`);
    text(s, C, [{ text: r.name + "  ", options: { bold: true, color: r.colour, fontSize: 12, fontFace: FONT.heading } }, { text: r.who, options: { fontSize: 9.5, color: C.text2 } }], 0.7, y + 0.05, 5.6, 0.26, { valign: "middle" }, `perspective ${i} name`);
    text(s, C, r.says, 0.7, y + 0.31, 5.6, 0.28, { fontSize: 10, color: C.text1, valign: "middle" }, `perspective ${i} says`);
  });
  L.card(pres, s, 0.5, 5.6, 6.0, 0.95, K.line_light, TINT.grey, "ontology card");
  text(s, C, [{ text: "One ontology. ", options: { bold: true, color: K.navy } }, { text: "Each perspective is its own module in CRAM-AICOR terms; the model proposes its vocabulary, a person reviews it, HermiT keeps the generated knowledge consistent, and EQL queries it as krrood objects.", options: {} }], 0.7, 5.68, 5.6, 0.8, { fontSize: 10.5, valign: "middle" }, "ontology text");

  // the levels as a staircase
  const levels = [
    { name: "Remember", q: "What is pouring, which roles?", colour: GRADE.defined },
    { name: "Understand", q: "Which phases, which event ends each?", colour: GRADE.generated },
    { name: "Apply", q: "At which tilt does this pitcher pour?", colour: GRADE.measured },
    { name: "Analyze", q: "Which forces oppose which?", colour: GRADE.measured },
    { name: "Evaluate", q: "Which constraints keep it safe?", colour: GRADE.measured },
    { name: "Create", q: "An executable designator", colour: K.navy },
  ];
  const sx = 6.9, sw = W - 0.5 - sx;
  heading(s, C, "Questions at every level of competence", sx, 1.45, sw, K.navy, "levels heading");
  const n = levels.length, gap = 0.1, cw = (sw - gap * (n - 1)) / n, top = 2.0, bottom = 5.5, rise = 0.42;
  levels.forEach((l, i) => {
    const x = sx + i * (cw + gap), y = top + (n - 1 - i) * rise;
    L.card(pres, s, x, y, cw, bottom - y, l.colour, K.background, `level ${i}`);
    s.addShape(pres.ShapeType.rect, { x: x + 0.02, y: y + 0.02, w: cw - 0.04, h: 0.08, fill: { color: l.colour }, line: { color: l.colour, width: 0 }, objectName: `level ${i} top` });
    text(s, C, l.name, x + 0.05, y + 0.14, cw - 0.1, 0.3, { bold: true, color: l.colour, fontFace: FONT.heading, fontSize: 9, valign: "middle" }, `level ${i} name`);
    text(s, C, l.q, x + 0.08, y + 0.48, cw - 0.16, bottom - y - 0.55, { fontSize: 9, color: C.text1 }, `level ${i} question`);
  });
  text(s, C, "after Bloom's taxonomy; each step coloured by the strongest evidence that can back it", sx, 5.55, sw, 0.28, { fontSize: 9, italic: true, color: C.text2, align: "center" }, "levels note");
  L.gradeLegend(pres, s, C, sx, 5.95, "grade legend", 3, 0.32);
}

// %% slide 4: the trap case
function unknown(pres, C, s) {
  const y = 1.5, h = 3.6;
  const cw = (W - 1.0 - 0.6) / 3;
  // the situation
  L.card(pres, s, 0.5, y, cw, h, K.navy, K.background, "situation card");
  heading(s, C, "The situation", 0.75, y + 0.15, cw - 0.5, K.navy, "situation heading");
  const facts = [["the only bowl", "stands upside down", K.red], ["a milk carton", "no stated opening", K.muted], ["a cereal box", "no stated opening", K.muted], ["the instruction", "leaves the goal to choose", K.navy]];
  facts.forEach(([what, state, colour], i) => {
    const fy = y + 0.7 + i * 0.62;
    L.pill(pres, s, what, 0.75, fy, 1.6, 0.3, colour, `fact ${i}`);
    text(s, C, state, 2.45, fy, cw - 2.2, 0.3, { fontSize: 10.5, color: C.text1, valign: "middle" }, `fact ${i} state`);
  });
  text(s, C, "The world states what it can measure: which objects are containers comes from the ontology, which way an opening faces from the mesh.", 0.75, y + 2.95, cw - 0.5, 0.6, { fontSize: 9.5, italic: true, color: C.text2 }, "situation note");

  // the data-bound answer
  const x2 = 0.8 + cw;
  L.card(pres, s, x2, y, cw, h, K.red, TINT.red, "bound card");
  heading(s, C, "A backend bound to its data", x2 + 0.25, y + 0.15, cw - 0.5, K.red, "bound heading");
  text(s, C, "No case like this was recorded or written. The default resolver kept the usual geometry and poured into the overturned bowl, 2 of 2 runs, reasoning only about the lateral error.", x2 + 0.25, y + 0.7, cw - 0.5, 1.3, { fontSize: 11 }, "bound text");
  text(s, C, "2 of 2", x2 + 0.25, y + 2.05, cw - 0.5, 0.65, { fontSize: 30, bold: true, color: K.red, fontFace: FONT.heading, valign: "middle" }, "bound count");
  text(s, C, "poured into the overturned bowl", x2 + 0.25, y + 2.75, cw - 0.5, 0.35, { fontSize: 13, bold: true, color: K.red, fontFace: FONT.heading, valign: "middle" }, "bound verdict");

  // the generative answer
  const x3 = 1.1 + 2 * cw;
  L.card(pres, s, x3, y, cw, h, K.teal, TINT.teal, "generative card");
  heading(s, C, "A generative model with its perspectives", x3 + 0.25, y + 0.15, cw - 0.5, K.teal, "generative heading");
  text(s, C, "Reads “the bowl's opening faces down”, knows from the image schemas that the lip must be over an opening, and leaves the bowl. One run refuses: the carton has no stated opening either.", x3 + 0.25, y + 0.7, cw - 0.5, 1.3, { fontSize: 11 }, "generative text");
  text(s, C, "5 of 6", x3 + 0.25, y + 2.05, cw - 0.5, 0.65, { fontSize: 30, bold: true, color: K.teal, fontFace: FONT.heading, valign: "middle" }, "generative count");
  text(s, C, "left the overturned bowl", x3 + 0.25, y + 2.75, cw - 0.5, 0.35, { fontSize: 13, bold: true, color: K.teal, fontFace: FONT.heading, valign: "middle" }, "generative verdict");
  L.pill(pres, s, "grounds: world fact + knowledge fragment", x3 + 0.25, y + h - 0.5, cw - 0.5, 0.3, K.teal, "grounds pill", K.background);

  L.band(pres, s, "Honest", "two runs per arm on one scene is a first sign, not a rate; the repetition did not reproduce it. What stays is structural: the world states the opening, and the answer states its grounds.", 0.5, 5.4, W - 1.0, 0.6, "honest band");
  text(s, C, "Upside-down bowl, opening stated: default 2 of 2 into the bowl; image schemas 0, failures 0, both 1; one refusal (2026-10-20, 8 runs).", 0.5, 6.15, W - 1.0, 0.3, { fontSize: 9.5, italic: true, color: C.text2, align: "center" }, "trap figures");
}

// %% slide 5: physics behind the knowledge
function physics(pres, C, s) {
  L.media(pres, s, C, "clip_a.mp4", 0.5, 1.5, 5.6, 3.15, "Isaac Sim: the pitcher at fill 0.5 tilts at 5°/s until the particle water leaves it, at 55.2°", "clip A: the pitcher pouring in Isaac Sim", "clip a");
  L.card(pres, s, 0.5, 5.0, 5.6, 1.0, K.line_light, TINT.grey, "measurement card");
  text(s, C, [{ text: "Measured, not generated. ", options: { bold: true, color: K.teal } }, { text: "Two containers at three fills each: the simulation pours 0.8–3.8° after the mesh's own geometry; three runs of a case are identical. Generated for the same situation: 7 different ranges in 21 copies.", options: {} }], 0.7, 5.08, 5.2, 0.85, { fontSize: 10.5, valign: "middle" }, "measurement text");

  // the causal chain
  const x = 6.5, w = W - 0.5 - x;
  const blocks = [
    { title: "Preconditions", colour: K.blue, fill: TINT.blue, grade: "Defined, Generated", lines: ["the source is held and open; the goal is open and upright", "the lip is over the goal's opening"] },
    { title: "Phases and their boundary events", colour: K.purple, fill: TINT.purple, grade: "Generated; onset Measured", lines: ["tilt until the flow onset: 55.2° for this pitcher at fill 0.5", "regulate the flow; terminate; return upright"] },
    { title: "Effects", colour: K.teal, fill: TINT.teal, grade: "Measured in the trial", lines: ["the substance accumulates in the goal", "with the lip 0.188 m beside the opening the pour misses; the prediction now sees it before the answer"] },
  ];
  const bh = 1.3, gap = 0.3;
  blocks.forEach((b, i) => {
    const y = 1.5 + i * (bh + gap);
    L.card(pres, s, x, y, w, bh, b.colour, b.fill, `chain ${i}`);
    text(s, C, b.title, x + 0.25, y + 0.1, w - 2.6, 0.35, { bold: true, color: b.colour, fontFace: FONT.heading, fontSize: 13, valign: "middle" }, `chain ${i} title`);
    L.pill(pres, s, b.grade, x + w - 2.4, y + 0.12, 2.15, 0.3, b.colour, `chain ${i} grade`, K.background);
    text(s, C, b.lines.map((t, j) => ({ text: t, options: { breakLine: j < b.lines.length - 1, paraSpaceAfter: 3 } })), x + 0.25, y + 0.5, w - 0.5, bh - 0.55, { fontSize: 10.5 }, `chain ${i} lines`);
    if (i < blocks.length - 1) L.arrow(pres, s, x + w / 2, y + bh, x + w / 2, y + bh + gap, K.line, `chain arrow ${i}`);
  });
  L.band(pres, s, "The rule", "a measured onset of 55.2° supports a generated 30–110° and refutes 70–110°; what no physics result reaches stays Generated, and says so.", 0.5, 6.15, W - 1.0, 0.55, "rule band");
}

module.exports = { "idea-problem": problem, "idea-ring": ring, "idea-knowledge": knowledge, "idea-unknown": unknown, "idea-physics": physics };
