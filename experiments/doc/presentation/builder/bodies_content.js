// The bodies of the content slides that are not diagrams: the problem, the principles,
// the three kinds of truth, the prompt protocol, the world tools, the perspectives, the
// results, the comparison, the limits, the roadmap and the summary. Every figure is the
// record's, cited in outline.js.

const L = require("./layouts.js");
const K = L.K, TINT = L.TINT, FONT = L.FONT, SIZE = L.SIZE, W = L.W;
const COMPONENT = L.theme.components;
const GRADE = L.theme.evidence_grades;

function text(slide, C, content, x, y, w, h, options, objectName) {
  slide.addText(content, Object.assign({ x, y, w, h, isTextBox: true, margin: 0, fontSize: 11, color: C.text1, valign: "top", objectName }, options || {}));
}

function heading(slide, C, content, x, y, w, colour, objectName, size) {
  text(slide, C, content, x, y, w, 0.36, { bold: true, color: colour || K.navy, fontFace: FONT.heading, fontSize: size || 13, valign: "middle" }, objectName);
}

// %% slide 2: the designator and its eight values
function gap(pres, C, s) {
  const rows = [
    ["Slot of PouringAction", "In the query", "Bound how"],
    ["source_container", "pouring_cup", { text: "stated by the plan", options: { color: K.navy, bold: true } }],
    ["target_container", "bowl.root", { text: "stated by the plan", options: { color: K.navy, bold: true } }],
    ["arm", "variable(Arm)", { text: "open: the resolver chooses", options: { color: K.red, bold: true } }],
    ["tilt_angle", "...", { text: "open: the resolver chooses", options: { color: K.red, bold: true } }],
    ["pour_side", "not written", { text: "coraplex's default", options: { color: K.gold, bold: true } }],
    ["pour_side_offset", "not written", { text: "coraplex's default", options: { color: K.gold, bold: true } }],
    ["pour_approach_offset", "not written", { text: "coraplex's default", options: { color: K.gold, bold: true } }],
    ["pour_height", "not written", { text: "coraplex's default", options: { color: K.gold, bold: true } }],
  ];
  heading(s, C, "The query a plan writes, and what each of the eight slots is bound by", 0.5, 1.45, 6.4, K.navy, "gap heading");
  L.table(pres, s, C, rows, 0.5, 1.9, 6.4, [2.2, 1.6, 2.6], "slots table", { rowHeight: 0.4, fontSize: 10.5 });
  text(s, C, "the(PouringAction)(source_container=pouring_cup, target_container=bowl.root, arm=variable(Arm), tilt_angle=..., )", 0.5, 5.6, 6.4, 0.5, { fontSize: 10, fontFace: FONT.mono, color: K.navy }, "query text");
  L.media(pres, s, C, "clip_b_miss.mp4", 7.2, 1.45, 5.63, 3.17, "Execute of run acca6ab3: with the geometry left to the defaults, the pitcher's mouth ends beside the bowl", "clip B': the defaults' pour missing the bowl", "clip b miss");
  text(s, C, "Nothing moves until every slot is bound. A default binds a slot too, silently: coraplex's defaults put the pitcher's mouth 0.188 m beside the bowl's opening.", 7.2, 5.1, 5.63, 0.9, { fontSize: 11.5, color: C.text2, italic: true }, "gap note");
  L.band(pres, s, "The question", "which values, chosen how, and on what grounds?", 0.5, 6.2, W - 1.0, 0.5, "gap band");
}

// %% slide 5: the five principles
function principles(pres, C, s) {
  const items = [
    { title: "A base layer on krrood", colour: K.teal, text: "llm_reasoner imports krrood and never a world model, a robot or a plan package; it learns a world only through krrood's symbol graph." },
    { title: "Structured prompts", colour: K.purple, text: "Everything a model reads is tagged sections holding JSON data and named rules; the only English is content: a query, a task, docstrings, source." },
    { title: "Generic behaviour", colour: K.navy, text: "Nothing is written for one action, object or world; behaviour follows from types, fields, defaults, docstrings and the query." },
    { title: "One source of truth", colour: K.blue, text: "The ontology: its working form (Python objects for EQL) and its reasoning form (RDF for HermiT) are both generated from it, never hand-written." },
    { title: "Physics confirms", colour: K.gold, fill: TINT.gold, text: "A quantity that decides a parameter comes from a shape, a simulation or the literature, never from the model. Decided 2026-10-25." },
  ];
  L.steps(pres, s, C, items, 0.5, 1.55, W - 1.0, 2.6, "principle");
  const traits = [
    ["Accuracy over token cost", "a little extra cost is preferred to any loss of accuracy, precision or structure that would cause repairs or reruns"],
    ["Components with one responsibility", "connected by declared contracts, not by reaching into each other's internals"],
    ["Checked, not trusted", "every answer's grounds are checked against their source; every hypothesis is admitted by the ontology's rules"],
  ];
  const tw = (W - 1.0 - 0.6) / 3;
  traits.forEach(([head, body], i) => {
    const x = 0.5 + i * (tw + 0.3);
    L.card(pres, s, x, 4.5, tw, 1.3, K.line_light, TINT.grey, `trait ${i}`);
    text(s, C, [{ text: head, options: { bold: true, color: K.navy, fontSize: 12, breakLine: true } }, { text: body, options: { color: C.text1, fontSize: 10.5 } }], x + 0.2, 4.62, tw - 0.4, 1.1, {}, `trait ${i} text`);
  });
  L.band(pres, s, "In one line", "a language model supplies the general understanding; an ontology, the live world and physics keep it honest.", 0.5, 6.1, W - 1.0, 0.55, "principles band");
}

// %% slide 6: three kinds of truth
function threeTruths(pres, C, s) {
  const cols = [
    { word: "Fact", who: "World injector", colour: COMPONENT.world_injector, fill: TINT.teal, knows: "What is in this world now: kinds, individuals, what holds what, what the world model's predicates say.", example: "The bowl stands upright on the counter; the pitcher is held in the right gripper; the pitcher holds a liquid at fill 0.5." },
    { word: "Stereotype", who: "Knowledge component", colour: COMPONENT.knowledge, fill: TINT.blue, knows: "What an action normally involves, and why: roles, phases, the events that end them, conditions, forces, failures and corrections.", example: "Pouring tilts a source until its lip is over the goal's opening; the flow starts at an onset tilt; too much tilt spills." },
    { word: "Decision", who: "Resolver", colour: COMPONENT.resolver, fill: K.background, knows: "What to do here: a value for every slot, a one-sentence why, and the grounds, each a world handle or a knowledge fragment.", example: "Right arm, tilt 1.85 rad, side offset −0.188 m, because the lip must end over the opening and the prediction says where it ends." },
  ];
  const cw = (W - 1.0 - 0.6) / 3;
  cols.forEach((c, i) => {
    const x = 0.5 + i * (cw + 0.3), y = 1.5, h = 4.4;
    L.card(pres, s, x, y, cw, h, c.colour, c.fill, `truth ${i}`);
    text(s, C, c.word, x + 0.3, y + 0.2, cw - 0.6, 0.7, { fontSize: 34, color: c.colour, fontFace: FONT.heading, valign: "middle" }, `truth ${i} word`);
    L.pill(pres, s, c.who, x + 0.3, y + 1.0, 1.9, 0.3, c.colour, `truth ${i} who`);
    text(s, C, c.knows, x + 0.3, y + 1.5, cw - 0.6, 1.3, { fontSize: 11.5 }, `truth ${i} knows`);
    text(s, C, [{ text: "For pouring  ", options: { bold: true, color: c.colour } }, { text: c.example, options: { color: C.text2 } }], x + 0.3, y + 2.9, cw - 0.6, 1.4, { fontSize: 10.5 }, `truth ${i} example`);
  });
  L.band(pres, s, "The limit that shapes the design", "a stereotype never produces a number like −0.188 m: it names what matters and how it constrains a parameter; the world supplies the values, and experience what worked before.", 0.5, 6.1, W - 1.0, 0.6, "truth band");
}

// %% slide 10: the prompt protocol
function prompts(pres, C, s) {
  const stacks = [
    { title: "System prompt", colour: K.navy, parts: ["<role>", "<inputs>  what each request section means", "<rules>  one named rule per line"] },
    { title: "Request", colour: K.blue, parts: ["<source type=…>  the action's code and docstrings", "<request>  the query in English", "<task>  <situation>  <action>", "<slots>  JSON", "<given>  JSON, the values the query fixes", "<output>  the answer's JSON schema"] },
    { title: "Repair", colour: K.red, parts: ["the request, then", "<rejected_answer>", "<rejection kind=…>  the failure as JSON"] },
  ];
  let y = 1.5;
  const lx = 0.5, lw = 4.3;
  stacks.forEach((st, i) => {
    const h = 0.4 + st.parts.length * 0.24;
    L.card(pres, s, lx, y, lw, h, st.colour, K.background, `stack ${i}`);
    text(s, C, st.title, lx + 0.2, y + 0.08, lw - 0.4, 0.3, { bold: true, color: st.colour, fontFace: FONT.heading, fontSize: 12 }, `stack ${i} title`);
    text(s, C, st.parts.map((p, j) => ({ text: p, options: { breakLine: j < st.parts.length - 1 } })), lx + 0.2, y + 0.4, lw - 0.4, h - 0.45, { fontSize: 9.5, fontFace: FONT.mono, color: C.text1 }, `stack ${i} parts`);
    y += h + 0.12;
  });
  const mx = 5.1, mw = 4.0;
  L.card(pres, s, mx, 1.5, mw, 2.55, K.line_light, TINT.grey, "slot entry card");
  heading(s, C, "One slot entry in <slots>", mx + 0.2, 1.55, mw - 0.4, K.navy, "slot entry heading", 12);
  text(s, C, '{"path": "CuttingAction.tool",\n "type": "Tool",\n "accepts": [{"type": "string",\n              "holds": "handle"}],\n "candidates": {"count": 2,\n   "kind": "…Tool",\n   "listed": [{"handle": "CuttingKnife#1",\n     "kind": "CuttingKnife", …}]}}', mx + 0.2, 1.95, mw - 0.4, 2.05, { fontSize: 9, fontFace: FONT.mono, color: K.navy }, "slot entry");
  L.card(pres, s, mx, 4.2, mw, 1.7, K.line_light, K.background, "placeholder card");
  text(s, C, [{ text: "Placeholders, not names. ", options: { bold: true, color: K.navy } }, { text: "<Kind>#<n> and <field>.<field> stand for the world's names, so the model learns the shape of a handle and of a path, not one world's vocabulary. Up to 10 candidates are listed; more are counted and searched with the tools.", options: {} }], mx + 0.2, 4.3, mw - 0.4, 1.5, { fontSize: 10.5 }, "placeholder text");
  const rx = 9.4, rw = W - 0.5 - rx;
  L.card(pres, s, rx, 1.5, rw, 2.55, K.line_light, TINT.grey, "answer card");
  heading(s, C, "The answer: one JSON array", rx + 0.2, 1.55, rw - 0.4, K.navy, "answer heading", 12);
  text(s, C, '[{"CuttingAction.tool": "CuttingKnife#1",\n  "CuttingAction.technique": "SLICE",\n  "why": "one sentence",\n  "grounds": [{"world": "CuttingKnife#1"},\n              {"knowledge": "ph_tilt"}]}]', rx + 0.2, 1.95, rw - 0.4, 2.05, { fontSize: 9, fontFace: FONT.mono, color: K.navy }, "answer entry");
  L.card(pres, s, rx, 4.2, rw, 1.7, K.line_light, K.background, "verdict card");
  heading(s, C, "Verdicts instead of guesses", rx + 0.2, 4.25, rw - 0.4, K.navy, "verdict heading", 12);
  ["no_fitting_candidate", "equally_fitting", "unsatisfiable"].forEach((v, i) => L.pill(pres, s, v, rx + 0.2, 4.7 + i * 0.38, 1.9, 0.3, K.red, `verdict ${i}`));
  text(s, C, "a ground naming nothing goes back as UnknownGround", rx + 2.2, 4.7, rw - 2.4, 1.1, { fontSize: 10, color: C.text2, italic: true }, "unknown ground");
  L.band(pres, s, "Every ground is checked", "a world handle must be live, a knowledge fragment must exist under its IRI; the resolver keeps the why and the grounds of every accepted answer.", 0.5, 6.15, W - 1.0, 0.55, "prompt band");
}

// %% slide 11: the world tools
function world(pres, C, s) {
  const rows = [
    ["Tool", "Answers"],
    ["list_kinds", "the kinds with identifier, summary and how many individuals each has"],
    ["describe_kind", "summary, ancestors, specialisations, field definitions"],
    ["find_individuals", "a search under conditions on field paths: total, counts per kind, handles"],
    ["describe_individual", "the individual as readable text, its holders first"],
    ["holders_of", "what holds it, with the field"],
    ["follow_path", "the value a field path reaches, with its form"],
    ["list_predicates", "the predicates and value functions the world model declares"],
    ["evaluate_predicate", "what a predicate says of the given arguments"],
  ];
  L.table(pres, s, C, rows, 0.5, 1.5, 7.4, [1.9, 5.5], "tools table", { rowHeight: 0.4, fontSize: 10.5 });
  const rx = 8.2, rw = W - 0.5 - rx;
  L.card(pres, s, rx, 1.5, rw, 1.75, K.teal, TINT.teal, "predicates card");
  text(s, C, [{ text: "Predicates of the world. ", options: { bold: true, color: K.teal } }, { text: "Functions declared with symbolic_function: in the pouring scene contact, is_supported_by, is_supporting and stable (which says it needs a simulation); with the robot annotated, 14 predicates, among them who holds what and what a camera sees.", options: {} }], rx + 0.2, 1.6, rw - 0.4, 1.6, { fontSize: 10.5 }, "predicates text");
  L.card(pres, s, rx, 3.4, rw, 1.3, K.blue, TINT.blue, "checked card");
  text(s, C, [{ text: "Checked before the search. ", options: { bold: true, color: K.blue } }, { text: "A wrong path or an inapplicable comparison is an error naming the fields to continue to, never an empty result; a guessed handle's error names the search that gives such handles.", options: {} }], rx + 0.2, 3.5, rw - 0.4, 1.15, { fontSize: 10.5 }, "checked text");
  L.card(pres, s, rx, 4.85, rw, 1.1, K.gold, TINT.gold, "limit card");
  text(s, C, [{ text: "Limit. ", options: { bold: true, color: K.gold } }, { text: "Values a world model computes in properties, such as poses and bounding boxes, are not reachable yet; a function appears only once its module is loaded.", options: {} }], rx + 0.2, 4.95, rw - 0.4, 0.95, { fontSize: 10.5 }, "limit text");
  L.band(pres, s, "Handles", "Kind#n, stable for the world's lifetime; the resolver turns them back into live objects, which is why it stays in the world's process.", 0.5, 6.15, W - 1.0, 0.55, "world band");
}

// %% slide 15: the perspectives
function perspectives(pres, C, s) {
  const items = [
    { title: "Action frame", kicker: "FRAMENET (FILLMORE), FRAMES (MINSKY)", colour: K.blue, lines: ["The roles: source, substance, goal, agent.", "What each is normally like, defaults, preconditions, what the action is for and what it brings about."] },
    { title: "Image schemas", kicker: "JOHNSON, LAKOFF, TALMY", colour: K.teal, lines: ["The spatial and force structure beneath.", "The substance goes along a path from the source to the goal and into it; the source tips; the goal holds."] },
    { title: "Script", kicker: "SCHANK AND ABELSON", colour: K.muted, lines: ["The expected sequence of events, preparation included."] },
    { title: "Predictive control", kicker: "JOHANSSON AND FLANAGAN", colour: K.purple, lines: ["Phases, the events that end them, sensory predictions, prediction errors and corrections."] },
    { title: "Failures", kicker: "WHAT GOES WRONG", colour: K.red, lines: ["What can go wrong and what corrects it; built and measured with the image schemas."] },
  ];
  L.columns(pres, s, C, items, 0.5, 1.5, W - 1.0, 2.9, "perspective");
  const notes = [
    ["Each is its own ontology module", "importing CRAM-AICOR and DUL; everything is stated in CRAM-AICOR terms where it has them, and CRAM-AICOR itself is never edited."],
    ["The model proposes, a person reviews", "the vocabulary of one perspective at a time is proposed by the model, checked by the rules and HermiT, and reviewed before it is used."],
    ["Offered opt-in", "the action core is always consulted; each perspective is ticked on or off by the requester, since the measurements showed they change the grounds more than the choices."],
  ];
  const nw = (W - 1.0 - 0.6) / 3;
  notes.forEach(([head, body], i) => {
    const x = 0.5 + i * (nw + 0.3);
    L.card(pres, s, x, 4.65, nw, 1.3, K.line_light, TINT.grey, `note ${i}`);
    text(s, C, [{ text: head, options: { bold: true, color: K.navy, fontSize: 11.5, breakLine: true } }, { text: body, options: { fontSize: 10.5 } }], x + 0.2, 4.75, nw - 0.4, 1.15, {}, `note ${i} text`);
  });
  L.band(pres, s, "Why several", "a language model knows all of these traditions; each says something the others do not, and together they describe the action from word to motion.", 0.5, 6.15, W - 1.0, 0.55, "perspectives band");
}

// %% slide 23: the live suite
function live(pres, C, s) {
  const rows = [
    ["Level", "Cases", "What it checks"],
    ["1 Almost given", "C1, P1", "values only, one candidate each"],
    ["2 Small lists", "C2, P2", "the right individual among 3 listed"],
    ["3 The whole world", "C3, P3", "finding individuals among 229 bodies with the tools"],
    ["4 Conditions on values", "C4, C5, P4, P5", "<, in_, not_, and_ ranges, or_"],
    ["5 Conditions on individuals", "C6, C7, P6, P7", "contains on name.name, exists over a Food's root, linked slots"],
    ["6 Everything open", "C8, P8, C9, P9", "all optional fields open; an(...) with limit 3"],
    ["7 Conflicts", "C10, P10, C11, C12", "a condition forcing an unsuitable tool; nothing to cut; contradictory bounds"],
    ["8 With a task", "T1–T6", "halving, thick slices under a limit, slowly, milk, sandwiches, the sink"],
  ];
  L.table(pres, s, C, rows, 0.5, 1.5, 8.0, [2.2, 1.7, 4.1], "levels table", { rowHeight: 0.4, fontSize: 10 });
  const rx = 8.8, rw = W - 0.5 - rx;
  const bigs = [["28 of 28", "as expected, each in one conversation", K.teal], ["229", "bodies in the PR2's apartment", K.navy], ["$0.98", "for the whole suite, Sonnet 5.5", K.gold]];
  bigs.forEach(([n, l, colour], i) => {
    const y = 1.5 + i * 1.1;
    L.card(pres, s, rx, y, rw, 0.95, colour, K.background, `big ${i} card`);
    text(s, C, n, rx + 0.2, y + 0.08, 1.9, 0.8, { fontSize: 30, color: colour, fontFace: FONT.heading, valign: "middle" }, `big ${i}`);
    text(s, C, l, rx + 2.1, y + 0.1, rw - 2.3, 0.75, { fontSize: 10.5, color: C.text2, valign: "middle" }, `big ${i} label`);
  });
  L.card(pres, s, rx, 4.85, rw, 1.15, K.line_light, TINT.grey, "knowledge cases card");
  text(s, C, [{ text: "Knowledge cases K1–K6: ", options: { bold: true, color: K.navy } }, { text: "all six as expected, $2.02; a generation $0.30–0.41 and about 3 minutes, a served hypothesis 9 s and $0.04; the task “until the bowl is half full” moved the tilt from 1.85 to 1.5 rad.", options: {} }], rx + 0.2, 4.95, rw - 0.4, 1.0, { fontSize: 10 }, "knowledge cases text");
  L.band(pres, s, "Still arbitrary", "the arm, in about 20 of 28 answers, because nothing said where things are or which gripper holds what; the run before had 23 of 28 and led to two fixes.", 0.5, 6.15, W - 1.0, 0.55, "live band");
}

// %% slide 24: perspectives measured, and the upside-down bowl
function trap(pres, C, s) {
  const rows1 = [
    ["pour_geometry_open, 3 runs", "First pour lands", "Any pour lands", "Cost per run"],
    ["Default", "0 of 3", "3 of 3", "$0.102"],
    ["Image schemas", "0 of 3", "3 of 3", "$0.126"],
    ["Failures", "0 of 2", "2 of 2", "$0.142"],
    ["Both", "0 of 3", "3 of 3", "$0.125"],
  ];
  heading(s, C, "The perspectives on the simulated pour (2026-10-19, 18 runs, $3.34)", 0.5, 1.45, 6.3, K.navy, "table 1 heading", 12);
  L.table(pres, s, C, rows1, 0.5, 1.85, 6.3, [2.4, 1.3, 1.3, 1.3], "perspectives table", { rowHeight: 0.38, fontSize: 10.5 });
  text(s, C, "Same verdict as the action frame before it: what the answers rest on changes, what they choose does not; every first pour kept the default geometry and missed by 0.188 m.", 0.5, 3.85, 6.3, 0.6, { fontSize: 10.5, color: C.text2, italic: true }, "table 1 note");
  const rows2 = [
    ["The bowl upside down, opening stated", "Poured into the bowl", "Chose the carton", "Refused"],
    ["Default", "2 of 2", "0", "0"],
    ["Image schemas", "0", "2", "0"],
    ["Failures", "0", "1", "1"],
    ["Both", "1", "1", "0"],
  ];
  heading(s, C, "The trap case (2026-10-20, 8 runs, $2.28)", 7.1, 1.45, 5.73, K.navy, "table 2 heading", 12);
  L.table(pres, s, C, rows2, 7.1, 1.85, 5.73, [2.53, 1.2, 1.1, 0.9], "trap table", { rowHeight: 0.38, fontSize: 10.5 });
  text(s, C, "With the up axis stated only as a vector, no arm noticed and the default arm poured into the bowl 2 of 2. The refusal said the carton has no stated opening either, which is arguably the right answer.", 7.1, 3.85, 5.73, 0.6, { fontSize: 10.5, color: C.text2, italic: true }, "table 2 note");
  const cards = [
    { title: "A first sign, not a rate", colour: K.gold, fill: TINT.gold, lines: ["Two runs per arm on one scene; the repetition with the final harness did not reproduce the change of choice."] },
    { title: "What it changed", colour: K.teal, fill: TINT.teal, lines: ["Which objects have an opening comes from the ontology (hasOpening on Container); which way it faces is measured from each mesh by rays."] },
    { title: "What it cost", colour: K.blue, fill: TINT.blue, lines: ["20–70 % more per run; 2 of 5 failures runs were cut off at 8192 output tokens and failed rather than being repaired."] },
  ];
  L.columns(pres, s, C, cards, 0.5, 4.6, W - 1.0, 1.4, "trap cards");
  L.band(pres, s, "Decided", "the perspectives stay opt-in; the geometric miss is addressed by the prediction, not by more knowledge.", 0.5, 6.15, W - 1.0, 0.5, "trap band");
}

// %% slide 26: conventional resolution and llm_reasoner side by side
function comparison(pres, C, s) {
  const keep = { fill: { color: TINT.gold } };
  const rows = [
    ["", "Resolution from knowledge CRAM already holds", "llm_reasoner"],
    ["Where the knowledge comes from", "Recorded experience or code written per action and world", "A model's general understanding of the action, bound to this world's facts"],
    ["What an answer is", "A value", "A value, a one-sentence why, and grounds checked against the world and the knowledge"],
    ["Consistency", "Whatever the data had", "Form checked by the ontology's acceptance rules; logic, optionally, by HermiT"],
    ["A new object, world or action", "New data or new code", "Whatever the ontology, the mesh and the live world describe"],
    [{ text: "What stays with the conventional methods", options: Object.assign({ bold: true, color: K.gold }, keep) }, { text: "Cheaper and deterministic; the experience of what actually worked, which llm_reasoner does not replace", options: keep }, { text: "About ten cents a resolution and tens of cents for new knowledge; a model's variation between runs", options: keep }],
  ];
  L.table(pres, s, C, rows, 0.5, 1.5, W - 1.0, [3.0, 4.4, 4.93], "comparison table", { rowHeight: 0.62, fontSize: 11 });
  L.band(pres, s, "Why both", "the paper keeps episodic memory as its own pillar; llm_reasoner adds the general understanding and the reasons, not a replacement for experience.", 0.5, 6.15, W - 1.0, 0.55, "comparison band");
}

// %% slide 27: the limits
function limits(pres, C, s) {
  const items = [
    ["Stereotypes give no numbers", "like −0.188 m; the geometric miss needed the prediction, not more knowledge."],
    ["The arm is an arbitrary choice", "until the world says which gripper holds what and where things are."],
    ["Generated ranges vary between runs", "7 different flow-onset ranges in 21 copies of one structure; the physics-backed quantities are the answer, and only the flow onset is built."],
    ["A long final turn can be cut off", "at 8192 output tokens, in 2 of 5 failures runs; the run failed rather than being repaired."],
    ["The geometric onset covers liquids only", "and two integer fields of the measurement records clash with their datatype under HermiT, though the closed-world checks pass."],
    ["Designed, not built", "the competency-question answerer, verdicts of measurements on hypotheses, frozen reviewed knowledge, observed episodes."],
  ];
  items.forEach(([head, body], i) => {
    const y = 1.55 + i * 0.78;
    L.marker(pres, s, String(i + 1), 0.55, y + 0.05, 0.34, K.muted, `limit ${i}`);
    text(s, C, [{ text: head + "  ", options: { bold: true, color: K.navy, fontSize: 13 } }, { text: body, options: { color: C.text1, fontSize: 11.5 } }], 1.1, y, W - 1.6, 0.7, { valign: "top" }, `limit ${i} text`);
  });
  L.band(pres, s, "Said plainly", "the record states each of these, with the run or the commit it was found in.", 0.5, 6.3, W - 1.0, 0.45, "limits band");
}

// %% slide 28: next
function next(pres, C, s) {
  const items = [
    { title: "Verdicts and grades", colour: K.teal, text: "C1 a measurement supports or refutes the fragments stating the same quantity. C2 an evidence grade on every part of an answer. C3 frozen, reviewed knowledge served first." },
    { title: "Questions and episodes", colour: K.blue, text: "C4 the question catalogue and the answerer for Remember and Understand. C5 observed episodes: contacts made and broken, flow begun and ended, in the ontology's own event terms, compared with the predicted order." },
    { title: "Constraints and the resolver", colour: K.navy, text: "C6 Analyze and Evaluate against measured limits and trial outcomes. C7 Apply and Create through the resolver, its grounds graded the same way." },
    { title: "More actions", colour: K.gold, fill: TINT.gold, text: "After pouring: navigating, picking up and placing, the actions a pouring plan performs anyway (D8). The knowledge keeps its own graph beside krrood's symbol graph." },
  ];
  L.steps(pres, s, C, items, 0.5, 1.55, W - 1.0, 2.9, "next step");
  L.card(pres, s, 0.5, 4.75, W - 1.0, 1.2, K.line_light, TINT.grey, "open questions card");
  text(s, C, [{ text: "Open questions the design note leaves to decide: ", options: { bold: true, color: K.navy } }, { text: "the form of a question query; who reviews knowledge before it is frozen; how close a measured value must lie to a generated range to support it; which simulator observes contacts and forces, and how they are read without changing the simulator packages.", options: {} }], 0.75, 4.85, W - 1.5, 1.0, { fontSize: 11, valign: "middle" }, "open questions text");
  L.band(pres, s, "Agreed", "2026-10-25, each step test-first and recorded in status.md; nothing of it built yet.", 0.5, 6.15, W - 1.0, 0.5, "next band");
}

// %% slide 29: summary
function summary(pres, C, s) {
  const statements = [
    { colour: K.purple, text: "A language model supplies the general understanding of an action, as a hypothesis, in the terms of an ontology that keeps it consistent and queryable." },
    { colour: K.teal, text: "The live world supplies the facts through krrood, physics supplies the numbers, and every answer says what it rests on." },
    { colour: K.navy, text: "Measured so far: 28 of 28 live cases; first pours landing 3 of 3 with the prediction; the flow-onset tilt measured within 0.8–3.8° of the geometry. The competency-question layer is next." },
  ];
  statements.forEach((st, i) => {
    const y = 1.55 + i * 1.3;
    L.card(pres, s, 0.5, y, W - 1.0, 1.1, st.colour, K.background, `statement ${i}`);
    L.marker(pres, s, String(i + 1), 0.75, y + 0.38, 0.34, st.colour, `statement ${i}`);
    text(s, C, st.text, 1.3, y + 0.15, W - 2.1, 0.85, { fontSize: 15, fontFace: FONT.heading, color: C.text1, valign: "middle" }, `statement ${i} text`);
  });
  L.card(pres, s, 0.5, 5.5, W - 1.0, 0.95, K.line_light, TINT.grey, "repository card");
  text(s, C, [
    { text: "Code and record  ", options: { bold: true, color: K.navy } },
    { text: "branch feature/physics_backed_knowledge  ·  llm_reasoner/doc/architecture.md  ·  llm_reasoner/doc/status.md", options: { fontFace: FONT.mono, fontSize: 11 } },
  ], 0.75, 5.5, W - 1.5, 0.95, { fontSize: 12, valign: "middle" }, "repository text");
  text(s, C, "Thank you", 0.5, 6.55, W - 1.0, 0.35, { fontSize: 14, color: C.text2, align: "center", fontFace: FONT.heading }, "thanks");
}

module.exports = { gap, principles, "three-truths": threeTruths, prompts, world, perspectives, live, trap, comparison, limits, next, summary };
