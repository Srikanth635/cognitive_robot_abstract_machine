// The bodies of the appendix slides: the two contracts, the knowledge cases and the
// scientific contract of the predictive-manipulation ontology, as tables.

const L = require("./layouts.js");
const K = L.K, TINT = L.TINT, FONT = L.FONT, W = L.W;

function text(slide, C, content, x, y, w, h, options, objectName) {
  slide.addText(content, Object.assign({ x, y, w, h, isTextBox: true, margin: 0, fontSize: 11, color: C.text1, valign: "top", objectName }, options || {}));
}

function heading(slide, C, content, x, y, w, objectName) {
  text(slide, C, content, x, y, w, 0.34, { bold: true, color: K.navy, fontFace: FONT.heading, fontSize: 12, valign: "middle" }, objectName);
}

// %% the world-access contract
function worldContract(pres, C, s) {
  L.card(pres, s, 0.5, 1.5, 3.6, 4.5, K.teal, TINT.teal, "direct calls card");
  heading(s, C, "Direct calls (WorldAccess)", 0.7, 1.6, 3.2, "direct calls heading");
  text(s, C, ["handle_of(individual) -> str", "individual_of(handle)", "is_live(handle)", "holders(handle) -> List[Holder]", "tools() -> List[Tool]", "version"].map((t, i, a) => ({ text: t, options: { breakLine: i < a.length - 1, paraSpaceAfter: 4 } })), 0.7, 2.05, 3.2, 1.9, { fontSize: 10, fontFace: FONT.mono }, "direct calls");
  text(s, C, [{ text: "Comparisons  ", options: { bold: true, color: K.teal } }, { text: "equal, not equal, less, less or equal, greater, greater or equal, contains; checked against the values a path reaches before the search runs.", options: {} }], 0.7, 4.0, 3.2, 1.0, { fontSize: 10 }, "comparisons");
  text(s, C, [{ text: "Value forms  ", options: { bold: true, color: K.teal } }, { text: "PlainValue, NumericValue, WorldStateDependentValue, IndividualReference, NestedIndividual, CollectionValue, UndescribedValue.", options: {} }], 0.7, 5.0, 3.2, 0.95, { fontSize: 10 }, "forms");
  const rows = [
    ["Tool", "Input", "Answer"],
    ["list_kinds", "within, listing (instantiated, present, all)", "kinds with identifier, summary, individual counts"],
    ["describe_kind", "kind", "summary, ancestors, specialisations, field definitions"],
    ["find_individuals", "kind, conditions (path, comparison, value or handle), limit ≤ 50, path", "the search in English, total, counts per kind, handles"],
    ["describe_individual", "handle, depth", "the individual as readable text, holders first"],
    ["holders_of", "handle", "holders with field and summary"],
    ["follow_path", "handle, path", "the value reached, with its form"],
    ["list_predicates", "about", "predicates and value functions with their arguments"],
    ["evaluate_predicate", "predicate, arguments", "what it says"],
  ];
  L.table(pres, s, C, rows, 4.4, 1.5, W - 0.5 - 4.4, [1.9, 3.3, 3.23], "world tools table", { rowHeight: 0.42, fontSize: 10 });
  L.band(pres, s, "Errors", "JSON in one shape: UnknownHandle names the search that gives such handles, InapplicableComparison the fields the path may continue to, PredicateFailed carries the world model's error.", 0.5, 6.15, W - 1.0, 0.55, "errors band");
}

// %% the knowledge-access contract
function knowledgeContract(pres, C, s) {
  const operations = [
    ["Operation", "Answers"],
    ["serve(query, instruction)", "the record identifier of the hypothesis for a request"],
    ["tools(*record_ids)", "the knowledge tools over those hypotheses and the ones they build on, read as one body of knowledge"],
    ["frame_parts_filled(record_id, frame_fields)", "the parts of a stored core frame each field of a designator fills"],
    ["citation(record_id, name)", "the IRI citing a fragment by the name the tools gave it"],
    ["instances_of(kind)", "the stored fragments of a vocabulary class: an EQL domain"],
    ["is_vocabulary_class(kind)", "whether a class's instances are knowledge rather than parts of the world"],
  ];
  heading(s, C, "Operations (KnowledgeAccess, version 1.3)", 0.5, 1.45, 7.0, "operations heading");
  L.table(pres, s, C, operations, 0.5, 1.85, 7.0, [2.9, 4.1], "operations table", { rowHeight: 0.46, fontSize: 10 });
  const tools = [
    ["Tool", "Answers"],
    ["knowledge_overview", "the action concept, the situation, the phases in order with subgoals, parameters and requirements, the roles; for an action core its frame and the perspectives read with it"],
    ["find_fragments(kind)", "the fragments of that class and its subclasses, each with name, class and label"],
    ["describe_fragment(name)", "its class and meaning, a statement per property it gives a value for, and the fragments linking to it"],
  ];
  heading(s, C, "The knowledge tools", 7.8, 1.45, 5.03, "tools heading");
  L.table(pres, s, C, tools, 7.8, 1.85, 5.03, [1.7, 3.33], "knowledge tools table", { rowHeight: 0.7, fontSize: 10 });
  L.card(pres, s, 7.8, 4.95, 5.03, 1.0, K.line_light, TINT.grey, "conventions card");
  text(s, C, "The knowledge tools follow the world tools' conventions: one JSON definition each with purpose, returns, use_when and avoid; answers as JSON; a question naming something the hypothesis lacks answered as an error in the shared shape.", 8.0, 5.03, 4.63, 0.85, { fontSize: 10 }, "conventions text");
  L.band(pres, s, "Fragments", "named as their citations name them, those of a perspective after its module (action-core/source), so names stay apart across the hypotheses read.", 0.5, 6.15, W - 1.0, 0.55, "fragments band");
}

// %% the knowledge cases
function knowledgeCases(pres, C, s) {
  const rows = [
    ["Case", "Task", "Situation chosen", "Hypothesis"],
    ["K1", "none", "both unknown", "generated, 1 attempt"],
    ["K2", "“Pour the milk from the cup into the bowl.”", "LIQUID", "generated, 1 attempt"],
    ["K3", "“Pour the cereal …”", "GRANULAR", "generated, 1 attempt"],
    ["K4", "“… milk … until the bowl is half full.”", "LIQUID, TARGET_QUANTITY_ATTAINED", "generated, 2 attempts"],
    ["K5", "“Pour all the cereal …”", "GRANULAR, SOURCE_EXHAUSTED", "generated, 2 attempts"],
    ["K6", "as K2", "LIQUID", "served from the store (K2's record), 9 s, $0.04"],
  ];
  heading(s, C, "The knowledge request (R7, 2026-10-12): six cases, all as expected, $2.02", 0.5, 1.45, W - 1.0, "cases heading");
  L.table(pres, s, C, rows, 0.5, 1.85, W - 1.0, [0.8, 4.3, 3.6, 3.63], "cases table", { rowHeight: 0.36, fontSize: 10.5 });
  const reading = [
    ["Reading them (I5)", "Knowledge read", "Result", "Grounding stated"],
    ["K1", "overview", "RIGHT, 1.85", "the LiquidPouringStructure and its ph_tilt phase set no other angle"],
    ["K3", "overview, find_fragments", "RIGHT, 1.85", "the granular structure gives no tilt value"],
    ["K4", "overview and 7 descriptions", "RIGHT, 1.5", "inside sp_tilt's 30–120° setpoint range and above th_flow's 40–110° onset"],
    ["K5", "overview", "RIGHT, 1.85", "GRANULAR, SOURCE_EXHAUSTED: tilt until the cereal has flowed out"],
  ];
  L.table(pres, s, C, reading, 0.5, 4.5, W - 1.0, [1.6, 2.6, 1.5, 6.63], "reading table", { rowHeight: 0.3, fontSize: 9.5, headerFontSize: 10 });
  L.band(pres, s, "Read", "every request read knowledge_overview first; $0.29 in total; the knowledge changed a choice only where the task asked for nuance, and every answer cited the fragments it relied on.", 0.5, 6.2, W - 1.0, 0.5, "reading band");
}

// %% the scientific contract
function flanaganContract(pres, C, s) {
  const statuses = [
    ["Status", "Meaning"],
    ["EMPIRICAL_FINDING", "the reviewed experiments directly report the behaviour"],
    ["SUPPORTED_INTERPRETATION", "the authors infer a control organisation from converging findings"],
    ["AUTHOR_HYPOTHESIS", "the authors propose a mechanism not yet established"],
    ["OPEN_QUESTION", "the paper states the relationship remains unresolved"],
    ["ENGINEERING_EXTENSION", "useful for robotic knowledge, not a claim of the paper"],
  ];
  L.table(pres, s, C, statuses, 0.5, 1.5, 5.6, [2.3, 3.3], "statuses table", { rowHeight: 0.42, fontSize: 10 });
  text(s, C, "A status describes scientific support, not software maturity or confidence in a generated instance.", 0.5, 4.1, 5.6, 0.5, { fontSize: 10, italic: true, color: C.text2 }, "status note");

  // the module chain
  const cx = 6.5, cw = 2.1, ch = 0.5;
  const chain = [["core", K.navy], ["object_dynamics", K.purple], ["prehension", K.teal], ["pouring", K.blue]];
  chain.forEach(([name, colour], i) => {
    const y = 1.5 + i * 0.85;
    L.card(pres, s, cx, y, cw, ch, colour, K.background, `module ${name}`);
    text(s, C, name, cx, y, cw, ch, { fontSize: 11, bold: true, color: colour, fontFace: FONT.mono, align: "center", valign: "middle" }, `module ${name} name`);
    if (i < chain.length - 1) L.arrow(pres, s, cx + cw / 2, y + ch, cx + cw / 2, y + 0.85, K.line, `module arrow ${i}`);
  });
  text(s, C, "dependency direction; no module imports an action-specific one upward", cx - 0.3, 4.95, cw + 0.6, 0.5, { fontSize: 9, italic: true, color: C.text2, align: "center" }, "chain note");
  L.card(pres, s, cx + cw + 0.4, 1.5, 2.1, 2.4, K.line, TINT.grey, "side capabilities card");
  text(s, C, [{ text: "Side capabilities", options: { bold: true, color: K.navy, fontSize: 11, breakLine: true } }, { text: "gaze\nobservation\nlearning\nperception", options: { fontFace: FONT.mono, fontSize: 10.5 } }], cx + cw + 0.6, 1.6, 1.7, 2.2, {}, "side capabilities");

  const rx = 11.3, rw = W - 0.5 - rx;
  const distinctions = ["mechanical event / sensory event", "prediction / observation", "subgoal / evidence of attainment", "feedforward command / feedback correction", "corrective action / model update", "mass / weight force", "description / occurrence", "actor / effector"];
  L.card(pres, s, rx, 1.5, rw, 4.0, K.line_light, K.background, "distinctions card");
  text(s, C, [{ text: "Distinctions kept", options: { bold: true, color: K.navy, fontSize: 11, breakLine: true } }].concat(distinctions.map((d, i) => ({ text: d, options: { fontSize: 9.5, breakLine: i < distinctions.length - 1, paraSpaceBefore: 3 } }))), rx + 0.15, 1.6, rw - 0.3, 3.8, {}, "distinctions");
  L.band(pres, s, "Authority", "Flanagan, Bowman and Johansson, Control strategies in object manipulation tasks, Current Opinion in Neurobiology 16, 2006; pouring is an engineering extension.", 0.5, 6.15, W - 1.0, 0.55, "authority band");
}

module.exports = {
  "appendix-world-contract": worldContract,
  "appendix-knowledge-contract": knowledgeContract,
  "appendix-knowledge-cases": knowledgeCases,
  "appendix-flanagan": flanaganContract,
};
