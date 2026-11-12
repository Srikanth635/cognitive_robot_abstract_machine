// Builds a deck from an outline on the approved layouts.
//
//   NODE_PATH=<dir with node_modules> PPTX_SKILL_SCRIPTS=<pptx skill scripts> node builder/deck.js [--short | --idea]
//
// Without a flag it builds the full talk from outline.js; `--short` its six-slide version;
// `--idea` the five-slide deck on the research idea from idea_outline.js. A slide whose
// id has a body registered is drawn with it, the others show the skeleton: what the
// slide says and the visual planned for it. Writes out/<name>.pptx.

const path = require("path");
const { applyTheme } = require(process.env.PPTX_SKILL_SCRIPTS + "/apply_theme.js");
const L = require("./layouts.js");
const idea = process.argv.includes("--idea");
const outline = idea ? require("./idea_outline.js") : Object.assign({ name: "llm_reasoner", title: "llm_reasoner: generative knowledge for CRAM's parameter resolution", bodiesModule: "./bodies.js" }, require("./outline.js"));
const { sections, slides } = outline;

const { pres, C } = L.createDeck(outline.title);
const K = L.K, TINT = L.TINT, FONT = L.FONT, W = L.W;

// %% renderers of finished slide bodies, by slide id; the rest show the skeleton
const BODIES = require(outline.bodiesModule);

// %% the skeleton body: what the slide says, and the visual planned for it
function skeletonBody(slide, entry) {
  const top = L.CONTENT_TOP, bottom = L.CONTENT_BOTTOM;
  const leftW = 7.9, gap = 0.35;
  const rightX = 0.5 + leftW + gap, rightW = W - 0.5 - rightX;
  const h = bottom - top;

  L.card(pres, slide, 0.5, top, leftW, h, K.line_light, K.background, "what the slide says");
  slide.addText("What the slide says", { x: 0.75, y: top + 0.15, w: leftW - 0.5, h: 0.3, isTextBox: true, margin: 0, fontSize: 12, bold: true, color: C.accent1, fontFace: FONT.heading, objectName: "points heading" });
  const points = entry.points.map((text, i) => ({ text, options: { bullet: { code: "25A0" }, color: C.text1, breakLine: i < entry.points.length - 1, paraSpaceAfter: 6 } }));
  slide.addText(points, { x: 0.75, y: top + 0.5, w: leftW - 0.5, h: h - 0.65, isTextBox: true, margin: 0, fontSize: 11.5, valign: "top", objectName: "points" });

  L.card(pres, slide, rightX, top, rightW, h, K.line_light, TINT.grey, "planned visual");
  slide.addText("Planned visual", { x: rightX + 0.25, y: top + 0.15, w: rightW - 0.5, h: 0.3, isTextBox: true, margin: 0, fontSize: 12, bold: true, color: C.accent1, fontFace: FONT.heading, objectName: "visual heading" });
  slide.addText(entry.visual, { x: rightX + 0.25, y: top + 0.5, w: rightW - 0.5, h: 2.2, isTextBox: true, margin: 0, fontSize: 11.5, italic: true, color: C.text2, valign: "top", objectName: "visual" });
  slide.addText("Rests on", { x: rightX + 0.25, y: top + 2.8, w: rightW - 0.5, h: 0.3, isTextBox: true, margin: 0, fontSize: 12, bold: true, color: C.accent1, fontFace: FONT.heading, objectName: "sources heading" });
  const sources = entry.sources.map((text, i) => ({ text, options: { color: C.text2, breakLine: i < entry.sources.length - 1, paraSpaceAfter: 3 } }));
  slide.addText(sources, { x: rightX + 0.25, y: top + 3.15, w: rightW - 0.5, h: h - 3.3, isTextBox: true, margin: 0, fontSize: 10.5, fontFace: FONT.mono, valign: "top", objectName: "sources" });
  L.pill(pres, slide, "skeleton: body to come", rightX + 0.25, bottom - 0.45, 1.9, 0.3, K.gold, "skeleton pill", TINT.gold);
}

// %% the slides; `--short` builds the six-slide version for a five-minute slot
const SHORT_IDS = ["title", "principles", "flow", "flow-onset", "live", "summary"];
const short = process.argv.includes("--short");
const chosen = short ? slides.filter((entry) => SHORT_IDS.includes(entry.id)) : slides;
const sectionTitle = Object.fromEntries(sections.map((s) => [s.id, s.title]));
let currentSection = null;
for (const entry of chosen) {
  if (entry.section !== currentSection) {
    pres.addSection({ title: sectionTitle[entry.section] });
    currentSection = entry.section;
  }
  const common = { sectionTitle: sectionTitle[entry.section] };
  if (entry.kind === "title") {
    const s = pres.addSlide(Object.assign({ masterName: "TITLE" }, common));
    s.addText(entry.title, { placeholder: "title" });
    s.addText(entry.subtitle, { placeholder: "subtitle" });
    s.addText(entry.tagline, { placeholder: "tagline" });
    s.addText("Srikanth Malineni  ·  AICOR Institute for Artificial Intelligence, University of Bremen  ·  2026", { x: 0.6, y: 5.2, w: W - 1.2, h: 0.4, isTextBox: true, margin: 0, fontSize: 12, color: C.text2, align: "center", objectName: "author line" });
    s.addNotes(entry.notes);
    continue;
  }
  if (entry.kind === "divider") {
    const s = pres.addSlide(Object.assign({ masterName: "DIVIDER" }, common));
    s.addText(entry.kicker, { placeholder: "kicker" });
    s.addText(entry.title, { placeholder: "title" });
    s.addText(entry.subtitle, { placeholder: "subtitle" });
    s.addNotes(entry.notes);
    continue;
  }
  const s = pres.addSlide(Object.assign({ masterName: "CONTENT" }, common));
  s.addText(entry.title, { placeholder: "title" });
  s.addText(entry.subtitle, { placeholder: "subtitle" });
  if (BODIES[entry.id]) BODIES[entry.id](pres, C, s, entry);
  else skeletonBody(s, entry);
  const full = [...new Set(entry.sources)].join("  ·  ");
  const files = [...new Set(entry.sources.map((t) => t.split(" §")[0]))].join("  ·  ");
  L.source(pres, s, C, full.length > 70 ? files : full);
  if (entry.title.length > 60) console.warn("title longer than one line:", entry.id, entry.title.length);
  s.addNotes(entry.notes);
}

// %% write
(async () => {
  const out = path.join(__dirname, "..", "out", (short ? outline.name + "_short" : outline.name) + ".pptx");
  await pres.writeFile({ fileName: out });
  await applyTheme(out, L.THEME);
  console.log("wrote", out, "with", chosen.length, "slides");
})();
