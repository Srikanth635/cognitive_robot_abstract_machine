// Writes the talk as a script to read: every slide's title, subtitle, what it says and
// its speaker notes, in order, as Markdown.
//
//   node builder/script.js > out/script.md

const { sections, slides } = require("./outline.js");

const sectionTitle = Object.fromEntries(sections.map((s) => [s.id, s.title]));
const lines = ["# llm_reasoner: the talk as a script", ""];
let current = null;
slides.forEach((entry, i) => {
  if (entry.section !== current) {
    current = entry.section;
    lines.push(`## ${sectionTitle[current]}`, "");
  }
  lines.push(`### ${i + 1}. ${entry.title}`, "");
  if (entry.subtitle) lines.push(`*${entry.subtitle}*`, "");
  if (entry.points) {
    for (const p of entry.points) lines.push(`- ${p}`);
    lines.push("");
  }
  if (entry.visual) lines.push(`**Planned visual:** ${entry.visual}`, "");
  if (entry.sources) lines.push(`**Rests on:** ${entry.sources.join("; ")}`, "");
  lines.push(`**Notes:** ${entry.notes}`, "");
});
process.stdout.write(lines.join("\n"));
