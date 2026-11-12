// The deck's layouts and drawing helpers, built from theme.json: every slide of the deck
// and of the samples is composed with these, so the look is defined once.
//
// The look follows the institute's "Action-Centric Knowledge Representation" deck: white
// slides, a navy title with a pale bar beside it, a grey subtitle, cards with a coloured
// border and a tinted fill, pills for small labels, a navy band for the one sentence that
// matters, and the University of Bremen and AICOR marks in the footer.

const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const here = path.resolve(__dirname, "..");
const theme = JSON.parse(fs.readFileSync(path.join(here, "theme.json"), "utf8"));
const K = theme.colours;
const TINT = theme.tints;
const FONT = theme.fonts;
const SIZE = theme.type_scale_points;
const LOGO = theme.logos;

const W = 13.333;
const H = 7.5;
const M = theme.canvas.margin_inches;
const CONTENT_TOP = 1.4;
const CONTENT_BOTTOM = 6.8;

// %% the theme pptxgenjs cannot write itself, applied after writeFile
const THEME = {
  name: theme.name,
  headFontFace: FONT.heading,
  bodyFontFace: FONT.body,
  colors: {
    dk1: K.ink, lt1: K.background, dk2: K.muted, lt2: TINT.grey,
    accent1: K.navy, accent2: K.blue, accent3: K.teal, accent4: K.gold, accent5: K.red, accent6: K.purple,
    hlink: K.blue, folHlink: K.purple,
  },
};

// %% a deck with the layouts defined
function createDeck(title) {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE";
  pres.author = "Srikanth Malineni";
  pres.title = title;
  pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
  const C = pres.SchemeColor;

  const footerLogos = (objects) => {
    const u = LOGO.university_of_bremen, a = LOGO.aicor;
    objects.push({ image: { x: u.x, y: u.y, w: u.w, h: u.h, path: path.join(here, u.path) } });
    objects.push({ image: { x: a.x, y: a.y, w: a.w, h: a.h, path: path.join(here, a.path) } });
  };

  const titleObjects = [
    { rect: { x: 0, y: 1.93, w: W, h: 2.71, fill: { color: K.title_band }, line: { color: K.title_band, width: 0 } } },
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 2.24, w: W - 1.2, h: 0.8, fontSize: SIZE.deck_title, color: C.accent1, align: "center", valign: "middle", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "subtitle", type: "body", x: 0.6, y: 3.02, w: W - 1.2, h: 0.6, fontSize: SIZE.deck_subtitle, color: C.accent1, align: "center", valign: "middle", margin: 0, fontFace: FONT.heading }, text: "" } },
    { placeholder: { options: { name: "tagline", type: "body", x: 0.6, y: 3.74, w: W - 1.2, h: 0.5, fontSize: SIZE.small, color: C.text1, align: "center", valign: "middle", margin: 0 }, text: "" } },
  ];
  footerLogos(titleObjects);
  pres.defineSlideMaster({ title: "TITLE", background: { color: K.background }, objects: titleObjects });

  const dividerObjects = [
    { rect: { x: 0, y: 2.55, w: W, h: 1.9, fill: { color: K.title_band }, line: { color: K.title_band, width: 0 } } },
    { placeholder: { options: { name: "kicker", type: "body", x: 0.6, y: 2.75, w: W - 1.2, h: 0.4, fontSize: SIZE.small, color: C.text2, align: "center", valign: "middle", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 3.15, w: W - 1.2, h: 0.8, fontSize: SIZE.deck_title, color: C.accent1, align: "center", valign: "middle", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "subtitle", type: "body", x: 0.6, y: 3.9, w: W - 1.2, h: 0.45, fontSize: SIZE.small, color: C.text1, align: "center", valign: "middle", margin: 0 }, text: "" } },
  ];
  footerLogos(dividerObjects);
  pres.defineSlideMaster({ title: "DIVIDER", background: { color: K.background }, objects: dividerObjects });

  const contentObjects = [
    { rect: { x: 0.5, y: 0.31, w: 0.08, h: 0.85, fill: { color: K.title_bar }, line: { color: K.title_bar, width: 0 } } },
    { placeholder: { options: { name: "title", type: "title", x: 0.72, y: 0.3, w: 12.03, h: 0.52, fontSize: SIZE.slide_title, bold: true, color: C.accent1, align: "left", valign: "middle", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "subtitle", type: "body", x: 0.74, y: 0.83, w: 12.03, h: 0.34, fontSize: SIZE.slide_subtitle, color: C.text2, align: "left", valign: "middle", margin: 0 }, text: "" } },
  ];
  footerLogos(contentObjects);
  pres.defineSlideMaster({ title: "CONTENT", background: { color: K.background }, objects: contentObjects });

  return { pres, C };
}

// %% drawing helpers, each taking the slide and the deck's scheme colours
function card(pres, slide, x, y, w, h, colour, fill, objectName) {
  slide.addShape(pres.ShapeType.roundRect, { x, y, w, h, rectRadius: 0.12, fill: { color: fill }, line: { color: colour, width: 1.5 }, objectName });
}

function pill(pres, slide, text, x, y, w, h, colour, objectName, fill) {
  slide.addShape(pres.ShapeType.roundRect, { x, y, w, h, rectRadius: h / 2, fill: { color: fill || K.background }, line: { color: colour, width: 1.25 }, objectName: objectName + " pill" });
  slide.addText(text, { x, y, w, h, isTextBox: true, margin: 0, fontSize: 9.5, bold: true, color: colour, align: "center", valign: "middle", objectName });
}

function marker(pres, slide, text, x, y, d, colour, objectName) {
  slide.addShape(pres.ShapeType.ellipse, { x, y, w: d, h: d, fill: { color: colour }, line: { color: colour, width: 0 }, objectName: objectName + " marker" });
  slide.addText(text, { x, y, w: d, h: d, isTextBox: true, margin: 0, fontSize: 11, bold: true, color: K.background, align: "center", valign: "middle", objectName: objectName + " marker text" });
}

function arrow(pres, slide, x1, y1, x2, y2, colour, objectName) {
  slide.addShape(pres.ShapeType.line, {
    x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1),
    line: { color: colour, width: 1.5, endArrowType: "triangle" }, flipH: x2 < x1, flipV: y2 < y1, objectName,
  });
}

function source(pres, slide, C, text) {
  slide.addText(text, { x: 3.5, y: 6.97, w: W - 7, h: 0.36, isTextBox: true, margin: 0, fontSize: SIZE.caption, italic: true, color: C.text2, align: "center", valign: "middle", objectName: "source" });
}

function band(pres, slide, lead, text, x, y, w, h, objectName) {
  slide.addShape(pres.ShapeType.roundRect, { x, y, w, h, rectRadius: 0.1, fill: { color: K.navy }, line: { color: K.navy, width: 0 }, objectName: objectName + " band" });
  slide.addText([
    { text: lead + "   ", options: { bold: true, color: K.band_lead } },
    { text, options: { color: K.background } },
  ], { x: x + 0.3, y, w: w - 0.6, h, isTextBox: true, margin: 0, fontSize: 14, valign: "middle", objectName });
}

// %% body templates: the shapes most content slides are made of

// How many lines a title takes in a box, estimated from the heading face's average glyph width.
function linesOf(text, fontSize, width) {
  const perLine = Math.max(1, Math.floor(width / (0.52 * fontSize / 72)));
  return Math.max(1, Math.ceil(text.length / perLine));
}

// A table in the deck's style: a navy header row, zebra rows, 10.5 pt text.
function table(pres, slide, C, rows, x, y, w, columnWidths, objectName, options) {
  const o = Object.assign({ fontSize: 10.5, headerFontSize: 11, rowHeight: 0.34 }, options || {});
  const header = rows[0].map((text) => ({ text, options: { bold: true, color: K.background, fill: { color: K.navy }, fontSize: o.headerFontSize, valign: "middle" } }));
  const body = rows.slice(1).map((row, i) => row.map((cell) => {
    const text = typeof cell === "string" ? cell : cell.text;
    const cellOptions = typeof cell === "string" ? {} : cell.options;
    return { text, options: Object.assign({ color: C.text1, fontSize: o.fontSize, fill: { color: i % 2 ? TINT.grey : K.background }, valign: "middle" }, cellOptions) };
  }));
  slide.addTable([header, ...body], { x, y, w, colW: columnWidths, rowH: o.rowHeight, border: { type: "solid", pt: 0.75, color: K.line_light }, margin: 0.06, fontFace: FONT.body, objectName, autoPage: false });
}

// A row of numbered step cards joined by arrows, in the colour each step is given.
function steps(pres, slide, C, items, x, y, w, h, objectName) {
  const gap = 0.28;
  const cw = (w - gap * (items.length - 1)) / items.length;
  items.forEach((item, i) => {
    const cx = x + i * (cw + gap);
    card(pres, slide, cx, y, cw, h, item.colour, item.fill || K.background, `${objectName} ${i} card`);
    marker(pres, slide, String(i + 1), cx + 0.15, y + 0.15, 0.3, item.colour, `${objectName} ${i}`);
    const titleLines = linesOf(item.title, 12.5, cw - 0.62);
    const titleHeight = 0.24 * titleLines + 0.12;
    slide.addText(item.title, { x: cx + 0.52, y: y + 0.12, w: cw - 0.62, h: titleHeight, isTextBox: true, margin: 0, fontSize: 12.5, bold: true, color: item.colour, fontFace: FONT.heading, valign: "top", objectName: `${objectName} ${i} title` });
    const textTop = y + 0.22 + titleHeight;
    slide.addText(item.text, { x: cx + 0.15, y: textTop, w: cw - 0.3, h: y + h - textTop - 0.1, isTextBox: true, margin: 0, fontSize: 10.5, color: C.text1, valign: "top", objectName: `${objectName} ${i} text` });
    if (i < items.length - 1) arrow(pres, slide, cx + cw + 0.03, y + h / 2, cx + cw + gap - 0.03, y + h / 2, K.line, `${objectName} ${i} arrow`);
  });
  return cw;
}

// Columns of equal width, each a card with a heading, an optional kicker and lines of text.
function columns(pres, slide, C, items, x, y, w, h, objectName) {
  const gap = 0.3;
  const cw = (w - gap * (items.length - 1)) / items.length;
  items.forEach((item, i) => {
    const cx = x + i * (cw + gap);
    card(pres, slide, cx, y, cw, h, item.colour, item.fill || K.background, `${objectName} ${i} card`);
    const titleLines = linesOf(item.title, 15, cw - 0.5);
    const titleHeight = 0.3 * titleLines + 0.1;
    slide.addText(item.title, { x: cx + 0.25, y: y + 0.18, w: cw - 0.5, h: titleHeight, isTextBox: true, margin: 0, fontSize: 15, bold: true, color: item.colour, fontFace: FONT.heading, valign: "top", objectName: `${objectName} ${i} title` });
    let top = y + 0.22 + titleHeight;
    if (item.kicker) {
      slide.addText(item.kicker, { x: cx + 0.25, y: top, w: cw - 0.5, h: 0.3, isTextBox: true, margin: 0, fontSize: 9.5, bold: true, color: C.text2, charSpacing: 1, objectName: `${objectName} ${i} kicker` });
      top += 0.34;
    }
    const lines = item.lines.map((text, j) => ({ text, options: { breakLine: j < item.lines.length - 1, paraSpaceAfter: 5 } }));
    slide.addText(lines, { x: cx + 0.25, y: top, w: cw - 0.5, h: y + h - top - 0.2, isTextBox: true, margin: 0, fontSize: 11, color: C.text1, valign: "top", objectName: `${objectName} ${i} lines` });
  });
  return cw;
}

// A frame for a video or a screenshot with a caption beneath; the media itself is added by
// the caller (addMedia or addImage at the frame's inner rectangle), or the frame stays as
// a placeholder naming what goes there.
function mediaFrame(pres, slide, C, x, y, w, h, caption, placeholderText, objectName) {
  card(pres, slide, x, y, w, h, K.line, TINT.grey, `${objectName} frame`);
  if (placeholderText) {
    slide.addText(placeholderText, { x: x + 0.3, y: y + 0.3, w: w - 0.6, h: h - 0.6, isTextBox: true, margin: 0, fontSize: 11, italic: true, color: C.text2, align: "center", valign: "middle", objectName: `${objectName} placeholder` });
  }
  if (caption) {
    slide.addText(caption, { x, y: y + h + 0.06, w, h: 0.3, isTextBox: true, margin: 0, fontSize: 10.5, italic: true, color: C.text2, align: "center", valign: "top", objectName: `${objectName} caption` });
  }
  return { x: x + 0.08, y: y + 0.08, w: w - 0.16, h: h - 0.16 };
}

// A legend of the evidence grades, one coloured dot and name each, in one row.
function gradeLegend(pres, slide, C, x, y, objectName, perRow, rowHeight) {
  const grades = theme.evidence_grades;
  const names = Object.keys(grades);
  const columns = perRow || names.length;
  names.forEach((name, i) => {
    const gx = x + (i % columns) * 1.45;
    const gy = y + Math.floor(i / columns) * (rowHeight || 0.3);
    slide.addShape(pres.ShapeType.ellipse, { x: gx, y: gy + 0.06, w: 0.18, h: 0.18, fill: { color: grades[name] }, line: { color: grades[name], width: 0 }, objectName: `${objectName} ${name} dot` });
    slide.addText(name[0].toUpperCase() + name.slice(1), { x: gx + 0.26, y: gy, w: 1.15, h: 0.3, isTextBox: true, margin: 0, fontSize: 10.5, color: C.text1, valign: "middle", objectName: `${objectName} ${name}` });
  });
}

// A video or picture from media/ fitted into a box (16:9 kept), with a caption; when the
// file is not there yet, a frame naming what goes there.
function media(pres, slide, C, file, x, y, w, h, caption, placeholderText, objectName) {
  const fs = require("fs");
  const full = path.join(here, "media", file);
  const ratio = 16 / 9;
  let mw = w, mh = w / ratio;
  if (mh > h) { mh = h; mw = h * ratio; }
  const mx = x + (w - mw) / 2, my = y;
  if (!fs.existsSync(full)) {
    mediaFrame(pres, slide, C, mx, my, mw, mh, caption, placeholderText, objectName);
    return;
  }
  card(pres, slide, mx - 0.04, my - 0.04, mw + 0.08, mh + 0.08, K.line, K.background, `${objectName} frame`);
  if (file.endsWith(".mp4")) {
    const poster = full.replace(/\.mp4$/, ".jpg");
    const cover = fs.existsSync(poster) ? "data:image/jpeg;base64," + fs.readFileSync(poster).toString("base64") : undefined;
    slide.addMedia({ type: "video", path: full, cover, x: mx, y: my, w: mw, h: mh, objectName });
  } else {
    slide.addImage({ path: full, x: mx, y: my, w: mw, h: mh, objectName });
  }
  if (caption) {
    slide.addText(caption, { x: mx, y: my + mh + 0.08, w: mw, h: 0.3, isTextBox: true, margin: 0, fontSize: 10.5, italic: true, color: C.text2, align: "center", valign: "top", objectName: `${objectName} caption` });
  }
}

module.exports = { theme, K, TINT, FONT, SIZE, W, H, M, CONTENT_TOP, CONTENT_BOTTOM, THEME, createDeck, card, pill, marker, arrow, source, band, table, steps, columns, mediaFrame, gradeLegend, media };
