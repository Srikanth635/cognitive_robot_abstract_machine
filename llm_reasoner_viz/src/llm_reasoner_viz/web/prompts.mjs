// Reading a prompt as the tagged sections it is made of. A tag opens or closes a section
// only when it stands alone on its line; a tag named in the middle of a line is prose.

const NAME = '[A-Za-z_][\\w.-]*';
const OPENING = new RegExp('^<(' + NAME + ')((?:\\s+[^<>]*)?)>\\s*$');
const CLOSING = new RegExp('^</(' + NAME + ')>\\s*$');
const ON_ONE_LINE = new RegExp('^<(' + NAME + ')((?:\\s+[^<>]*)?)>(.*)</\\1>\\s*$');
const BLANK_LINES_AT_THE_START = /^(?:[ \t]*\n)+/;
const WHITESPACE_AT_THE_END = /\s+$/;

function tidy(lines) {
  return lines.join('\n').replace(BLANK_LINES_AT_THE_START, '').replace(WHITESPACE_AT_THE_END, '');
}

function sectionOf(name, attributes, content) {
  return { name: name, attributes: attributes.trim(), content: content, length: content.length };
}

// The sections of a prompt in the order they stand, and the text outside any of them.
export function parseSections(text) {
  const sections = [];
  const loose = [];
  let open = null;
  let depth = 0;

  function close() {
    sections.push(sectionOf(open.name, open.attributes, tidy(open.lines)));
    open = null;
    depth = 0;
  }

  text.split('\n').forEach(function (line) {
    if (open === null) {
      const single = ON_ONE_LINE.exec(line);
      if (single) {
        sections.push(sectionOf(single[1], single[2], single[3].trim()));
        return;
      }
      const opening = OPENING.exec(line);
      if (opening) {
        open = { name: opening[1], attributes: opening[2], lines: [] };
        return;
      }
      loose.push(line);
      return;
    }
    const closing = CLOSING.exec(line);
    if (closing && closing[1] === open.name) {
      if (depth === 0) {
        close();
        return;
      }
      depth -= 1;
    }
    const opening = OPENING.exec(line);
    if (opening && opening[1] === open.name) depth += 1;
    open.lines.push(line);
  });
  if (open !== null) close();
  return { sections: sections, loose: tidy(loose) };
}

// Whether the text is a JSON object or array, which is the only JSON worth laying out.
export function isJson(text) {
  if (!/^\s*[\[{]/.test(text)) return false;
  try {
    JSON.parse(text);
    return true;
  } catch (error) {
    return false;
  }
}

export function prettyJson(text) {
  return JSON.stringify(JSON.parse(text), null, 2);
}
