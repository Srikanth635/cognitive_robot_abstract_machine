/* ============================================================================
 * addon/designator_view.js — the Designator tab: the action designator for the
 * active instruction, as collapsible Lisp or JSON.
 *
 * The payload (see cram_viz.designator_description) is one tree: the action with the
 * slots of coraplex's equivalent action, then the generated hypothesis, every fragment
 * described once and referred to by its id elsewhere. Both spellings are produced from
 * one line model -- an indent and coloured tokens per line, a collapsed description
 * folded into one line -- so they fold alike; the DOM only draws those lines.
 * ==========================================================================*/
(function (root) {
  'use strict';

  const INDENT = 2;
  // descriptions deeper than the hypothesis structure start collapsed
  const OPEN_DEPTH = 1;

  function isDescription(value) { return value.node === 'description'; }

  function labelOf(description) {
    const label = description.properties.find((entry) => entry.key === 'label');
    return label ? label.values[0].text : '';
  }

  // %% tree ----------------------------------------------------------------------

  function forEachDescription(description, visit, depth, ancestors) {
    const level = depth || 0;
    const path = (ancestors || []).concat([description.id]);
    visit(description, level, path);
    description.properties.forEach((entry) => entry.values.forEach((value) => {
      if (isDescription(value)) forEachDescription(value, visit, level + 1, path);
    }));
  }

  function initiallyCollapsed(tree) {
    const collapsed = new Set();
    forEachDescription(tree, (description, depth) => {
      if (depth > OPEN_DEPTH) collapsed.add(description.id);
    });
    return collapsed;
  }

  function pathTo(tree, id) {
    let found = null;
    forEachDescription(tree, (description, depth, path) => {
      if (description.id === id && !found) found = path;
    });
    return found;
  }

  function toJson(description, nested) {
    const object = {};
    object[description.kind] = description.type;
    if (nested) object.id = description.id;
    description.properties.forEach((entry) => {
      const values = entry.values.map((value) => (isDescription(value) ? toJson(value, true) : value.data));
      object[entry.key] = entry.many ? values : values[0];
    });
    return object;
  }

  // %% line model ------------------------------------------------------------------
  // a line is { indent, tokens }; a token is { text, cls, ref?, title?, toggle? }

  function token(text, cls, extra) { return Object.assign({ text: text, cls: cls }, extra || {}); }

  function valueToken(value) {
    return token(value.text, 'dz-' + value.kind, value.reference ? { ref: value.reference, title: value.note } : { title: value.note });
  }

  function closeLast(lines, text) {
    lines[lines.length - 1].tokens.push(token(text, 'dz-punct'));
  }

  function lispLines(description, indent, collapsed) {
    const head = [
      token('', '', { toggle: description.id, open: !collapsed.has(description.id) }),
      token('(' + description.article + ' ', 'dz-punct'),
      token(description.kind, 'dz-kind'),
      token(' (type ', 'dz-punct'), token(description.type, 'dz-type'), token(')', 'dz-punct'),
    ];
    if (collapsed.has(description.id)) {
      const label = labelOf(description);
      if (label) head.push(token(' ' + label, 'dz-literal'));
      head.push(token(' …)', 'dz-punct'));
      return [{ indent: indent, tokens: head, id: description.id }];
    }
    if (description.note) head.push(token('  ; ' + description.note, 'dz-comment'));
    const lines = [{ indent: indent, tokens: head, id: description.id }];
    description.properties.forEach((entry) => {
      const inner = indent + INDENT;
      if (!entry.values.some(isDescription)) {
        const tokens = [token('(', 'dz-punct'), token(entry.key, 'dz-key')];
        entry.values.forEach((value) => { tokens.push(token(' ', '')); tokens.push(valueToken(value)); });
        tokens.push(token(')', 'dz-punct'));
        const note = entry.values.length === 1 && entry.values[0].kind !== 'reference' && entry.values[0].note;
        if (note) tokens.push(token('  ; ' + note, 'dz-comment'));
        lines.push({ indent: inner, tokens: tokens });
        return;
      }
      lines.push({ indent: inner, tokens: [token('(', 'dz-punct'), token(entry.key, 'dz-key')] });
      entry.values.forEach((value) => {
        if (isDescription(value)) lines.push(...lispLines(value, inner + INDENT, collapsed));
        else lines.push({ indent: inner + INDENT, tokens: [valueToken(value)] });
      });
      closeLast(lines, ')');
    });
    closeLast(lines, ')');
    return lines;
  }

  function jsonScalar(value) {
    if (value.kind === 'reference') {
      return [token('{"$ref": ', 'dz-punct'), token(JSON.stringify(value.reference), 'dz-reference', { ref: value.reference, title: value.note }), token('}', 'dz-punct')];
    }
    return [token(JSON.stringify(value.data), 'dz-' + value.kind, { title: value.note })];
  }

  function jsonLines(description, indent, collapsed, nested, prefix, suffix) {
    const open = !collapsed.has(description.id);
    const head = (prefix || []).concat([
      token('', '', { toggle: description.id, open: open }),
      token('{', 'dz-punct'),
    ]);
    const fields = [[description.kind, [token(JSON.stringify(description.type), 'dz-type')]]];
    if (nested) fields.push(['id', [token(JSON.stringify(description.id), 'dz-literal')]]);
    if (!open) {
      head.push(token(' ' + JSON.stringify(description.kind) + ': ', 'dz-kind'), token(JSON.stringify(description.type), 'dz-type'));
      const label = labelOf(description);
      if (label) head.push(token(', "label": ' + label, 'dz-literal'));
      head.push(token(' … }' + (suffix || ''), 'dz-punct'));
      return [{ indent: indent, tokens: head, id: description.id }];
    }
    const lines = [{ indent: indent, tokens: head, id: description.id }];
    const inner = indent + INDENT;
    const entries = fields.map(([key, tokens]) => ({ key: key, tokens: tokens }))
      .concat(description.properties.map((entry) => ({ entry: entry })));
    entries.forEach((item, index) => {
      const comma = index < entries.length - 1 ? ',' : '';
      const keyToken = [token(JSON.stringify(item.key || item.entry.key), 'dz-key'), token(': ', 'dz-punct')];
      if (item.tokens) {
        lines.push({ indent: inner, tokens: keyToken.concat(item.tokens, [token(comma, 'dz-punct')]) });
        return;
      }
      const entry = item.entry;
      const scalarOnly = !entry.values.some(isDescription);
      if (!entry.many) {
        const value = entry.values[0];
        if (isDescription(value)) lines.push(...jsonLines(value, inner, collapsed, true, keyToken, comma));
        else lines.push({ indent: inner, tokens: keyToken.concat(jsonScalar(value), [token(comma, 'dz-punct')]) });
        return;
      }
      if (scalarOnly) {
        const tokens = keyToken.concat([token('[', 'dz-punct')]);
        entry.values.forEach((value, position) => {
          tokens.push(...jsonScalar(value));
          if (position < entry.values.length - 1) tokens.push(token(', ', 'dz-punct'));
        });
        tokens.push(token(']' + comma, 'dz-punct'));
        lines.push({ indent: inner, tokens: tokens });
        return;
      }
      lines.push({ indent: inner, tokens: keyToken.concat([token('[', 'dz-punct')]) });
      entry.values.forEach((value, position) => {
        const separator = position < entry.values.length - 1 ? ',' : '';
        if (isDescription(value)) lines.push(...jsonLines(value, inner + INDENT, collapsed, true, [], separator));
        else lines.push({ indent: inner + INDENT, tokens: jsonScalar(value).concat([token(separator, 'dz-punct')]) });
      });
      lines.push({ indent: inner, tokens: [token(']' + comma, 'dz-punct')] });
    });
    lines.push({ indent: indent, tokens: [token('}' + (suffix || ''), 'dz-punct')] });
    return lines;
  }

  function lineText(line) {
    return ' '.repeat(line.indent) + line.tokens.map((part) => part.text).join('');
  }

  function lispText(tree, collapsed) {
    return lispLines(tree, 0, collapsed || new Set()).map(lineText).join('\n');
  }

  // %% view ------------------------------------------------------------------------

  function build(container, payload) {
    const tree = payload.designator;
    let mode = 'lisp';
    let collapsed = initiallyCollapsed(tree);
    container.innerHTML =
      '<div class="dz-toolbar">' +
      '  <button data-mode="lisp">Lisp</button><button data-mode="json">JSON</button>' +
      '  <span class="dz-gap"></span>' +
      '  <button data-action="expand">Expand all</button><button data-action="collapse">Collapse all</button>' +
      '  <button data-action="copy">Copy</button>' +
      '</div>' +
      '<div class="dz-tree"></div>';
    const treeEl = container.querySelector('.dz-tree');
    const modeButtons = Array.from(container.querySelectorAll('[data-mode]'));

    function lines() {
      return mode === 'lisp' ? lispLines(tree, 0, collapsed) : jsonLines(tree, 0, collapsed, false);
    }

    function render() {
      modeButtons.forEach((button) => button.classList.toggle('on', button.dataset.mode === mode));
      treeEl.innerHTML = '';
      lines().forEach((line) => {
        const row = document.createElement('div');
        row.className = 'dz-row';
        row.style.paddingLeft = line.indent + 'ch';
        if (line.id) row.dataset.id = line.id;
        line.tokens.forEach((part) => {
          if (part.toggle !== undefined) {
            const toggle = document.createElement('span');
            toggle.className = 'dz-toggle';
            toggle.textContent = part.open ? '▾' : '▸';
            toggle.dataset.toggle = part.toggle;
            row.appendChild(toggle);
            return;
          }
          const span = document.createElement('span');
          span.className = part.cls;
          span.textContent = part.text;
          if (part.title) span.title = part.title;
          if (part.ref) span.dataset.ref = part.ref;
          row.appendChild(span);
        });
        treeEl.appendChild(row);
      });
    }

    function reveal(id) {
      (pathTo(tree, id) || []).forEach((ancestor) => collapsed.delete(ancestor));
      render();
      const row = treeEl.querySelector('[data-id="' + CSS.escape(id) + '"]');
      if (!row) return;
      row.scrollIntoView({ block: 'center', behavior: 'smooth' });
      row.classList.add('dz-flash');
      setTimeout(() => row.classList.remove('dz-flash'), 1500);
    }

    treeEl.addEventListener('click', (event) => {
      const toggle = event.target.closest('[data-toggle]');
      if (toggle) {
        const id = toggle.dataset.toggle;
        if (collapsed.has(id)) collapsed.delete(id); else collapsed.add(id);
        render();
        return;
      }
      const reference = event.target.closest('[data-ref]');
      if (reference) reveal(reference.dataset.ref);
    });
    modeButtons.forEach((button) => button.addEventListener('click', () => { mode = button.dataset.mode; render(); }));
    container.querySelector('[data-action="expand"]').addEventListener('click', () => { collapsed = new Set(); render(); });
    container.querySelector('[data-action="collapse"]').addEventListener('click', () => { collapsed = initiallyCollapsed(tree); render(); });
    container.querySelector('[data-action="copy"]').addEventListener('click', () => {
      const text = mode === 'lisp' ? lispText(tree) : JSON.stringify(toJson(tree), null, INDENT);
      navigator.clipboard.writeText(text);
    });
    render();
  }

  root.DesignatorView = { build, lispText, toJson, initiallyCollapsed, pathTo };
})(typeof window !== 'undefined' ? window : this);
