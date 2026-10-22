/* ============================================================================
 * addon/designator_completion.js — what the designator box offers for the word being typed.
 *
 * The action designators by name, and, inside the argument list of one, the parameters it
 * takes. cramera's own completion (core/completion.js) finds the word under the caret and
 * ranks the matches, and its suggestion menu shows them; this decides which names are in
 * question, from the vocabulary the server offers:
 * {name, kind: 'class' | 'field', detail, module, owner}.
 * ==========================================================================*/
(function () {
  'use strict';

  const KIND = Object.freeze({ designator: 'class', parameter: 'field' });

  // where the argument list of a designator opens: `Name(` or, as krrood writes a match
  // query, `the(Name)(`
  function opening(designators) {
    return new RegExp('\\b(' + designators.join('|') + ')\\b\\)?\\s*\\(', 'g');
  }

  // the designator whose argument list the caret is in, with the text of the list so far
  function listBefore(designators, text) {
    if (!designators.length) return null;
    const pattern = opening(designators);
    let last = null;
    let match;
    while ((match = pattern.exec(text)) !== null) last = match;
    if (!last) return null;
    const written = text.slice(last.index + last[0].length);
    let depth = 0;
    for (const character of written) {
      if (character === '(') depth += 1;
      if (character === ')') depth -= 1;
      if (depth < 0) return null;
    }
    return { designator: last[1], written: written, depth: depth };
  }

  // the entries worth offering at the caret: the designators, or, where a parameter
  // can be named, the parameters of the designator being written
  function entriesAt(vocabulary, text, caret) {
    const token = Completion.tokenAt(text, caret);
    if (token.owner) return [];
    const designators = vocabulary
      .filter(function (entry) { return entry.kind === KIND.designator; })
      .map(function (entry) { return entry.name; });
    const list = listBefore(designators, text.slice(0, token.start));
    let offered;
    if (!list) {
      offered = vocabulary.filter(function (entry) { return entry.kind === KIND.designator; });
    } else if (list.depth === 0 && /(^|[,(])\s*$/.test(list.written)) {
      offered = vocabulary.filter(function (entry) {
        return entry.kind === KIND.parameter && entry.owner === list.designator;
      });
    } else {
      return [];
    }
    // a name already typed in full has nothing left to offer
    return offered.filter(function (entry) { return entry.name !== token.prefix; });
  }

  window.DesignatorCompletion = { entriesAt: entriesAt };
})();
