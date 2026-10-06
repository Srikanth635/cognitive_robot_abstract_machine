/* ============================================================================
 * panels/graph/designator.js — the Designator tab's left pane.
 *
 * Renders the reflected designator (see cram_viz.kb's "designator" view) as
 * Lisp keyword syntax, one slot per line. The right pane stays the ordinary
 * graph.js/window.Graph rendering of the same payload's nodes/edges -- this
 * module only owns the left pane and forwards clicks on a slot's line to
 * whatever the graph panel wants to do with that node id (see panel.js, which
 * wires this to the same select() a graph node click already goes through, so
 * both panes highlight, focus and describe identically).
 * ==========================================================================*/
(function () {
  let el = null;
  function resolveDom() { el = el || document.getElementById('designator-lisp'); }

  let selectCb = function () {};

  function slotLine(slot) {
    const line = document.createElement('div');
    line.className = 'designator-lisp-line';
    line.dataset.nodeId = slot.nodeId;
    line.appendChild(document.createTextNode('  '));
    const keyword = document.createElement('span');
    keyword.className = 'designator-lisp-keyword';
    keyword.textContent = slot.keyword;
    line.appendChild(keyword);
    line.appendChild(document.createTextNode(' '));
    const value = document.createElement('span');
    value.className = 'designator-lisp-value ' + (
      slot.resolved ? 'resolved' : (slot.kind === 'referent' ? 'referent' : 'value-slot')
    );
    value.textContent = slot.value;
    line.appendChild(value);
    // the same FrameRoleKind name the Hypothesis Graph tab's action frame uses for
    // this participant -- one shared vocabulary, shown as a trailing Lisp comment
    // rather than a second, differently-worded label
    if (slot.frameRole) {
      const role = document.createElement('span');
      role.className = 'designator-lisp-role';
      role.textContent = '  ; ' + slot.frameRole;
      line.appendChild(role);
    }
    line.addEventListener('click', function () { selectCb(slot.nodeId); });
    return line;
  }

  // the slot's frame-semantic property groups, nested under it as a :framenet
  // block -- the same data the graph half already shows as child nodes, reused
  // here rather than left graph-only
  function frameNetLines(slot) {
    if (!slot.frameNet || !slot.frameNet.length) return [];
    const lines = [];
    const header = document.createElement('div');
    header.className = 'designator-lisp-line designator-lisp-nested';
    header.dataset.nodeId = slot.nodeId;
    header.textContent = ':framenet';
    header.addEventListener('click', function () { selectCb(slot.nodeId); });
    lines.push(header);
    slot.frameNet.forEach(function (group) {
      const line = document.createElement('div');
      line.className = 'designator-lisp-line designator-lisp-nested designator-lisp-nested-2';
      line.dataset.nodeId = slot.nodeId;
      const keyword = document.createElement('span');
      keyword.className = 'designator-lisp-keyword';
      keyword.textContent = group.keyword;
      line.appendChild(keyword);
      line.appendChild(document.createTextNode(' ('));
      const values = document.createElement('span');
      values.className = 'designator-lisp-framenet-value';
      values.textContent = group.values.map(function (v) { return '"' + v + '"'; }).join(' ');
      line.appendChild(values);
      line.appendChild(document.createTextNode(')'));
      line.addEventListener('click', function () { selectCb(slot.nodeId); });
      lines.push(line);
    });
    return lines;
  }

  function build(payload) {
    resolveDom();
    el.innerHTML = '';
    const lisp = payload && payload.lisp;
    if (!lisp) return;
    const head = document.createElement('div');
    head.className = 'designator-lisp-line designator-lisp-static designator-lisp-head';
    head.textContent = '(' + lisp.head;
    el.appendChild(head);
    lisp.slots.forEach(function (slot) {
      el.appendChild(slotLine(slot));
      frameNetLines(slot).forEach(function (line) { el.appendChild(line); });
    });
    const tail = document.createElement('div');
    tail.className = 'designator-lisp-line designator-lisp-static';
    tail.textContent = ')';
    el.appendChild(tail);
  }

  // called back when the graph half selects a node (directly, or via our own
  // click) so the clicked/selected slot's line reads as selected too
  function highlight(nodeId) {
    resolveDom();
    el.querySelectorAll('.designator-lisp-line').forEach(function (line) {
      line.classList.toggle('active', !!nodeId && line.dataset.nodeId === nodeId);
    });
  }

  window.DesignatorLisp = {
    build: build,
    highlight: highlight,
    onSelect: function (cb) { selectCb = cb; },
  };
})();
