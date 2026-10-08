/* ============================================================================
 * panels/graph/ontology-graph.js — the Ontology Graph tab: the ontology behind the
 * active hypothesis, drawn at the level of classes.
 *
 * Nodes are the microworld classes the hypothesis instantiates, the classes they
 * specialise up to CRAM-AICOR and DUL, and the concepts the hypothesis cites; edges
 * are is-a links, the structure's components, the relations the hypothesis states,
 * cited concepts, disjointness, and the defined classes whose definition instances
 * meet, which is where their requirements come from. The layout follows the L2 ontology graph: an SVG
 * force layout that cools and settles, a legend whose entries toggle categories and
 * edge kinds, three levels of detail, a search box and a panel describing the
 * selected class.
 *
 * OntologyGraphModel holds everything that decides what is shown and described, so it
 * can be tested without a browser; build() draws it.
 * ==========================================================================*/
(function (root) {
  'use strict';

  const CATEGORY = {
    used: { color: '#155ea8', text: 'class the hypothesis instantiates', radius: 11 },
    microworld: { color: '#3b7dd8', text: 'microworld class without instances', radius: 8 },
    linked: { color: '#7a869a', text: 'CRAM-AICOR / DUL', radius: 8 },
    introduced: { color: '#c0392b', text: 'concept the hypothesis introduced', radius: 6, dashed: true },
  };
  const KIND = {
    is_a: { color: '#334155', text: 'is-a (subClassOf)', arrow: true },
    component: { color: '#155ea8', text: 'component the structure declares', arrow: true },
    relation: { color: '#2e8b7a', text: 'relation the hypothesis states', arrow: true },
    cites: { color: '#b7791f', text: 'concept cited', arrow: true },
    disjoint: { color: '#c0392b', text: 'disjoint-with', dashed: true },
    classified_as: { color: '#7c3aed', text: 'classified by a definition', arrow: true, dashed: true },
  };
  // level of detail: the categories each level leaves out
  const LEVELS = [
    { name: 'Hypothesis', hidden: ['microworld', 'linked', 'introduced'] },
    { name: '+ backing', hidden: ['introduced'] },
    { name: 'All', hidden: [] },
  ];
  // edge kinds too numerous to draw by default; the legend switches them on
  const HIDDEN_KINDS = ['relation', 'disjoint'];
  // how many matches the search lists by name; the rest are only counted
  const MATCHES_LISTED = 30;
  const INITIAL_LEVEL = 1;

  // force layout with cooling, as in the L2 ontology graph
  const REPULSION = 1700, LINK_LENGTH = 92, SPRING = 0.02, GRAVITY = 0.0019;
  const DAMPING = 0.9, SPEED_LIMIT = 26, REPULSION_LIMIT = 45;
  const ALPHA_MINIMUM = 0.004, ALPHA_DECAY = 0.02;

  function escapeHtml(text) {
    return String(text).replace(/[&<>"']/g, function (character) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[character];
    });
  }

  // %% model ---------------------------------------------------------------------

  function OntologyGraphModel(payload) {
    this.nodes = payload.nodes || [];
    this.edges = payload.edges || [];
    this.details = payload.details || {};
    this.byId = {};
    this.nodes.forEach((node) => { this.byId[node.id] = node; });
    this.edges = this.edges.filter((edge) => this.byId[edge.source] && this.byId[edge.target]);
    this.hiddenKinds = new Set(HIDDEN_KINDS);
    this.setLevel(INITIAL_LEVEL);
  }

  OntologyGraphModel.prototype.setLevel = function (level) {
    this.level = level;
    this.hiddenCategories = new Set(LEVELS[level].hidden);
  };

  OntologyGraphModel.prototype.toggleCategory = function (category) {
    if (this.hiddenCategories.has(category)) this.hiddenCategories.delete(category);
    else this.hiddenCategories.add(category);
  };

  OntologyGraphModel.prototype.toggleKind = function (kind) {
    if (this.hiddenKinds.has(kind)) this.hiddenKinds.delete(kind);
    else this.hiddenKinds.add(kind);
  };

  OntologyGraphModel.prototype.isNodeVisible = function (node) {
    return !this.hiddenCategories.has(node.category);
  };

  OntologyGraphModel.prototype.isEdgeVisible = function (edge) {
    return !this.hiddenKinds.has(edge.kind)
      && this.isNodeVisible(this.byId[edge.source])
      && this.isNodeVisible(this.byId[edge.target]);
  };

  OntologyGraphModel.prototype.edgeLabel = function (edge) {
    if (!edge.label) return '';
    return edge.count > 1 ? edge.label + ' ×' + edge.count : edge.label;
  };

  OntologyGraphModel.prototype.nodeLabel = function (node) {
    return node.count ? node.label + ' ×' + node.count : node.label;
  };

  OntologyGraphModel.prototype.matches = function (node, query) {
    const text = query.toLowerCase().trim();
    if (!text) return true;
    const detail = this.details[node.id] || {};
    return [node.label, detail.prefixed, detail.description]
      .some((part) => (part || '').toLowerCase().includes(text));
  };

  // the classes a search finds, those the hypothesis uses first, whatever is hidden
  OntologyGraphModel.prototype.matching = function (query) {
    if (!query.trim()) return [];
    const rank = Object.keys(CATEGORY);
    return this.nodes
      .filter((node) => this.matches(node, query))
      .sort((a, b) => rank.indexOf(a.category) - rank.indexOf(b.category));
  };

  // a class and the classes its drawn links reach; links switched off in the legend,
  // or to classes the level hides, highlight nothing
  OntologyGraphModel.prototype.connected = function (id) {
    const found = new Set([id]);
    this.edges.forEach((edge) => {
      if (!this.isEdgeVisible(edge)) return;
      if (edge.source === id) found.add(edge.target);
      if (edge.target === id) found.add(edge.source);
    });
    return found;
  };

  OntologyGraphModel.prototype.summary = function () {
    const count = (category) => this.nodes.filter((node) => node.category === category).length;
    return this.nodes.length + ' classes · ' + count('used') + ' instantiated · '
      + count('linked') + ' CRAM-AICOR/DUL · ' + count('introduced') + ' introduced · '
      + this.edges.length + ' links';
  };

  OntologyGraphModel.prototype.panelHtml = function (id) {
    const node = this.byId[id];
    const detail = this.details[id] || {};
    const category = CATEGORY[node.category];
    const link = (target) => '<a href="#" data-select="' + escapeHtml(target) + '">'
      + escapeHtml(this.byId[target].label) + '</a>';
    const group = (title, rows) => rows.length
      ? '<div class="grp">' + title + '</div>' + rows.map((row) => '<div class="rel">' + row + '</div>').join('')
      : '';
    const outgoing = (kind) => this.edges.filter((edge) => edge.source === id && edge.kind === kind);
    let html = '<h2>' + escapeHtml(node.label) + ' <span style="color:' + category.color
      + ';font-size:.7rem">' + category.text + '</span></h2>';
    html += '<p class="sub"><code>' + escapeHtml(detail.prefixed || id) + '</code>'
      + (detail.source ? ' · ' + escapeHtml(detail.source) : '')
      + (detail.evidence ? ' · ' + escapeHtml(detail.evidence) : '') + '</p>';
    if (detail.description) html += '<p class="desc">' + escapeHtml(detail.description) + '</p>';
    const undrawn = this.edges.filter((edge) =>
      (edge.source === id || edge.target === id) && !this.isEdgeVisible(edge)).length;
    if (undrawn) {
      html += '<p class="sub">' + undrawn + ' of its links are not drawn: switch their kind on in the legend, or show more classes. They are all listed below.</p>';
    }
    if (detail.definition) html += group('defined as', [escapeHtml(detail.definition)]);
    html += group('requires', (detail.requirements || []).map(escapeHtml));
    html += group('literature', (detail.claims || []).map((claim) =>
      '<b>' + escapeHtml(claim.id) + '</b> ' + escapeHtml(claim.text)
      + (claim.evidence ? ' <span class="muted">(' + escapeHtml(claim.evidence) + ')</span>' : '')));
    html += group('is-a', outgoing('is_a').map((edge) => '▸ ' + link(edge.target)));
    const classified = (edge) => edge.count + ' of ' + this.byId[edge.source].count + ' instances';
    html += group('classified as', outgoing('classified_as').map((edge) =>
      '▸ ' + link(edge.target) + ' <span class="muted">(' + classified(edge) + ')</span>'));
    html += group('classifies', this.edges
      .filter((edge) => edge.target === id && edge.kind === 'classified_as')
      .map((edge) => '◂ ' + link(edge.source) + ' <span class="muted">(' + classified(edge) + ')</span>'));
    html += group('components', outgoing('component').map((edge) =>
      '<b>' + escapeHtml(this.edgeLabel(edge)) + '</b> ' + link(edge.target)));
    html += group('relations', outgoing('relation').map((edge) =>
      '<b>' + escapeHtml(this.edgeLabel(edge)) + '</b> → ' + link(edge.target)));
    html += group('cites', outgoing('cites').map((edge) =>
      '<b>' + escapeHtml(this.edgeLabel(edge)) + '</b> → ' + link(edge.target)));
    html += group('disjoint from', this.edges
      .filter((edge) => edge.kind === 'disjoint' && (edge.source === id || edge.target === id))
      .map((edge) => '⟂ ' + link(edge.source === id ? edge.target : edge.source)));
    const instances = detail.instances || [];
    const more = (detail.count || 0) - instances.length;
    html += group('in this hypothesis', instances.map(escapeHtml)
      .concat(more > 0 ? ['… and ' + more + ' more'] : []));
    html += group('referenced by', this.edges
      .filter((edge) => edge.target === id && edge.kind !== 'disjoint' && edge.kind !== 'classified_as')
      .map((edge) => '← ' + link(edge.source) + ' <span class="muted">('
        + escapeHtml(this.edgeLabel(edge) || KIND[edge.kind].text) + ')</span>'));
    return html;
  };

  // %% view ----------------------------------------------------------------------

  const SVG_NAMESPACE = 'http://www.w3.org/2000/svg';
  // the layout and the window listeners of the graph drawn last, released on a rebuild
  let release = function () {};

  function svgElement(name, attributes) {
    const element = document.createElementNS(SVG_NAMESPACE, name);
    Object.keys(attributes || {}).forEach((key) => element.setAttribute(key, attributes[key]));
    return element;
  }

  function build(container, payload) {
    release();
    const model = new OntologyGraphModel(payload);
    container.innerHTML =
      '<div class="og-layout">' +
      '<svg class="og-canvas"></svg>' +
      '<aside class="og-side">' +
      '  <h1>Ontology Graph</h1>' +
      '  <p class="sub">' + escapeHtml(payload.instruction || '') + '</p>' +
      '  <p class="sub">' + escapeHtml(model.summary()) + '</p>' +
      '  <input class="og-search" placeholder="Find a class or concept…">' +
      '  <div class="sub og-hits"></div>' +
      '  <div class="reveal">' + LEVELS.map((level, index) =>
        '<button data-level="' + index + '">' + level.name + '</button>').join('') + '</div>' +
      '  <div class="og-legend"></div>' +
      '  <div class="og-panel"><p class="sub">Click a class to see what the ontology says about it and how the hypothesis uses it.</p></div>' +
      '</aside>' +
      '</div>';
    const svg = container.querySelector('.og-canvas');
    const panel = container.querySelector('.og-panel');
    const legend = container.querySelector('.og-legend');
    const search = container.querySelector('.og-search');
    const hits = container.querySelector('.og-hits');
    const levelButtons = Array.from(container.querySelectorAll('.reveal button'));

    svg.appendChild(svgElement('defs')).innerHTML =
      '<marker id="og-arrow" viewBox="0 0 10 10" refX="18" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="#334155"/></marker>';
    const layers = svgElement('g');
    const edgeLayer = svgElement('g'), labelLayer = svgElement('g'), nodeLayer = svgElement('g');
    layers.append(edgeLayer, labelLayer, nodeLayer);
    svg.appendChild(layers);

    let width = svg.clientWidth || 900, height = svg.clientHeight || 600;
    const positions = {};
    model.nodes.forEach((node, index) => {
      const angle = index / model.nodes.length * 2 * Math.PI;
      const radius = Math.min(width, height) * 0.42 * (0.6 + Math.random() * 0.4);
      positions[node.id] = { x: width / 2 + Math.cos(angle) * radius, y: height / 2 + Math.sin(angle) * radius, vx: 0, vy: 0 };
    });

    const edgeViews = model.edges.map((edge) => {
      const kind = KIND[edge.kind];
      const line = svgElement('line', { class: 'og-edge', stroke: kind.color });
      if (kind.dashed) line.setAttribute('stroke-dasharray', '4 3');
      if (kind.arrow) line.setAttribute('marker-end', 'url(#og-arrow)');
      edgeLayer.appendChild(line);
      let label = null;
      const text = model.edgeLabel(edge);
      if (text) {
        label = svgElement('text', { class: 'og-elabel' });
        label.textContent = text;
        labelLayer.appendChild(label);
      }
      return { edge, line, label };
    });
    const nodeViews = model.nodes.map((node) => {
      const category = CATEGORY[node.category];
      const group = svgElement('g', { class: 'og-node' });
      const circle = svgElement('circle', { r: category.radius, fill: category.color, stroke: '#fff' });
      if (category.dashed) {
        circle.setAttribute('stroke', category.color);
        circle.setAttribute('stroke-dasharray', '2 2');
        circle.setAttribute('fill', '#fff');
      }
      const label = svgElement('text', { x: category.radius + 3, y: 4 });
      label.textContent = model.nodeLabel(node);
      group.append(circle, label);
      group.addEventListener('click', (event) => { event.stopPropagation(); select(node.id); });
      group.addEventListener('mousedown', (event) => startDrag(event, node.id));
      nodeLayer.appendChild(group);
      return { node, group };
    });

    // %% layout
    let alpha = 0, frame = null, dragged = null, dragOffset = null;
    function visibleNodes() { return model.nodes.filter((node) => model.isNodeVisible(node)); }
    function tick() {
      const live = visibleNodes();
      for (let i = 0; i < live.length; i++) {
        const a = positions[live[i].id];
        for (let j = i + 1; j < live.length; j++) {
          const b = positions[live[j].id];
          let dx = a.x - b.x, dy = a.y - b.y;
          const distanceSquared = dx * dx + dy * dy || 1, distance = Math.sqrt(distanceSquared);
          const force = Math.min(REPULSION / distanceSquared, REPULSION_LIMIT) * alpha;
          dx /= distance; dy /= distance;
          a.vx += dx * force; a.vy += dy * force; b.vx -= dx * force; b.vy -= dy * force;
        }
      }
      model.edges.forEach((edge) => {
        if (!model.isEdgeVisible(edge)) return;
        const a = positions[edge.source], b = positions[edge.target];
        let dx = b.x - a.x, dy = b.y - a.y;
        const distance = Math.sqrt(dx * dx + dy * dy) || 1, force = (distance - LINK_LENGTH) * SPRING * alpha;
        dx /= distance; dy /= distance;
        a.vx += dx * force; a.vy += dy * force; b.vx -= dx * force; b.vy -= dy * force;
      });
      live.forEach((node) => {
        const p = positions[node.id];
        p.vx += (width / 2 - p.x) * GRAVITY * alpha; p.vy += (height / 2 - p.y) * GRAVITY * alpha;
        if (node.id === dragged) return;
        p.vx *= DAMPING; p.vy *= DAMPING;
        const speed = Math.hypot(p.vx, p.vy);
        if (speed > SPEED_LIMIT) { p.vx *= SPEED_LIMIT / speed; p.vy *= SPEED_LIMIT / speed; }
        p.x += p.vx; p.y += p.vy;
      });
      draw();
    }
    function loop() {
      alpha += (0 - alpha) * ALPHA_DECAY;
      tick();
      if (alpha > ALPHA_MINIMUM) frame = requestAnimationFrame(loop);
      else { alpha = 0; frame = null; }
    }
    function reheat(value) {
      alpha = Math.max(alpha, value === undefined ? 0.7 : value);
      if (!frame) frame = requestAnimationFrame(loop);
    }
    function draw() {
      edgeViews.forEach(({ edge, line, label }) => {
        const a = positions[edge.source], b = positions[edge.target];
        line.setAttribute('x1', a.x); line.setAttribute('y1', a.y);
        line.setAttribute('x2', b.x); line.setAttribute('y2', b.y);
        if (label) { label.setAttribute('x', (a.x + b.x) / 2); label.setAttribute('y', (a.y + b.y) / 2); }
      });
      nodeViews.forEach(({ node, group }) => {
        const p = positions[node.id];
        group.setAttribute('transform', 'translate(' + p.x + ',' + p.y + ')');
      });
    }

    // %% interaction
    let viewBox = { x: 0, y: 0, w: width, h: height };
    function applyViewBox() { svg.setAttribute('viewBox', viewBox.x + ' ' + viewBox.y + ' ' + viewBox.w + ' ' + viewBox.h); }
    function toSvg(event) {
      const box = svg.getBoundingClientRect();
      return { x: viewBox.x + (event.clientX - box.left) / box.width * viewBox.w, y: viewBox.y + (event.clientY - box.top) / box.height * viewBox.h };
    }
    function startDrag(event, id) {
      dragged = id;
      const point = toSvg(event);
      dragOffset = [point.x - positions[id].x, point.y - positions[id].y];
      event.preventDefault();
      reheat(0.5);
    }
    let pan = null;
    svg.addEventListener('mousedown', (event) => {
      if (event.target === svg) pan = [event.clientX, event.clientY, viewBox.x, viewBox.y];
    });
    function onMouseMove(event) {
      if (dragged) {
        const point = toSvg(event), p = positions[dragged];
        p.x = point.x - dragOffset[0]; p.y = point.y - dragOffset[1]; p.vx = p.vy = 0;
        reheat(0.3);
      } else if (pan) {
        const box = svg.getBoundingClientRect();
        viewBox.x = pan[2] - (event.clientX - pan[0]) / box.width * viewBox.w;
        viewBox.y = pan[3] - (event.clientY - pan[1]) / box.height * viewBox.h;
        applyViewBox();
      }
    }
    function onMouseUp() { dragged = null; pan = null; }
    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);
    const sizeWatcher = new ResizeObserver(() => resize());
    release = function () {
      sizeWatcher.disconnect();
      if (frame) cancelAnimationFrame(frame);
      frame = null;
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseup', onMouseUp);
    };
    svg.addEventListener('wheel', (event) => {
      event.preventDefault();
      const scale = event.deltaY > 0 ? 1.1 : 0.9, point = toSvg(event);
      viewBox.x = point.x - (point.x - viewBox.x) * scale; viewBox.y = point.y - (point.y - viewBox.y) * scale;
      viewBox.w *= scale; viewBox.h *= scale;
      applyViewBox();
    }, { passive: false });

    function centre(id) {
      const p = positions[id];
      viewBox.x = p.x - viewBox.w / 2;
      viewBox.y = p.y - viewBox.h / 2;
      applyViewBox();
    }
    function select(id) {
      const keep = model.connected(id);
      nodeViews.forEach(({ group }) => group.classList.remove('match'));
      nodeViews.forEach(({ node, group }) => {
        group.classList.toggle('dim', !keep.has(node.id));
        group.classList.toggle('hi', node.id === id);
      });
      edgeViews.forEach(({ edge, line, label }) => {
        const touching = (edge.source === id || edge.target === id) && model.isEdgeVisible(edge);
        line.style.opacity = touching ? 1 : 0.08;
        if (label) label.style.opacity = touching ? 1 : 0.08;
      });
      panel.innerHTML = model.panelHtml(id);
    }
    // a class named in the panel or in the search results is selected and centred
    function onSelectLink(event) {
      const target = event.target.closest('[data-select]');
      if (!target) return;
      event.preventDefault();
      const id = target.dataset.select;
      // matches the level hides were shown for the search; only the chosen one stays
      applyVisibility();
      nodeViews.forEach(({ node, group }) => { if (node.id === id) group.style.display = ''; });
      select(id);
      centre(id);
    }
    panel.addEventListener('click', onSelectLink);
    hits.addEventListener('click', onSelectLink);
    svg.addEventListener('click', () => {
      nodeViews.forEach(({ group }) => group.classList.remove('dim', 'hi'));
      edgeViews.forEach(({ line, label }) => { line.style.opacity = 1; if (label) label.style.opacity = 1; });
    });

    function applyVisibility() {
      nodeViews.forEach(({ node, group }) => { group.style.display = model.isNodeVisible(node) ? '' : 'none'; });
      edgeViews.forEach(({ edge, line, label }) => {
        const shown = model.isEdgeVisible(edge);
        line.style.display = shown ? '' : 'none';
        if (label) label.style.display = shown ? '' : 'none';
      });
      legend.querySelectorAll('[data-category]').forEach((row) =>
        row.classList.toggle('off', model.hiddenCategories.has(row.dataset.category)));
      legend.querySelectorAll('[data-kind]').forEach((row) =>
        row.classList.toggle('off', model.hiddenKinds.has(row.dataset.kind)));
      levelButtons.forEach((button, index) => button.classList.toggle('on', index === model.level));
    }
    legend.innerHTML =
      '<div class="lg-title">classes</div>' + Object.keys(CATEGORY).map((key) =>
        '<div data-category="' + key + '"><span class="swatch" style="background:' + CATEGORY[key].color + '"></span>' + CATEGORY[key].text + '</div>').join('') +
      '<div class="lg-title">links</div>' + Object.keys(KIND).map((key) =>
        '<div data-kind="' + key + '"><span class="swatch" style="background:' + KIND[key].color + '"></span>' + KIND[key].text + '</div>').join('');
    legend.querySelectorAll('[data-category]').forEach((row) => row.addEventListener('click', () => {
      model.toggleCategory(row.dataset.category); applyVisibility(); reheat();
    }));
    legend.querySelectorAll('[data-kind]').forEach((row) => row.addEventListener('click', () => {
      model.toggleKind(row.dataset.kind); applyVisibility(); reheat(0.3);
    }));
    levelButtons.forEach((button, index) => button.addEventListener('click', () => {
      model.setLevel(index); applyVisibility(); reheat(0.9);
    }));
    function clearSearch() {
      nodeViews.forEach(({ group }) => group.classList.remove('dim', 'match'));
      edgeViews.forEach(({ line, label }) => { line.style.opacity = 1; if (label) label.style.opacity = 1; });
      applyVisibility();
      hits.innerHTML = '';
    }
    search.addEventListener('input', () => {
      const found = model.matching(search.value);
      if (!search.value.trim()) { clearSearch(); return; }
      const ids = new Set(found.map((node) => node.id));
      nodeViews.forEach(({ node, group }) => {
        const matching = ids.has(node.id);
        group.classList.toggle('dim', !matching);
        group.classList.toggle('match', matching);
        if (matching) group.style.display = '';
      });
      edgeViews.forEach(({ edge, line, label }) => {
        const between = ids.has(edge.source) && ids.has(edge.target);
        line.style.opacity = between ? 1 : 0.05;
        if (label) label.style.opacity = between ? 1 : 0.05;
      });
      hits.innerHTML = found.length + ' match' + (found.length === 1 ? '' : 'es') +
        (found.length ? '<div class="og-matches">' + found.slice(0, MATCHES_LISTED).map((node) =>
          '<a href="#" data-select="' + escapeHtml(node.id) + '"><span class="swatch" style="background:' +
          CATEGORY[node.category].color + '"></span>' + escapeHtml(model.nodeLabel(node)) + '</a>').join('') +
          (found.length > MATCHES_LISTED ? '<span class="muted">… and ' + (found.length - MATCHES_LISTED) + ' more</span>' : '') +
          '</div>' : '');
    });
    search.addEventListener('keydown', (event) => {
      if (event.key !== 'Enter') return;
      const first = hits.querySelector('[data-select]');
      if (first) first.click();
    });

    function resize() {
      width = svg.clientWidth || width; height = svg.clientHeight || height;
      viewBox = { x: 0, y: 0, w: width, h: height };
      applyViewBox();
      reheat(0.3);
    }
    applyVisibility();
    applyViewBox();
    sizeWatcher.observe(svg);
    reheat(0.9);
    return { model, resize };
  }

  root.OntologyGraph = { build, OntologyGraphModel, LEVELS, HIDDEN_KINDS, INITIAL_LEVEL };
})(typeof window !== 'undefined' ? window : this);
