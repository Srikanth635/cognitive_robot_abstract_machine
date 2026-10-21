// Tests the frame that shares the EQL panel and the designator panel as two tabs: moving the
// panels into it as the page boots, the tab bar showing one at a time, the page's address
// and the EQL panel's descriptions choosing the tab, and layouts with nothing to share.
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const WEB_ADDON = path.join(
  __dirname, '..', '..', '..', 'cramera_resolution', 'src', 'cramera_resolution', 'web', 'addon',
);

// %% a page of panels in slots
function node(tag) {
  const created = {
    tag: tag, className: '', textContent: '', title: '', dataset: {}, children: [], parentNode: null,
    listeners: {}, classes: new Set(),
    classList: {
      toggle(name, force) {
        const on = force === undefined ? !this.owner.classes.has(name) : force;
        if (on) this.owner.classes.add(name); else this.owner.classes.delete(name);
      },
      contains(name) { return this.owner.classes.has(name); },
    },
    addEventListener(event, listener) { this.listeners[event] = listener; },
    detach() {
      if (!this.parentNode) return;
      this.parentNode.children = this.parentNode.children.filter(function (child) { return child !== created; });
      this.parentNode = null;
    },
    appendChild(child) {
      child.detach();
      child.parentNode = this;
      this.children.push(child);
      return child;
    },
    insertBefore(child, before) {
      child.detach();
      child.parentNode = this;
      this.children.splice(this.children.indexOf(before), 0, child);
      return child;
    },
  };
  created.classList.owner = created;
  return created;
}

function loadPage(options) {
  const settings = options || {};
  const everything = [];
  function made(tag) { const created = node(tag); everything.push(created); return created; }
  const slot = made('div');
  const otherSlot = made('div');
  const panels = {};
  ['eql', 'resolution-designators', 'graph'].forEach(function (id) {
    if (settings.without === id) return;
    const section = made('section');
    section.dataset.panel = id;
    (settings.apart === id ? otherSlot : slot).appendChild(section);
    panels[id] = section;
  });

  const handlers = {};
  const bus = { on(event, listener) { (handlers[event] = handlers[event] || []).push(listener); } };
  global.document = {
    createElement(tag) { return made(tag); },
    querySelector(selector) {
      const wanted = selector.match(/^\[data-panel="(.*)"\]$/)[1];
      return everything.filter(function (candidate) { return candidate.dataset.panel === wanted; })[0] || null;
    },
  };
  const booted = [];
  global.Panels = { boot() { booted.push(true); } };
  global.window = { Bus: bus, location: { search: settings.search || '' } };
  global.URLSearchParams = URLSearchParams;
  new Function(fs.readFileSync(path.join(WEB_ADDON, 'panel_tabs.js'), 'utf8'))();
  return {
    slot, otherSlot, panels, booted, handlers, boot() { global.Panels.boot(); },
    publish(event) { (handlers[event] || []).forEach(function (listener) { listener({}); }); },
    frame() { return slot.children.filter(function (child) { return child.dataset.panel === 'resolution-tabs'; })[0]; },
  };
}

function bar(page) { return page.frame().children[0]; }
function body(page) { return page.frame().children[1]; }
function booted(options) { const page = loadPage(options); page.boot(); return page; }
function lit(page) { return bar(page).children.map(function (b) { return b.classes.has('active'); }); }
function stepped(page) {
  return [page.panels.eql, page.panels['resolution-designators']].map(function (s) { return s.classes.has('tab-inactive'); });
}

// %% joining the page's boot
test('the page still boots its panels as it did', function () {
  const page = loadPage();
  page.boot();
  assert.deepStrictEqual(page.booted, [true]);
});

test('nothing is grouped before the page boots its panels', function () {
  assert.strictEqual(loadPage().frame(), undefined);
});

// %% the frame
test('the frame takes the place of the EQL panel in its column', function () {
  const page = booted();
  assert.deepStrictEqual(
    page.slot.children.map(function (child) { return child.dataset.panel; }),
    ['resolution-tabs', 'graph'],
  );
});

test('both panels are in the body of the frame, EQL first', function () {
  const page = booted();
  assert.deepStrictEqual(
    body(page).children.map(function (child) { return child.dataset.panel; }),
    ['eql', 'resolution-designators'],
  );
});

test('the frame is laid out in the classes cramera gives a tab group', function () {
  const page = booted();
  assert.deepStrictEqual(
    [page.frame().className, bar(page).className, body(page).className],
    ['panel-tabs', 'tab-bar', 'tab-body'],
  );
});

test('each panel has a tab named for it', function () {
  const page = booted();
  assert.deepStrictEqual(
    bar(page).children.map(function (button) { return button.textContent; }),
    ['EQL Query', 'Underspecified Action Designators'],
  );
});

// %% one tab at a time
test('the page starts on the EQL tab', function () {
  const page = booted();
  assert.deepStrictEqual(lit(page), [true, false]);
  assert.deepStrictEqual(stepped(page), [false, true]);
});

test('a tab shows its panel and steps the other aside', function () {
  const page = booted();
  bar(page).children[1].listeners.click();
  assert.deepStrictEqual(lit(page), [false, true]);
  assert.deepStrictEqual(stepped(page), [true, false]);
});

test('the first tab brings its panel back', function () {
  const page = booted();
  bar(page).children[1].listeners.click();
  bar(page).children[0].listeners.click();
  assert.deepStrictEqual(stepped(page), [false, true]);
});

test('a panel is stepped aside with a class of its own, not the one the visibility menu toggles', function () {
  const page = booted();
  assert.strictEqual(page.panels['resolution-designators'].classes.has('hidden'), false);
});

// %% what chooses the tab
test('a page opened on a resolution starts on the designators', function () {
  const page = booted({ search: '?scene=simple_pouring&run=run-1' });
  assert.deepStrictEqual(lit(page), [false, true]);
});

test('a page opened on a scene alone starts on the EQL tab', function () {
  const page = booted({ search: '?scene=simple_pouring' });
  assert.deepStrictEqual(lit(page), [true, false]);
});

['entity:select', 'scene:part-clicked'].forEach(function (event) {
  test('a ' + event + ' brings the EQL tab forward, where it is described', function () {
    const page = booted();
    bar(page).children[1].listeners.click();
    page.publish(event);
    assert.deepStrictEqual(lit(page), [true, false]);
  });
});

// %% layouts with nothing to share
test('a layout without the designator panel is left as it is', function () {
  const page = booted({ without: 'resolution-designators' });
  assert.strictEqual(page.frame(), undefined);
  assert.deepStrictEqual(page.slot.children.map(function (child) { return child.dataset.panel; }), ['eql', 'graph']);
});

test('a layout without the EQL panel is left as it is', function () {
  const page = booted({ without: 'eql' });
  assert.strictEqual(page.frame(), undefined);
});

test('panels in different columns are left where they are', function () {
  const page = booted({ apart: 'resolution-designators' });
  assert.strictEqual(page.frame(), undefined);
  assert.deepStrictEqual(page.otherSlot.children.map(function (child) { return child.dataset.panel; }), ['resolution-designators']);
});
