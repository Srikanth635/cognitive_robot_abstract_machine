// Tests the frame that shares panels as tabs: moving the panels of a group into it, the tab
// bar showing one at a time, the page's address and the bus choosing the tab, and groups
// with nothing to share.
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const WEB = path.join(__dirname, '..', '..', '..', '..', 'cramera', 'src', 'cramera', 'web');

const GROUP = [
  { panel: 'eql', label: 'EQL Query', title: 'ask in EQL', comesForwardOn: ['entity:select', 'scene:part-clicked'] },
  { panel: 'resolution-designators', label: 'Underspecified Action Designators', title: 'resolve', openedByParameter: 'run' },
];

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
  global.window = { Bus: bus, location: { search: '' }, CRAMERA_TABS: settings.groups };
  global.URLSearchParams = URLSearchParams;
  new Function(fs.readFileSync(path.join(WEB, 'core', 'panel-tabs.js'), 'utf8'))();
  return {
    slot, otherSlot, panels, handlers,
    install() { global.window.PanelTabs.install(settings.groups || [GROUP], bus, settings.search || ''); },
    publish(event) { (handlers[event] || []).forEach(function (listener) { listener({}); }); },
    frame() { return slot.children.filter(function (child) { return child.dataset.panel === 'tabs:eql+resolution-designators'; })[0]; },
  };
}

function bar(page) { return page.frame().children[0]; }
function body(page) { return page.frame().children[1]; }
function booted(options) { const page = loadPage(options); page.install(); return page; }
function lit(page) { return bar(page).children.map(function (b) { return b.classes.has('active'); }); }
function stepped(page) {
  return [page.panels.eql, page.panels['resolution-designators']].map(function (s) { return s.classes.has('tab-inactive'); });
}

// %% installing
test('nothing is grouped before the groups are installed', function () {
  assert.strictEqual(loadPage().frame(), undefined);
});

test('the groups the page configures are installed when none are given', function () {
  const page = loadPage({ groups: [GROUP] });
  global.window.PanelTabs.install();
  assert.notStrictEqual(page.frame(), undefined);
});

test('a page with no groups is left as it is', function () {
  const page = loadPage();
  global.window.CRAMERA_TABS = undefined;
  global.window.PanelTabs.install();
  assert.strictEqual(page.frame(), undefined);
});

test('the tab that opens first is the one the address names', function () {
  loadPage();
  assert.strictEqual(global.window.PanelTabs.startingTab(GROUP, 'run=a'), 1);
});

test('the tab that opens first is the first when the address names none', function () {
  loadPage();
  assert.strictEqual(global.window.PanelTabs.startingTab(GROUP, ''), 0);
});

// %% the frame
test('the frame takes the place of the EQL panel in its column', function () {
  const page = booted();
  assert.deepStrictEqual(
    page.slot.children.map(function (child) { return child.dataset.panel; }),
    ['tabs:eql+resolution-designators', 'graph'],
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
