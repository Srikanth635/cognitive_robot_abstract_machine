/* ============================================================================
 * panel-groups.js — which panels share one frame as tabs (core/panel-tabs.js).
 *
 * Each group lists its panels in the order of their tabs, with the tab's label and when the
 * panel comes forward besides a click on its tab: opened by a query parameter of the
 * page's address, or brought forward by an event on the bus. A panel the layout does not
 * mount leaves its group as it is.
 * ==========================================================================*/
window.CRAMERA_TABS = [
  [
    {
      panel: 'eql', label: 'EQL Query', title: 'ask the knowledge base a question in EQL',
      comesForwardOn: ['entity:select', 'scene:part-clicked'],
    },
    {
      panel: 'resolution-designators', label: 'Underspecified Action Designators',
      title: 'resolve an underspecified action with llm_reasoner', openedByParameter: 'run',
    },
  ],
];
