/* ============================================================================
 * addon/resolution_view.js — what a resolution, its trials and the runs kept read as.
 *
 * Pure functions from the payloads of /api/resolution/* to markup and text, so the
 * designator panel only wires them to the page and they are testable alone.
 * ==========================================================================*/
(function () {
  'use strict';

  const SITUATION_UNKNOWN = 'an unknown situation';

  function esc(text) {
    return String(text).replace(/[&<>]/g, function (character) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;' }[character];
    });
  }

  function attributeText(text) {
    return esc(text).replace(/"/g, '&quot;');
  }

  function queryHtml(query) {
    return '<pre class="resolved-query">' + esc(query) + '</pre>';
  }

  // where a pour tried in the simulation left the source container's mouth, against the
  // target's opening, as the simulation measured it
  function trialText(outcome) {
    if (outcome.motion_failed) return 'the motion failed: ' + outcome.motion_failed;
    return (outcome.lands_in_opening ? 'landed' : 'missed') + ': the mouth ' +
      outcome.mouth_distance_from_opening_centre + ' m from the opening\'s centre (' +
      outcome.mouth_ahead_of_opening_centre + ' m ahead, ' +
      outcome.mouth_right_of_opening_centre + ' m right), ' +
      outcome.mouth_above_opening + ' m above it';
  }

  function trialsHtml(trials) {
    if (!trials.length) return '';
    let html = '<div class="ansub">tried in the simulation</div>';
    trials.forEach(function (trial, index) {
      html += '<div class="ansrow"><span class="tag">' + (index + 1) + '</span>' +
        '<div class="body"><span class="sub">' + esc(trialText(trial.outcome)) +
        '</span></div></div>';
    });
    return html;
  }

  function actionHtml(designator, action) {
    let html = '<div class="ansub">' + esc(designator) + '</div>';
    action.slots.forEach(function (slot) {
      html += '<div class="ansrow"><span class="tag">' + (slot.given ? 'given' : 'chosen') +
        '</span><div class="body"><span class="name">' + esc(slot.name) + ' = ' +
        esc(slot.value) + '</span></div></div>';
    });
    html += '<div class="ansrow"><div class="body"><span class="sub">why: ' +
      esc(action.why) + '</span></div></div>';
    action.grounds.forEach(function (ground) {
      html += '<div class="ansrow"><div class="body"><span class="sub">rests on ' +
        esc(ground.source) + ': ' + esc(ground.name) + '</span></div></div>';
    });
    return html;
  }

  function situationText(situation) {
    const features = Object.keys(situation).map(function (feature) {
      return feature + ' = ' + situation[feature];
    });
    return features.length ? features.join(', ') : SITUATION_UNKNOWN;
  }

  function consultedText(resolution) {
    const perspectives = resolution.perspective_record_ids || [];
    return 'knowledge consulted for ' + esc(situationText(resolution.situation)) +
      (resolution.record_id ? ' (hypothesis ' + esc(resolution.record_id) + ')' : '') +
      (perspectives.length
        ? ' with the action core and perspectives ' + esc(perspectives.join(', '))
        : '');
  }

  function resolutionHtml(resolution, viewer) {
    let html = '<div class="goal">resolved</div>' + queryHtml(resolution.query);
    resolution.actions.forEach(function (action) {
      html += actionHtml(resolution.designator, action);
    });
    html += trialsHtml(resolution.trials || []);
    html += '<p class="hint-txt">' + consultedText(resolution) +
      ' · resolver cost $' + resolution.cost.toFixed(3) +
      ' · run ' + esc(resolution.run) +
      (viewer ? ' · <a href="' + attributeText(viewer) + '" target="_blank" rel="noopener">' +
        'inspect it in the llm_reasoner viewer</a>' : '') +
      '</p>';
    return html;
  }

  // why a designator cannot be resolved in this scene, or how to use it
  function designatorHint(designator) {
    if (designator.available) {
      return designator.identifier + ': click to copy it into the search bar, then Resolve';
    }
    const lacks = designator.unbound.map(function (name) {
      return 'an object named ' + name;
    }).concat(designator.without_candidates.map(function (slot) {
      return 'a candidate for ' + slot;
    }));
    return designator.identifier + ' needs ' + lacks.join(', ') + ', which this scene lacks';
  }

  // one run of the list: which scene and designator, and the query it was made from
  function runText(run) {
    return run.scene + ' · ' + run.designator + ' · ' + run.query;
  }

  function runTitle(run) {
    return 'run ' + run.name + ' · ' + run.at + ' · resolver cost $' + run.cost.toFixed(3) +
      ' · ' + run.trials + ' pour' + (run.trials === 1 ? '' : 's') + ' tried — click to show it';
  }

  window.ResolutionView = {
    esc: esc,
    attributeText: attributeText,
    queryHtml: queryHtml,
    trialText: trialText,
    trialsHtml: trialsHtml,
    resolutionHtml: resolutionHtml,
    designatorHint: designatorHint,
    runText: runText,
    runTitle: runTitle,
  };
})();
