// The controls of a replay: play and pause, a step to the next or the previous event, the
// speed, and a slider along the run.

import { el } from './dom.mjs';
import { formatDuration } from './format.mjs';
import { SPEEDS } from './replay.mjs';

export class ReplayView {
  // ``onChange`` is called after the controls have changed the replay.
  constructor(container, onChange) {
    this.replay = null;
    this.onChange = onChange;
    const act = (change) => () => { change(this.replay); onChange(); };
    this.back = el('button', { type: 'button', title: 'Go to the previous event', onclick: act((replay) => replay.step(-1)) }, '‹ event');
    this.toggle = el('button', { type: 'button', class: 'play', onclick: act((replay) => (replay.playing ? replay.pause() : replay.play())) });
    this.forward = el('button', { type: 'button', title: 'Go to the next event', onclick: act((replay) => replay.step(1)) }, 'event ›');
    this.speed = el('select', { 'aria-label': 'Replay speed', onchange: () => { this.replay.speed = Number(this.speed.value); onChange(); } },
      SPEEDS.map((speed) => el('option', { value: speed }, speed + '×')));
    this.slider = el('input', { type: 'range', min: 0, step: 'any', 'aria-label': 'Position in the run',
      oninput: () => { this.replay.seek(Number(this.slider.value)); onChange(); } });
    this.clock = el('span', { class: 'clock' });
    container.replaceChildren(this.back, this.toggle, this.forward, this.speed, this.slider, this.clock);
  }

  // Shows where a replay is; the controls do not call ``onChange`` for this.
  show(replay) {
    this.replay = replay;
    this.toggle.textContent = replay.playing ? '❚❚ pause' : '▶ play';
    this.speed.value = String(replay.speed);
    this.slider.max = String(replay.duration);
    this.slider.value = String(replay.position);
    this.clock.textContent = formatDuration(replay.position) + ' / ' + formatDuration(replay.duration);
  }
}
