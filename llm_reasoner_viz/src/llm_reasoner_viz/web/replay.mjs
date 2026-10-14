// Replaying a run that has happened: what it looked like at any moment of it, and how a
// replay moves through them.

import { RunModel } from './model.mjs';

// How many seconds of the run pass in a second of replaying, to choose from.
export const SPEEDS = Object.freeze([1, 4, 16, 64]);
export const DEFAULT_SPEED = 4;

export class Replay {
  constructor(events) {
    this.events = events;
    const dates = events.filter(function (event) { return typeof event.at === 'number'; })
      .map(function (event) { return event.at; });
    this.start = dates.length === 0 ? 0 : Math.min(...dates);
    this.end = dates.length === 0 ? 0 : Math.max(...dates);
    // The moments something happened at, earliest first.
    this.moments = Array.from(new Set(dates)).sort(function (first, second) { return first - second; });
    // The moment of the run the replay is at.
    this.time = this.start;
    this.speed = DEFAULT_SPEED;
    this.playing = false;
  }

  get duration() {
    return this.end - this.start;
  }

  // How far into the run the replay is, in seconds.
  get position() {
    return this.time - this.start;
  }

  get atEnd() {
    return this.time >= this.end;
  }

  seek(seconds) {
    this.time = Math.min(this.end, Math.max(this.start, this.start + seconds));
  }

  play() {
    if (this.atEnd) this.time = this.start;
    this.playing = true;
  }

  pause() {
    this.playing = false;
  }

  // Lets time pass: a replay that is playing moves on by that much at its speed, and
  // stops when it reaches the end.
  advance(seconds) {
    if (!this.playing) return;
    this.seek(this.position + seconds * this.speed);
    if (this.atEnd) this.playing = false;
  }

  // Goes to the moment of the next event (direction 1) or of the one before (-1).
  step(direction) {
    this.playing = false;
    const time = this.time;
    const reachable = this.moments.filter(function (moment) {
      return direction > 0 ? moment > time : moment < time;
    });
    if (reachable.length === 0) return;
    this.time = direction > 0 ? reachable[0] : reachable[reachable.length - 1];
  }

  // The run as it was at the moment the replay is at: what had been written by then.
  modelAt() {
    const time = this.time;
    const model = new RunModel();
    this.events.forEach(function (event) {
      if (typeof event.at === 'number' && event.at <= time) model.add(event);
    });
    return model;
  }
}
