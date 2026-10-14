// Numbers written for reading.

const SECONDS_PER_MINUTE = 60;
const MILLISECONDS_PER_SECOND = 1000;

// A whole number with its thousands set apart.
export function formatCount(count) {
  return count.toLocaleString('en-US');
}

export function formatDuration(seconds) {
  if (seconds < 0.5 / MILLISECONDS_PER_SECOND) return '<1 ms';
  if (seconds < 1) return Math.round(seconds * MILLISECONDS_PER_SECOND) + ' ms';
  if (seconds < SECONDS_PER_MINUTE) return seconds.toFixed(2) + ' s';
  const minutes = Math.floor(seconds / SECONDS_PER_MINUTE);
  return minutes + ' min ' + Math.round(seconds - minutes * SECONDS_PER_MINUTE) + ' s';
}

// How long after the start of the run a moment is.
export function formatOffset(seconds) {
  return '+' + seconds.toFixed(seconds < 10 ? 2 : 1) + ' s';
}

export function formatCost(dollars) {
  return '$' + dollars.toFixed(dollars < 0.1 ? 4 : 2);
}
