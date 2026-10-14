// Where spans lie on a timeline: a time is a distance from the origin, and spans that would
// sit on one another take rows of their own.

export function xOf(time, origin, pixelsPerSecond) {
  return (time - origin) * pixelsPerSecond;
}

// How many pixels a second takes for a run to fit the width shown, never fewer than the
// least that keeps a second visible.
export function fitScale(duration, width, leastPixelsPerSecond = 0) {
  if (duration <= 0) return width;
  return Math.max(width / duration, leastPixelsPerSecond);
}

// The spans placed from the earliest to the latest start, each with its position, its
// width and the row it takes. Whether two spans meet is judged on what is drawn, so spans
// too short to tell apart do not hide one another. A span with no end lasts until now.
export function packTracks(spans, settings) {
  const ordered = spans.slice().sort(function (first, second) {
    if (first.start !== second.start) return first.start - second.start;
    const firstEnd = first.end === null ? Infinity : first.end;
    const secondEnd = second.end === null ? Infinity : second.end;
    return secondEnd - firstEnd;
  });
  const rowEnds = [];
  return ordered.map(function (span) {
    const end = span.end === null ? settings.now : span.end;
    const x = xOf(span.start, settings.origin, settings.pxPerSecond);
    const width = Math.max(settings.minWidth, (end - span.start) * settings.pxPerSecond);
    let track = rowEnds.findIndex(function (rowEnd) { return rowEnd + settings.gap <= x; });
    if (track === -1) track = rowEnds.length;
    rowEnds[track] = x + width;
    return { span: span, x: x, width: width, track: track };
  });
}
