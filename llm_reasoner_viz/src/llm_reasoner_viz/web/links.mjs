// Where the viewer sends the reader: cram-viz, at the scene and the run it is showing.

// The parameters of the address cram-viz reads.
const PARAMETER = Object.freeze({ SCENE: 'scene', RUN: 'run' });

// The address that opens a run in cram-viz, in its scene when the log names one: cram-viz
// shows what the run resolved and the hypothesis it consulted, and can perform it.
export function cramVizAddress(baseUrl, scene, run) {
  const address = new URL(baseUrl);
  if (!address.pathname.endsWith('/')) address.pathname += '/';
  if (scene) address.searchParams.set(PARAMETER.SCENE, scene);
  address.searchParams.set(PARAMETER.RUN, run);
  return address.toString();
}
