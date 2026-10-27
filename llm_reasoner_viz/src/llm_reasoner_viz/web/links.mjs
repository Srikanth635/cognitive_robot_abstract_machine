// Where the viewer sends the reader: cramera, at the scene and the run it is showing.

// The parameters of the address cramera reads.
const PARAMETER = Object.freeze({ SCENE: 'scene', RUN: 'run' });

// The address that opens a run in cramera, in its scene when the log names one: cramera
// shows what the run resolved and the hypothesis it consulted, and can perform it.
export function crameraAddress(baseUrl, scene, run) {
  const address = new URL(baseUrl);
  if (!address.pathname.endsWith('/')) address.pathname += '/';
  if (scene) address.searchParams.set(PARAMETER.SCENE, scene);
  address.searchParams.set(PARAMETER.RUN, run);
  return address.toString();
}
