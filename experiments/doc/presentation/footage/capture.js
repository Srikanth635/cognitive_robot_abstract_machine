// Captures the browser footage and screenshots for the deck from the running viewers:
// cram_viz at http://localhost:8711 and llm_reasoner_viz at http://127.0.0.1:8721.
// Chrome is driven through puppeteer-core on the display, so the three.js scene renders
// on the GPU; a clip is recorded as WebM and encoded to H.264 by encode.sh.
//
//   NODE_PATH=<dir with node_modules> node footage/capture.js execute <run> <out.webm> [seconds]
//   NODE_PATH=<dir with node_modules> node footage/capture.js replay <run> <out.webm> [seconds]
//   NODE_PATH=<dir with node_modules> node footage/capture.js shots <run> <out directory>

const fs = require("fs");
const path = require("path");
const puppeteer = require("puppeteer-core");

const CRAM_VIZ = "http://localhost:8711";
const VIEWER = "http://127.0.0.1:8721";
const SCENE = "simple_pouring";
const SCALE = Number(process.env.VIEWPORT_SCALE || 1);
const WIDTH = 1920 * SCALE, HEIGHT = 1080 * SCALE;

async function browser() {
  return puppeteer.launch({
    executablePath: "/usr/bin/google-chrome",
    headless: false,
    defaultViewport: { width: WIDTH, height: HEIGHT },
    args: [`--window-size=${WIDTH},${HEIGHT + 90}`, "--window-position=0,0", "--no-first-run", "--disable-infobars", "--hide-scrollbars", "--autoplay-policy=no-user-gesture-required"],
  });
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// %% a fresh resolution, with the viewer recorded while it follows the run live
const http = require("http");
const os = require("os");
const RUNS = path.join(os.homedir(), ".cram_viz", "runs");

function post(url, body) {
  return new Promise((resolve, reject) => {
    const data = JSON.stringify(body);
    const request = http.request(url, { method: "POST", headers: { "Content-Type": "application/json", "Content-Length": Buffer.byteLength(data) }, timeout: 0 }, (response) => {
      let text = "";
      response.on("data", (chunk) => { text += chunk; });
      response.on("end", () => resolve(JSON.parse(text)));
    });
    request.on("error", reject);
    request.write(data);
    request.end();
  });
}

function get(url) {
  return new Promise((resolve, reject) => {
    http.get(url, (response) => {
      let text = "";
      response.on("data", (chunk) => { text += chunk; });
      response.on("end", () => resolve(JSON.parse(text)));
    }).on("error", reject);
  });
}

async function resolve(designatorName, out, recordFile) {
  const designators = await get(`${CRAM_VIZ}/api/designators?scene=${SCENE}`);
  const designator = designators.designators.find((d) => d.identifier === designatorName);
  if (!designator) throw new Error(`no designator ${designatorName}; have ${designators.designators.map((d) => d.identifier).join(", ")}`);
  const before = new Set(fs.readdirSync(RUNS));
  const started = Date.now();
  const resolution = post(`${CRAM_VIZ}/api/designator/resolve`, { query: designator.query, scene: SCENE });
  let run = null;
  while (!run) {
    await sleep(2000);
    const fresh = fs.readdirSync(RUNS).filter((d) => !before.has(d) && fs.existsSync(path.join(RUNS, d, "events.jsonl")));
    if (fresh.length) run = fresh[0];
    if (Date.now() - started > 600000) throw new Error("no run appeared");
  }
  console.log("run", run);
  const b = await browser();
  const page = await b.newPage();
  await page.goto(`${VIEWER}/?run=${run}`, { waitUntil: "networkidle2", timeout: 120000 });
  await sleep(2000);
  const recorder = await page.screencast({ path: out });
  const payload = await resolution;
  await sleep(6000);
  await recorder.stop();
  await page.screenshot({ path: out.replace(/\.webm$/, "_end.png") });
  await b.close();
  fs.writeFileSync(recordFile, JSON.stringify({ run, designator: designatorName, seconds: (Date.now() - started) / 1000, payload }, null, 2));
  console.log("resolved", run, payload.ok, "in", (Date.now() - started) / 1000, "s");
}

// %% cram_viz: open a run, perform it under the live bridge, record the scene
async function execute(run, out, seconds) {
  const b = await browser();
  const page = await b.newPage();
  await page.goto(`${CRAM_VIZ}/?scene=${SCENE}&run=${run}`, { waitUntil: "networkidle2", timeout: 120000 });
  await page.waitForFunction(() => { const e = document.querySelector("#designator-execute"); return e && !e.disabled; }, { timeout: 120000 });
  await sleep(3000);
  await page.click("#designator-execute");
  await sceneOnly(page);
  const recorder = await page.screencast({ path: out });
  // the Live button appears once the bridge answers; attach to it
  await page.waitForFunction(() => { const e = document.querySelector("#live-btn"); return e && e.style.display !== "none"; }, { timeout: 180000 });
  await page.click("#live-btn");
  await sleep(seconds * 1000);
  await recorder.stop();
  await b.close();
}

// %% the 3D scene alone, filling the page: the other panels hidden for the recording, the camera closer
async function sceneOnly(page) {
  if (process.env.SCENE_ONLY === "0") return;
  await page.addStyleTag({ content: `
    .panel-robot-scene { position: fixed !important; inset: 0 !important; z-index: 9999 !important; width: 100vw !important; height: 100vh !important; max-height: none !important; display: flex !important; flex-direction: column !important; }
    .panel-robot-scene > *:not(.panel-head) { flex: 1 1 auto !important; min-height: 0 !important; }
    .panel-robot-scene .stage { height: calc(100vh - 120px) !important; min-height: calc(100vh - 120px) !important; }
    .panel-robot-scene canvas { width: 100% !important; height: 100% !important; }
  ` });
  await sleep(1500);
  await page.evaluate(() => window.dispatchEvent(new Event("resize")));
  await sleep(1500);
  const canvas = await page.$(".panel-robot-scene canvas");
  const box = canvas ? await canvas.boundingBox() : null;
  if (box) {
    await page.mouse.move(box.x + box.width * 0.5, box.y + box.height * 0.5);
    for (let i = 0; i < Number(process.env.ZOOM_TICKS || 0); i++) { await page.mouse.wheel({ deltaY: -300 }); await sleep(120); }
  }
  await sleep(1000);
}

// %% cram_viz: play the scene's recorded demo motion (the recorded coraplex and giskardpy trajectory)
async function play(out, seconds) {
  const b = await browser();
  const page = await b.newPage();
  await page.goto(`${CRAM_VIZ}/?scene=${SCENE}`, { waitUntil: "networkidle2", timeout: 120000 });
  await page.waitForSelector("#play-btn", { visible: true, timeout: 120000 });
  await sleep(6000);
  await sceneOnly(page);
  const recorder = await page.screencast({ path: out });
  await page.click("#play-btn");
  await sleep(seconds * 1000);
  await recorder.stop();
  await b.close();
}

// %% llm_reasoner_viz: replay a finished run and record the viewer following it
async function replay(run, out, seconds) {
  const b = await browser();
  const page = await b.newPage();
  await page.goto(`${VIEWER}/?run=${run}`, { waitUntil: "networkidle2", timeout: 120000 });
  await sleep(3000);
  await page.click("#replay-toggle");
  await page.waitForSelector("#replay-bar button.play", { visible: true, timeout: 30000 });
  await page.evaluate(() => {
    const speed = document.querySelector("#replay-bar select");
    if (speed && [...speed.options].some((o) => o.value === "4")) { speed.value = "4"; speed.dispatchEvent(new Event("change")); }
  });
  await sleep(1000);
  const recorder = await page.screencast({ path: out });
  await page.click("#replay-bar button.play");
  await sleep(seconds * 1000);
  await recorder.stop();
  await b.close();
}

// %% screenshots of both viewers on one run
async function shots(run, directory) {
  fs.mkdirSync(directory, { recursive: true });
  const b = await browser();
  const page = await b.newPage();
  await page.goto(`${CRAM_VIZ}/?scene=${SCENE}&run=${run}`, { waitUntil: "networkidle2", timeout: 120000 });
  await page.waitForFunction(() => { const e = document.querySelector("#designator-execute"); return e && !e.disabled; }, { timeout: 120000 });
  await sleep(4000);
  await page.screenshot({ path: path.join(directory, "cram_viz_page.png") });
  const answer = await page.$("#answer");
  if (answer) {
    await page.evaluate(() => document.querySelector("#answer").scrollIntoView({ block: "start" }));
    await sleep(500);
    await answer.screenshot({ path: path.join(directory, "cram_viz_answer.png") });
  }
  for (const tab of ["Knowledge", "Hypothesis Graph", "Ontology Graph", "Designator", "Plan"]) {
    const clicked = await page.evaluate((name) => {
      const button = [...document.querySelectorAll("button, .tab, [role=tab]")].find((e) => e.textContent.trim() === name);
      if (button) button.click();
      return !!button;
    }, tab);
    if (!clicked) continue;
    await sleep(2500);
    await page.screenshot({ path: path.join(directory, `cram_viz_${tab.toLowerCase().replace(/ /g, "_")}.png`) });
  }
  await page.goto(`${VIEWER}/?run=${run}`, { waitUntil: "networkidle2", timeout: 120000 });
  await sleep(4000);
  await page.screenshot({ path: path.join(directory, "viewer_run.png") });
  await b.close();
}

(async () => {
  const [mode, run, out, seconds] = process.argv.slice(2);
  if (mode === "resolve") await resolve(run, out, seconds);
  else if (mode === "execute") await execute(run, out, Number(seconds || 25));
  else if (mode === "replay") await replay(run, out, Number(seconds || 20));
  else if (mode === "play") await play(run, Number(out || 60));
  else if (mode === "shots") await shots(run, out);
  else throw new Error("mode: execute | replay | shots");
  console.log("done", mode, out);
})();
