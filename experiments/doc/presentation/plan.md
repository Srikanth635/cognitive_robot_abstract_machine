# `llm_reasoner` presentation: design and content plan

A phased, incremental plan for a professional slide deck that presents `llm_reasoner`
as part of the CRAM story. Everything in it is taken from the package's own record
([`architecture.md`](../architecture.md), [`status.md`](../status.md),
[`knowledge_perspectives.md`](../knowledge_perspectives.md),
[`competency_questions.md`](../competency_questions.md),
[`pour_prediction.md`](../pour_prediction.md)) and from the visual identity of the other
CRAM packages. Nothing on a slide may say more than those documents do; every number
on a slide is traced to the section it comes from.

Written 2026-10-26, on `feature/physics_backed_knowledge`. The deck is built in the
phases below, each reviewed before the next begins.

## 0. Decisions to take first

| # | Question | Recommendation | Why |
|---|---|---|---|
| Q1 | Delivery format | **PowerPoint (`.pptx`) as the master**, with a PDF export; optionally a web copy as a claude.ai Slides artifact for sharing a link | `.pptx` embeds real video (the simulation clips play inside the slide), is what the institute presents with, and stays editable by you. The web Slides type takes images, SVG and GIF but no video, so a web copy would use GIF loops |
| Q2 | Audience and length | One deck of about 24 slides for a 20–25 minute talk to a robotics audience that knows CRAM but not `llm_reasoner`; a 6-slide short version is cut from it in Phase 6 | The "normal viewer" is served by the video slides and the one-sentence slide titles; the detail lives in speaker notes and appendix slides |
| Q3 | Theme | **The institute's deck** (see §1): white slides, navy titles, coloured cards, Calibri | Decided by the user on 2026-10-26 after seeing the dark cram_viz theme; the talk then sits beside the group's other talks |
| Q4 | Fresh footage | Record **one** new resolution run for the video (about $0.10–0.50 of Sonnet 5.5, a few minutes), reuse the 26 recorded runs in `~/.cram_viz/runs` for screenshots | A new run gives a clean, complete log for the viewer and a NEEM to show; the stored runs supply the measured numbers |
| Q5 | Logos | The University of Bremen and AICOR marks, taken from the reference deck into `assets/logos/`; CRAM's ring diagram (`doc/img/cronut.svg`) for slide 3; the krrood and coraplex marks where a package is named | The footer then matches the institute's deck exactly |

Decided 2026-10-26: the recommendations stand on every question (Q1 `.pptx` master
with embedded video and a PDF export; Q2 about 24 slides for 20–25 minutes plus a short
version; Q3 the look of the institute's *Action-Centric Knowledge Representation* deck (changed from the dark cram_viz theme after the first samples); Q4 one fresh resolution run for the pour clip; Q5
the University of Bremen and AICOR marks, found in that deck).

## 1. Design system: the institute's deck

Decided 2026-10-26: the look follows the institute's own deck
*Action-Centric Knowledge Representation* (`Action_Centric_KR.pptx`, August 2026), not
cram_viz's dark theme, so this talk sits beside the group's other talks. Its values are
measured from that deck's slides and kept in `theme.json`, which every build reads.

### 1.1 Palette

| Token | Hex | Used for |
|---|---|---|
| Background | `FFFFFF` | every slide |
| Ink, muted | `2B3440`, `6B7787` | body text; subtitles and captions |
| Navy | `012060` | titles, the resolver, the one-sentence band |
| Blue | `2E5FA3` | the knowledge component, Derived |
| Teal | `0E7C86` | the world injector, success, Measured, "in use" |
| Gold | `A9741F` | the simulation, warnings, Generated, "in progress" |
| Red | `DA0831` | failures, refuted, what is left open |
| Purple | `6B4FA8` | the language model, Sourced |
| Tints | `E9F0FA` blue, `E3F1F2` teal, `FAF3E5` gold, `FCEBEF` red, `F1EDF9` purple, `F5F7FA` grey | card fills |
| Pale blue | `C2DAE4` | the bar beside a title and the title slide's band |

Component colours (resolver navy, language model purple, world injector teal,
knowledge blue, simulation gold) and evidence grades (Measured teal, Derived blue,
Sourced purple, Defined grey, Generated gold) each keep one colour throughout the deck
and appear with the same legend wherever they are used.

### 1.2 Type and layout

- Calibri Light for titles and card headings, Calibri for text, Courier New for code,
  queries and IRIs: the reference deck's faces, present on every Office installation.
- Title 27 pt bold navy at the top left with the pale bar beside it, a 13.5 pt grey
  subtitle below; the title slide centres a 36 pt title and a 26 pt second line on the
  pale band.
- Content in cards: white or tinted fill, a 1.5 pt border in the card's colour, a numbered
  or icon circle, a bold heading; pills for small labels and statuses; a navy band with
  white text for the one sentence a slide must leave behind.
- Footer: the University of Bremen mark bottom left and the AICOR mark bottom right (both
  taken from the reference deck into `assets/logos/`); the source of a slide's numbers in
  italics at the bottom centre; no slide numbers, as in the reference.
- Body text never below 10.5 pt; one idea per slide; a diagram, a table, a big number or a
  video, never a wall of bullets.

## 2. Story and outline

The arc: *CRAM resolves underspecified actions; the resolution methods it has need
data per action and world; `llm_reasoner` lets a generative model supply the general
understanding, keeps it honest with an ontology, the live world and physics, and shows
its reasons.* Pouring is the running example on every slide.

| # | Slide (title as a statement) | Content | Visual | Source |
|---|---|---|---|---|
| 1 | `llm_reasoner`: generative knowledge for CRAM | Title, second line, one-sentence tagline, author, institute, date | The pale title band; the institute's marks | — |
| 2 | A plan says *pour the milk into the bowl*; the robot needs eight values | What a `PouringAction` designator leaves open (source, target, arm, tilt, side, offsets, height) | The designator with its open slots highlighted | `knowledge_perspectives.md` §2 |
| 3 | CRAM already resolves parameters, with knowledge it has to own first | The Knowledge Executive in the CRAM architecture (Beetz and Kümpel, KI 2026, Fig. 2); where today's resolution backends get their knowledge: probabilistic models learned from NEEMs and selective, hand-written backends, both bound to the actions, objects and worlds they were built from, and silent about why | The CRAM "cronut" with the Knowledge Executive lit | `doc/index.rst`, `knowledge_perspectives.md` §1 |
| 4 | Section: *The idea* | — | divider | — |
| 5 | A language model hypothesises; the ontology, the world and physics keep it honest | The four principles as one sentence each: generic behaviour from types and docstrings, structured prompts, the ontology as the single source of truth, the world read through krrood only | Four icons on one line | `architecture.md` §1 |
| 6 | Three kinds of truth: fact, stereotype, decision | World injector knows what *is*; knowledge component what an action *normally* involves; resolver what to do *here*, with reasons | Three columns in the component colours | `knowledge_perspectives.md` §1 |
| 7 | Section: *Architecture* | — | divider | — |
| 8 | Four components, two contracts, one base layer | Resolver, world injector, knowledge component, clients; `WorldAccess` 1.0 and `KnowledgeAccess` 1.3; sits on krrood, imports no world model | Architecture diagram redrawn from `architecture.md` §2 in the component colours | `architecture.md` §2, §3.3, §3.7 |
| 9 | One resolution, step by step | Query in → situation pass → knowledge served or generated → world and knowledge tools → answer with grounds → simulated trial → repair, up to three attempts | Sequence diagram, 7 steps; the same colours as the viewer's timeline | `architecture.md` §3.1–3.7 |
| 10 | Everything the model reads is structured | Tagged sections, JSON data, named rules; placeholders `<Kind>#<n>` instead of a world's names; verdicts (`unsatisfiable`, no fitting candidate, ties) | A prompt skeleton, annotated | `architecture.md` §3.2 |
| 11 | The world is read, never guessed | Eight world tools; 14 predicates on the pouring scene (`contact`, `is_supported_by`, who holds what, what a camera sees); checked conditions, handles | Table of tools, 2 columns | `status.md` §2 World injector |
| 12 | Section: *The knowledge base* | — | divider | — |
| 13 | The ontology is the single source of truth, in two forms | DUL → CRAM-AICOR → the perspective modules → a generated hypothesis; working form: Python objects queried by EQL (closed world); reasoning form: RDF checked by HermiT (open world); both generated from the ontology, never hand-written | Layer diagram with the two forms side by side | `architecture.md` §1, §2.3; `status.md` §4 (2026-10-12) |
| 14 | Generated knowledge is admitted only when the ontology accepts it | The generation loop: structured prompt → instances of generated classes → acceptance checks → HermiT consistency → located contradictions back to the model as a repair prompt → stored once per situation and served from then on | Loop diagram with the repair edge | `README.md`, `architecture.md` §2.3 |
| 15 | One action, five perspectives | Action frame (FrameNet, Minsky), image schemas (Johnson, Lakoff, Talmy), script (Schank and Abelson), predictive control (Johansson and Flanagan), failures and corrections; what each says about pouring | Five cards | `knowledge_perspectives.md` §1, §5 |
| 16 | Flanagan's phases: events that end a phase and the forces they change | A pouring phase structure from a stored hypothesis: phases in order, the mechanical event at each boundary (contact made, lift-off, flow onset, flow end), the forces each event changes, prediction against observation | Phase strip with force-dynamic events as markers | `flanagan_scientific_contract.md`; the stored `pour_milk_sample` hypothesis |
| 17 | Section: *Backing the knowledge* | — | divider | — |
| 18 | Every answer says what it rests on | Evidence grades Measured, Derived, Sourced, Defined, Generated; evidence confirms or refutes a hypothesis (a measured onset of 55.2° supports 30–110°, refutes 70–110°) | Grade legend and one worked example | `competency_questions.md` §3 |
| 19 | Physics decides the numbers: the flow-onset tilt | Isaac Sim PBD water against the mesh-derived onset; six cases, simulation 0.8–3.8° after geometry; against generated ranges of 0–120° and 40–120°, both narrow and agree; 21 generations of the same request gave 7 different ranges | Table of the six cases; **video clip A** | `status.md` §2 Physical measurements |
| 20 | Predicting where the pour ends turns two tries into one | Defaults miss by 0.188 m; with the prediction the first pour lands 3 of 3 against 0 of 3, cost per run $0.046–0.091 against $0.083–0.133; every prediction within 5 mm of the simulated pour | Big-number trio and a top-down plan of the landings (as the viewer draws it); **video clip B** | `pour_prediction.md` §4 |
| 21 | Competency questions, from *what is pouring* to *is this pour safe* | Bloom levels Remember → Create, what answers each, what can back it | Staircase, graded by colour | `competency_questions.md` §2 |
| 22 | Section: *Does it work* | — | divider | — |
| 23 | 28 of 28 cases on a PR2 kitchen of 229 bodies, $0.98 | The eight levels of the live suite (given values → whole world → conditions → everything open → conflicts → tasks); knowledge served from the store in 9 s for $0.04 | Table of levels; three big numbers | `status.md` §3 |
| 24 | Perspectives change the grounds, and, when the world states enough, the choice | Phase 4 and the upside-down bowl: the default arm poured into the overturned bowl 2 of 2; with the opening stated, 5 of 6 perspective runs left it; honest note that two runs per arm is a first sign, not a rate | Two small tables | `knowledge_perspectives.md` §13.4, §13.7 |
| 25 | You can watch it think | cram_viz: scene, answer with grounds, hypothesis and ontology graphs, Execute; `llm_reasoner_viz`: system map, timeline, attempts, prompts in full, NEEMs of every pour tried | Two screenshots with callouts; **video clip C** | cram_viz and viewer READMEs |
| 26 | Conventional resolution and `llm_reasoner`, side by side | Needs training data per action and world vs. pretrained understanding plus the live world; a number vs. a number with its grounds; silent on consistency vs. checked by HermiT; no new objects vs. any object the ontology and mesh describe; and what the conventional methods keep: cheaper, deterministic, and experience of what worked before (NEEMs), which `llm_reasoner` does not replace | Two-column comparison | `knowledge_perspectives.md` §2 (the limit paragraph), §13 |
| 27 | What it does not do yet | Stereotypes give no numbers like −0.188 m; the arm is an arbitrary choice without hold information; generated ranges vary between runs; a cut-off at 8192 output tokens failed 2 of 5 runs; geometric onset covers liquids only | Plain list, five lines | `status.md` §3, §7; `knowledge_perspectives.md` §13.6 |
| 28 | Next: physics-backed knowledge for every quantity, and more actions | Events observed in simulated episodes graded Measured; navigating, picking up and placing after pouring; the competency-question catalogue CQ-001–CQ-065 | Roadmap in three steps | `competency_questions.md` §7; `knowledge_perspectives.md` D8 |
| 29 | Summary | Three sentences; the link to the repository branch | — | — |
| A1–A4 | Appendix | The world-access and knowledge-access contracts; the prompt protocol in full; the K1–K6 knowledge cases; the Flanagan evidence statuses | tables | `architecture.md` §3, `status.md` §3 |

Four section dividers plus 25 content slides is a 25-minute talk at one minute a slide;
the videos carry the rest.

## 3. Phases

Each phase ends with something you can look at, and nothing in a later phase begins
before the earlier one is accepted. Checkpoints are marked ▶.

### Phase 1: Foundations (this document, then the design file)

1. Agree the decisions of §0.
2. Write the design tokens as one file the build reads (`presentation/theme.json`:
   colours, fonts, sizes, component colours, grade colours), so every slide, diagram and
   video overlay draws from one place and a theme change is one edit.
3. Collect the logos into `presentation/assets/logos/` (converted to PNG where the format
   needs it) and note which ones you still have to supply.
▶ You see the palette and type on one sample slide (title, one content slide) and approve
the look before any content is written.

Done 2026-10-26: `theme.json` (the values of §1), the marks under `assets/logos/`, and
`builder/samples.js`, which writes `out/samples.pptx` with three slides: the title, the
architecture (slide 8) and the big-number slide (slide 20). A first set in cram_viz's
dark theme was rejected; the set on the institute's look is awaiting approval. Tooling:
`pptxgenjs` in a node project and the pptx QA scripts' Python packages in a virtual
environment, both outside the repository; the deck's faces are Office's own, so no font
needs installing.

### Phase 2: Content skeleton

1. The deck with every slide of §2 as title plus speaker notes only: the notes carry the
   full text and the source reference; no visuals yet.
2. A check that every number in the notes is in the cited document section.
▶ You read the deck as a script, reorder or cut slides, and the outline is frozen.

Done 2026-10-26: `builder/outline.js` holds every slide as data (title, subtitle, what it
says, the visual planned, the sources it rests on, the speaker notes); `builder/deck.js`
renders it on the layouts of `builder/layouts.js` as `out/llm_reasoner.pptx`, 33 slides:
the 29 of §2 and four appendix slides. A slide without a finished body shows a
skeleton: what it says, the planned visual and its sources. The architecture and the
pour-prediction bodies approved in Phase 1 are already in (`builder/bodies.js`; a later
phase adds one function per slide there). `builder/check_numbers.py` finds every number
of a slide in the files it cites: 33 slides checked, none missing. `builder/script.js`
writes the whole talk as `out/script.md` to read. Awaiting the outline review.

### Phase 3: Diagrams and templates

Drawn as SVG in the theme (one source file per diagram under `presentation/diagrams/`),
rendered to PNG at 2× for the slides, so they can be redrawn when the architecture moves:

1. The CRAM architecture with the Knowledge Executive lit (slide 3), adapted from the cronut.
2. The four-component architecture (slide 8), the same shape as `architecture.md` §2's mermaid.
3. The resolution sequence (slide 9), in the viewer's timeline colours.
4. The ontology layers and the two forms (slide 13).
5. The generation and repair loop (slide 14).
6. The Flanagan phase strip with force-dynamic events (slide 16), drawn from the stored
   sample hypothesis (`cram_viz/.../sample_hypotheses/pour_milk_sample`), not invented.
7. The evidence staircase (slide 21) and the landing plan (slide 20), the latter drawn
   from a run's recorded pour outcomes.
8. The slide templates of §1.2 as a `.pptx` layout master.
▶ Diagrams reviewed as a contact sheet, then placed.

Done 2026-10-26, with two changes to the list above. The diagrams are drawn as native
PowerPoint shapes from `theme.json` (`builder/bodies.js`, `builder/architecture_body.js`),
not as SVG files rendered to PNG: they are as redrawable, and they stay editable in
PowerPoint in the deck's own fonts; the `diagrams/` folder is therefore not used. Slide 3
shows CRAM's packages as layers (plans, the krrood layer with its resolution backends,
the world, motion, the robot) rather than the "cronut", which is KnowRob's ring figure;
if the KI 2026 paper's Fig. 2 is supplied, it can take that slide. Drawn and reviewed:
slides 3, 8, 9, 13, 14, 16 (the phase strip from the stored sample hypothesis's 15
phases, boundary events and the forces they start or end), 18 (the grade legend and the
measured onset against two generated ranges), 20 (with the landings of run 77fc9006 as
the viewer draws them) and 21. The templates of §1.2 are the layouts TITLE, DIVIDER and
CONTENT and the body helpers `table`, `steps`, `columns`, `mediaFrame` and
`gradeLegend` in `builder/layouts.js`; Phase 5 builds the remaining slides from them.

### Phase 4: Footage and screenshots

All three clips are 8–15 seconds, 1920×1080, H.264 `.mp4` for the deck and a GIF/WebP
loop for a web copy. Everything needed is on this machine (Isaac Sim 6.0 at `~/isaacsim`,
the RTX 4070, ffmpeg, Chrome, MuJoCo in the project environment) and the repository
already has the recorders (`semantic_digital_twin.adapters.isaac.video_recording`,
`semantic_digital_twin.adapters.mujoco_video_recording`). Nothing in those packages is
changed; the recording scripts live under `presentation/footage/`.

| Clip | Shows | How | Used on |
|---|---|---|---|
| A | The pitcher tilting until the simulated water leaves it, with the onset angle appearing when it does | `experiments.flow_onset_measurement`'s stage under `IsaacVideoRecorder`, one camera, the pitcher at fill 0.5; angle overlay burnt in with ffmpeg | slide 19 |
| B | The PR2 pouring into the bowl: first the default pour missing by 0.188 m, then the predicted, corrected pour landing | cram_viz Execute of a recorded run under the live bridge, captured from Chrome with ffmpeg; two takes cut side by side | slide 20, and the title still |
| C | The viewer following a run live: components flashing, the timeline growing, an attempt rejected, the answer with grounds | Screen capture of `llm_reasoner_viz` replaying the run (`events.jsonl` replay) | slide 25 |
| D (optional) | A physical pour under MuJoCo (`pr2_holding_a_cup`) if Clip B's kinematic motion looks too clean | `experiments.physics_viewer` with `MujocoVideoRecorder` | slide 2 or 20 |

Screenshots: cram_viz's answer panel with grounds, the hypothesis and ontology graph
tabs, the viewer's attempts strip and prompt inspector, one NEEM plan. Taken from one
run so names agree across slides.

Clip B needs one fresh resolution run (Q4); clips A, C and the screenshots need no model
call. The cost of the run is reported when it is made.
▶ Clips reviewed in a scratch player before they go into the deck.

Done 2026-10-26 (the scripts are under `footage/`; the clips and screenshots under
`media/`, git-ignored):

- Clip A, `footage/flow_onset_clip.py`, run with `isaac-cram-env` and
  `OMNI_KIT_ACCEPT_EULA=YES`, the worktree's `experiments`, `llm_reasoner`, `cram_viz` and
  `physics_simulators` first on `PYTHONPATH`: the harness's own stage, filling and tilt,
  with a camera on the mesh's bounds, the water drawn as spheres and frames taken through
  the replicator's step. The pitcher at fill 0.5 started to pour at 55.3° in the clip
  (the stored measurement says 55.2°); 203 frames, encoded at 0.6 times speed to 10.8 s.
  Two lessons: a virtual-memory cap (`ulimit -v`) makes the GPU runtime fail with
  *CUDA out of memory*, and the renderer gives no frame without a replicator step.
- Clip C, `footage/capture.js replay`: the viewer replaying run 77fc9006 at four times
  speed, from Chrome on the display through puppeteer-core, trimmed to 14 s.
- Screenshots, `footage/capture.js shots`: cram_viz's Designator tab with the resolved
  designator, its why, its grounds and the hypothesis; the viewer with the timeline, the
  attempts and where the pours ended. The answer panel itself cannot be shot after a
  reload, because the scene's first step replaces it.
- Clip B, `footage/capture.js execute`: cram_viz served from the integration branch at
  `2f84959fcf` (the ORM interfaces generated in that checkout, the scene bundles taken
  from this one), Execute of run 77fc9006 (the corrected pour landing) and of run
  acca6ab3 (the accepted pour missing), the 3D scene enlarged to the page for the
  recording and the robot's region cropped at encode time: `media/clip_b_land.mp4`
  (14 s) and `media/clip_b_miss.mp4` (12.5 s), both at 1.6 times speed. The missed pour
  is for slide 2, the landing for slide 20.

What the recording needed, for next time:

- The viewers run from `/home/malineni/envs/new-cram-env/bin/python` with
  `/opt/ros/jazzy/setup.bash` *and* `~/ros_ws/install/setup.bash` sourced (coraplex needs
  `rclpy` and `json_msgs`), the wanted checkout's `*/src` directories first on
  `PYTHONPATH` and `CRAM_VIZ_SCENES` pointing at its scene bundles; `.env` holds the keys.
  Appending `$PYTHONPATH` matters: replacing it drops ROS.
- A checkout other than the main one needs its ORM interfaces:
  `scripts/regenerate_all_orm.py` with that checkout first on `PYTHONPATH`, ROS sourced
  and the project's `.venv/bin` (for `ruff`) on `PATH`.
- The Anthropic account's credit balance was too low on 2026-10-26, so no fresh
  resolution could be made (the attempt is run `eb52c026`, failed after 11 s, no cost).
- The stored runs of 2026-10-21 open and execute only on the code of their day: the
  current branch renamed an enumeration member and the knowledge format
  (`FLOW_ONSET_TILT_RADIANS`, `ConceptWrittenAsIndividual`), and the afternoon's adaptation
  to the upstream main changed the arm slot from an enumeration to a robot part. The
  integration branch at `2f84959fcf` (2026-10-21 12:32) is the version the landing run
  77fc9006 was recorded with.

### Phase 5: Build the deck

Incrementally, in the order the talk is given, three or four slides at a time, each batch
sent to you as the deck grows: title and section 1 → architecture → knowledge base →
backing → results → close → appendix. Speaker notes on every slide. Built with the
project's Python interpreter (`python-pptx`; a new dependency is asked about before it
is added), from the theme file, the diagram PNGs and the clips, so the whole deck can be
rebuilt after a content change.
▶ The complete deck and its PDF export.

Done 2026-10-26: every slide has its finished body (`builder/bodies.js`,
`builder/bodies_content.js`, `builder/bodies_appendix.js`, one function per slide id),
built from `builder/outline.js` and the templates of `builder/layouts.js`; the clips and
screenshots of Phase 4 are embedded from `media/`. The builder is
`NODE_PATH=<node_modules> PPTX_SKILL_SCRIPTS=<pptx skill scripts> node builder/deck.js`;
the PDF comes from LibreOffice (`soffice --headless --convert-to pdf`). The deck
validates (the pptx skill's checker) and every number in it was found in its cited
source (`builder/check_numbers.py`: 33 slides, none missing). Visual review found and
fixed wrapped card titles, which the step and column templates now size themselves.

### Phase 6: Review and variants

1. Content review against the record: every claim traced; the honest limits kept.
2. Visual review: contrast, consistent places for repeated elements, nothing under 20 pt,
   videos play from the slide.
3. A 6-slide short version (slides 1, 5, 9, 19 or 20, 23, 29) for a five-minute slot.
4. Optional: a web copy as a Slides artifact with GIF loops in place of the videos, for
   sharing by link.
▶ Final deck, PDF, short version; this plan updated with what was delivered.

Done 2026-10-26. Content review: every number traced (`builder/check_numbers.py`);
one characterisation the record does not word itself (how the probabilistic backends
resolve) reworded to what can be said plainly, "learned from data". Visual review of
all 33 slides at full size: one overflow (the setup card of slide 19) fixed. The short
version is `node builder/deck.js --short`, six slides (title, principles, the resolution
flow, the flow-onset measurement, the live suite, the summary), `out/llm_reasoner_short.pptx`
and its PDF. Delivered: `out/llm_reasoner.pptx`, `out/llm_reasoner.pdf`,
`out/llm_reasoner_short.pptx`, `out/llm_reasoner_short.pdf`, `out/script.md`; dated
copies of the full deck and PDF in the home directory as asked. The author line names
the author alone, as decided on 2026-10-26.

## 4. Where it lives

```
llm_reasoner/doc/presentation/
  plan.md            this document
  theme.json         design tokens (Phase 1)
  assets/logos/      the marks (Phase 1)
  diagrams/*.svg     diagram sources (Phase 3)
  footage/           recording scripts and the clips' settings (Phase 4)
  builder/             the deck builder (Phase 5)
  llm_reasoner.pptx  the deck, and its PDF (Phase 5)
```

Large binaries (clips, PNG renders, the deck) are kept out of git by a `.gitignore` in
`presentation/`, like cram_viz's scene bundles; the sources that produce them are tracked.

## 5. Risks and how the plan meets them

| Risk | Met by |
|---|---|
| The development moves on while the deck is built | Phase 2 freezes the outline; numbers are re-checked against `status.md` in Phase 6, and the builder re-renders from the sources |
| A number on a slide that the record does not support | Every number carries its source in the caption and the notes; Phase 2 and 6 check them |
| Isaac Sim footage takes long or fails to render headless | Clip A is the only Isaac clip; MuJoCo (clip D) and cram_viz (clip B) are the fallbacks for the pour |
| A mark or colour drifts from the institute's deck | Every value is measured from that deck and kept in `theme.json`; the marks are its own files |
| Overstating against the conventional methods | Slide 26 keeps what they do better (cost, determinism, experience from NEEMs) and slide 27 lists the limits |


## 6. A second deck: the research idea in five slides (2026-10-26)

Asked for after the talk deck: five slides on the idea rather than the implementation,
with pictures rather than prose. `builder/idea_outline.js` holds them and
`builder/bodies_idea.js` draws them; `node builder/deck.js --idea` writes
`out/llm_reasoner_idea.pptx`, and `builder/check_numbers.py idea_outline.js` checks its
numbers. The slides: the research problem and objective (instruction, the open slots,
execution; the two conventional backends); the idea as a ring in the manner of the
KnowRob figure (the executable action at the centre, the four sources of truth around it,
the perspectives as the outer ring); the perspectives beside the competence levels after
Bloom; generative understanding in a situation nobody prepared (the upside-down bowl);
physics behind the knowledge (the Isaac Sim clip beside the chain of preconditions,
phases and effects, with the verdict rule). Same theme, same sources, same checks.