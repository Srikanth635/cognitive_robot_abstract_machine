// The five-slide deck on the research idea: the problem, the idea as one picture, the
// knowledge in several perspectives and the levels of questions it answers, generative
// understanding in situations nobody prepared, and physics behind the knowledge. Less
// prose, more pictures; the implementation stays in the full deck.

const sections = [{ id: "idea", title: "The research idea" }];

const slides = [
  {
    id: "idea-problem", section: "idea", kind: "content",
    title: "An Instruction Names the Goal; Execution Needs Every Value",
    subtitle: "The research problem, and why the knowledge CRAM holds today is not enough",
    points: [
      "“Pour the milk into the bowl” leaves open which arm, which tilt, from which side, how far beside the goal, how high; eight slots of coraplex's PouringAction.",
      "A default binds a slot too, silently: with coraplex's defaults the pitcher's mouth ends 0.188 m beside the bowl's opening.",
      "A backend written by hand answers only what somebody thought to write; a model learned from data answers only inside the data it saw; neither says why.",
      "Objective: turn any underspecified instruction into an executable action designator, every choice explained and checked.",
    ],
    visual: "instruction, the open slots as red pills, the robot executing; two cards for the conventional backends; the objective band",
    sources: ["knowledge_perspectives.md §1", "knowledge_perspectives.md §2", "pour_prediction.md §1"],
    notes: "The problem in one picture. An instruction names a goal; a motion needs values the instruction never states, eight of them for a pour. CRAM's resolution backends supply them from what they already hold: code someone wrote for the action, or a model learned from recorded data. Both are silent outside what was written or recorded, and both give a value without a reason. The objective is an executable designator for any instruction, however underspecified, with every choice explained and checked. Sources: knowledge_perspectives.md sections 1 and 2; pour_prediction.md section 1.",
  },
  {
    id: "idea-ring", section: "idea", kind: "content",
    title: "The Model Hypothesises; Ontology, World and Physics Decide",
    subtitle: "The language model supplies the general understanding of an action; the ontology, the live world and physics decide what stands",
    points: [
      "Centre: the executable action, every slot bound, every choice explained.",
      "Four sources of truth around it: the generative model hypothesises what the action normally involves; the ontology keeps the hypothesis consistent and queryable; the live world supplies the facts through krrood; physics measures the numbers.",
      "Outer ring: the stereotypical knowledge in several perspectives, FrameNet roles, Minsky frames and scripts, image schemas, Flanagan's phases, failures and corrections, each its own module in CRAM-AICOR terms.",
    ],
    visual: "a ring diagram in the style of the KnowRob figure: centre circle, four quadrants, an outer ring of five perspectives; a legend beside it",
    sources: ["architecture.md §1", "knowledge_perspectives.md §1", "competency_questions.md §1"],
    notes: "The idea as one picture. At the centre is what the robot needs: an executable action with every slot bound and every choice explained. Around it are the four things that produce and discipline it. A generative model supplies the general understanding of the action, always as a hypothesis. An ontology keeps that hypothesis consistent and queryable, in CRAM-AICOR terms, checked by a reasoner. The live world supplies the facts, read through krrood. And physics measures the numbers that decide a parameter, confirming or refuting what the model hypothesised. The outer ring is the knowledge itself, described from several perspectives, each a small ontology module of its own. Sources: architecture.md section 1; knowledge_perspectives.md section 1; competency_questions.md section 1.",
  },
  {
    id: "idea-knowledge", section: "idea", kind: "content",
    title: "Knowledge in Several Perspectives, Questions at Every Level",
    subtitle: "What each tradition says about pouring, and the competence levels the knowledge is to answer, after Bloom",
    points: [
      "Action frame (FrameNet, Minsky): roles, typical fillers, defaults, preconditions, what the action brings about. Script (Schank and Abelson): the expected sequence. Image schemas (Johnson, Lakoff, Talmy): path, containment, the source tips, the goal holds. Predictive control (Johansson and Flanagan): phases, boundary events, predictions and corrections. Failures: what goes wrong and what corrects it.",
      "Levels: Remember (what the action and its parts are), Understand (phases and their order), Apply (the onset tilt of this pitcher, half full), Analyze (which force-dynamic events, what opposes what), Evaluate (which constraints keep this pour safe), Create (an executable designator).",
      "Every answer carries its evidence grade: Measured, Derived, Sourced, Defined or Generated.",
    ],
    visual: "left, the five perspectives as a compact stack; right, the six levels as a staircase coloured by the grade that can back them, with the grade legend",
    sources: ["knowledge_perspectives.md §1", "competency_questions.md §2", "competency_questions.md §3"],
    notes: "The knowledge is stereotypical: what pouring normally involves, from several traditions that each see something the others do not. FrameNet and Minsky give the roles, defaults and preconditions; scripts the sequence; image schemas the spatial and force structure; Johansson and Flanagan the phases, the events that end them, and the corrections; a failures module what goes wrong. That knowledge is to answer questions at every level of competence, ordered after Bloom: from what the action is, through its phases, to the onset tilt of this pitcher, the forces inside the action, the constraints that keep it safe, and finally an executable designator. Every answer says what it rests on, with one of five evidence grades. Sources: knowledge_perspectives.md section 1; competency_questions.md sections 2 and 3.",
  },
  {
    id: "idea-unknown", section: "idea", kind: "content",
    title: "Generative Understanding for Situations Nobody Prepared",
    subtitle: "A backend bound to its data has nothing to say; a model with its perspectives reasons from what the world states",
    points: [
      "The situation: the only bowl stands upside down, a milk carton and a cereal box beside it; the instruction leaves the goal to choose.",
      "The default arm poured into the overturned bowl 2 of 2, reasoning only about the lateral error.",
      "With the opening stated, 5 of 6 runs with a perspective left the bowl; one refused, saying the carton has no stated opening either, which is arguably the right answer.",
      "Honest: two runs per arm on one scene is a first sign, not a rate; the repetition did not reproduce it; which objects have an opening now comes from the ontology, which way it faces from the mesh.",
    ],
    visual: "three columns: the situation, the data-bound backend's answer, the generative answer with its grounds; a band with the honest caveat",
    sources: ["knowledge_perspectives.md §13.7"],
    notes: "Why a generative model at all? Because the situations a robot meets were not all recorded or written down. The trap case: the only bowl stands upside down, with a milk carton and a cereal box beside it. A backend bound to its data has no case like this, and the default resolver poured into the overturned bowl, both times, reasoning only about the lateral error. With the world stating that the bowl's opening faces down, five of six runs with a perspective left the bowl, and one refused, saying the carton has no stated opening either, which is arguably the right answer. Two runs per arm on one scene is a first sign, not a rate, and the repetition did not reproduce it; what stayed is structural: which objects have an opening now comes from the ontology, and which way it faces is measured from the mesh. Source: knowledge_perspectives.md section 13.7.",
  },
  {
    id: "idea-physics", section: "idea", kind: "content",
    title: "Physics Behind the Knowledge: Preconditions, Phases, Effects",
    subtitle: "The model hypothesises the structure of the action; a simulation confirms or refutes every part it reaches",
    points: [
      "Preconditions: the lip over the goal's opening, the source held and open, the goal upright and open; stated by the perspectives, checked against the world.",
      "Phases and their boundary events: tilt until the flow onset, regulate, terminate, return upright; the onset measured in Isaac Sim at 55.2° for the pitcher at fill 0.5, 3.0° after the mesh's geometry.",
      "Effects: the substance accumulates in the goal; with the lip 0.188 m beside the opening the pour misses, which the prediction now sees before the answer.",
      "A measured onset of 55.2° supports a generated range of 30–110° and refutes one of 70–110°; what no physics result reaches stays Generated, and says so.",
    ],
    visual: "left, the Isaac Sim clip of the pitcher pouring; right, a causal chain of preconditions, phases and effects with grade tags; the verdict rule as a band",
    sources: ["competency_questions.md §1", "competency_questions.md §3", "competency_questions.md §4", "status.md §2", "pour_prediction.md §1"],
    notes: "The last piece is physics. A language model can say what pouring involves, but not at which tilt this pitcher starts to pour, and two generations of the same situation gave different ranges. So the quantities that decide a parameter come from geometry and simulation: the clip shows the pitcher at half fill tilting in Isaac Sim until the particle water leaves it, at 55.2 degrees, three degrees after the mesh's own geometry says. The same simulations are to observe the events the knowledge predicts, contacts made and broken, flow begun and ended, and so confirm or refute the hypothesised phases, preconditions and effects. A measured onset of 55.2 degrees supports a generated range of 30 to 110 and refutes one of 70 to 110; what no physics result reaches stays Generated and says so. Sources: competency_questions.md sections 1, 3 and 4; status.md section 2; pour_prediction.md section 1.",
  },
];

module.exports = { name: "llm_reasoner_idea", title: "llm_reasoner: the research idea", sections, slides, bodiesModule: "./bodies_idea.js" };
