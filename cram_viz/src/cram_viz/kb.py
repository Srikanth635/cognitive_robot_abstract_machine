"""
The recorded demo scene as an EQL (Entity Query Language) knowledge base.

EQL is krrood's pythonic relational query language. This module models the recorded
coraplex/giskardpy episode — bench objects, robot parts, action episodes, per-joint
motion — as plain dataclasses and exposes:

fresh_namespace()  -> dict for evaluating one EQL query (fresh variables)
run_query(code)    -> execute an EQL query string, return JSON-able result
graph_payload()    -> nodes/edges/details/presets for the UI knowledge graph
view_payload(name) -> one of the graph-panel tabs (knowledge / kinematics / plan /
chart)

krrood is imported lazily: without it the static viewer still works, only the EQL panel
is unavailable. Scene bundles are read from paths.scenes_dir().

The knowledge graph also merges in :mod:`cram_viz.action_knowledge` and
:mod:`cram_viz.entity_knowledge`, static, hand-authored ontologies (keyed by action verb
and by BenchObject name respectively) that attach to every matching ActionEpisode or
object via ``has_knowledge`` edges -- see :func:`graph_payload`.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
from collections import Counter, defaultdict
from enum import StrEnum
from dataclasses import dataclass, fields, is_dataclass
from functools import cache

from krrood.entity_query_language import factories as eql_factories
from krrood.entity_query_language.predicate import HasType
from krrood.symbol_graph.symbol_graph import SymbolGraph
from llm_reasoner.ontologies.predictive_manipulation import (
    core,
    gaze,
    pouring,
    prehension,
)
from llm_reasoner.hypotheses.graph import (
    HypothesisGraphNode,
    HypothesisGraphRelation,
)
from llm_reasoner.ontologies.terms import ConceptReference, DescriptionEntity
from llm_reasoner.resolver.reflection import SlotKind
from typing_extensions import Any, Callable, Dict, List, Optional, Tuple

from cram_viz.hypothesis_knowledge.ontology_backing import OntologyBacking
from cram_viz import (
    action_knowledge,
    configure_console_logging,
    designator_knowledge,
    entity_knowledge,
    hypothesis_knowledge,
    paths,
)


def build_symbol_graph() -> None:
    """
    Rebuild krrood's symbol graph over every symbol class imported so far.

    The graph takes its class diagram from the classes known when it is first created,
    which happens while the first package declaring symbols is still being imported;
    the knowledge vocabulary's classes are only usable once the graph knows them. A
    pytest run gets the graph from its fixtures; the viewer's process builds it here,
    after everything is imported.
    """
    SymbolGraph.clear()
    SymbolGraph()


def scene_name(selected_scene: Optional[str] = None) -> Optional[str]:
    """
    The requested scene, ``CRAM_VIZ_SCENE``, or the scenes-index default.
    """
    if selected_scene is not None:
        return selected_scene if re.fullmatch(r"[\w-]+", selected_scene) else None
    environment_override = os.environ.get("CRAM_VIZ_SCENE")
    if environment_override:
        return environment_override
    try:
        index = json.load(open(os.path.join(str(paths.scenes_dir()), "index.json")))
        return index["default"]
    except (OSError, ValueError, KeyError):
        return None


def scene_dir(selected_scene: Optional[str] = None) -> Optional[str]:
    """
    Directory of the active scene bundle, or None without one.
    """
    name = scene_name(selected_scene)
    return os.path.join(str(paths.scenes_dir()), name) if name else None


def load_scene(
    selected_scene: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    The active scene's (scene.json, trajectory.json), or ``({}, {})``.
    """
    directory = scene_dir(selected_scene)
    if not directory:
        return {}, {}
    try:
        scene = json.load(open(os.path.join(directory, "scene.json")))
    except (OSError, ValueError):
        return {}, {}
    try:
        trajectory = json.load(
            open(os.path.join(directory, scene.get("trajectory", "trajectory.json")))
        )
    except (OSError, ValueError):
        trajectory = {}
    return scene, trajectory


def load_urdf(
    selected_scene: Optional[str] = None,
) -> Tuple[List[str], List[Dict[str, str]]]:
    """
    Parse the active scene's robot URDF into (links, joints).

    Used by the kinematic-tree view; a regex parse suffices because the bundled URDFs
    are flat.
    """
    scene, _ = load_scene(selected_scene)
    robot_model = next(
        (model for model in scene.get("models", []) if model.get("robot")), None
    )
    directory = scene_dir(selected_scene)
    if not robot_model or not directory:
        return [], []
    try:
        text = open(
            os.path.join(directory, robot_model["urdf"]),
            encoding="utf-8",
            errors="replace",
        ).read()
    except OSError:
        return [], []
    links = re.findall(r'<link\s+name="([^"]+)"', text)
    joints = []
    for joint in re.finditer(
        r'<joint\s+name="([^"]+)"\s+type="([^"]+)">(.*?)</joint>', text, re.S
    ):
        body = joint.group(3)
        parent = re.search(r'<parent\s+link="([^"]+)"', body)
        child = re.search(r'<child\s+link="([^"]+)"', body)
        if parent and child:
            joints.append(
                {
                    "name": joint.group(1),
                    "type": joint.group(2),
                    "parent": parent.group(1),
                    "child": child.group(1),
                }
            )
    return links, joints


# %% the entity model ---------------------------------------------------------
@dataclass(unsafe_hash=True)
class Position:
    """
    A world position in metres.
    """

    x: float
    """
    World x coordinate.
    """

    y: float
    """
    World y coordinate.
    """

    z: float
    """
    World z coordinate.
    """

    def __repr__(self) -> str:
        return "(%.2f, %.2f, %.2f)" % (self.x, self.y, self.z)


@dataclass(unsafe_hash=True)
class Gripper:
    """
    An end effector of the recorded robot.
    """

    name: str
    """
    Part name from the scene's robot annotation.
    """

    side: str
    """
    Body side the gripper belongs to ('left' / 'right').
    """

    opening_m: float = 0.085
    """
    Maximum opening width in metres (Robotiq 2F-85 default).
    """


@dataclass(unsafe_hash=True)
class Arm:
    """
    A manipulator of the recorded robot.
    """

    name: str
    """
    Part name from the scene's robot annotation.
    """

    side: str
    """
    Body side of the arm ('left' / 'right').
    """

    robot: str
    """
    Name of the robot this arm belongs to.
    """

    gripper: Gripper
    """
    The end effector mounted on this arm.
    """


@dataclass(unsafe_hash=True)
class Robot:
    """
    The robot that executed the recorded episode.
    """

    name: str
    """
    Robot name from the scene bundle.
    """

    arm_count: int
    """
    Number of annotated arms.
    """


@dataclass(unsafe_hash=True)
class BenchObject:
    """
    A loose object (or named location) in the scene.
    """

    name: str
    """
    Object identifier, e.g. ``milk``.
    """

    kind: str
    """
    ``object`` for graspable things, ``location`` for named areas.
    """

    label: str
    """
    Human-readable display name.
    """

    height_m: float
    """
    Approximate object height in metres.
    """

    position: Position
    """
    Spawn position recorded at frame 0 of the episode.
    """


@dataclass(unsafe_hash=True)
class ActionEpisode:
    """
    One executed plan segment of the recording.
    """

    name: str
    """
    Segment step name, e.g. ``transport_milk``.
    """

    index: int
    """
    Position of the episode in execution order.
    """

    start_frame: int
    """
    First trajectory frame of the episode.
    """

    end_frame: int
    """
    Frame after the last trajectory frame of the episode.
    """

    duration_s: float
    """
    Episode duration in seconds.
    """

    performed_by: Optional[Arm]
    """
    The arm that performed the manipulation, if any.
    """

    picks: Optional[BenchObject]
    """
    The object the episode picks up, if any.
    """

    places_at: Optional[BenchObject]
    """
    The location the object is placed at, if any.
    """


@dataclass(unsafe_hash=True)
class JointMotion:
    """
    Per-joint motion statistics over the whole recorded trajectory.
    """

    name: str
    """
    Joint name (without the model prefix).
    """

    arm_side: str
    """
    Body side the joint belongs to ('left' / 'right' / 'body' / …).
    """

    min_rad: float
    """
    Smallest recorded joint position (radians or metres).
    """

    max_rad: float
    """
    Largest recorded joint position (radians or metres).
    """

    range_rad: float
    """Travelled range, ``max_rad - min_rad``."""


# %% the CRAM architecture entities --------------------------------------------
@dataclass(unsafe_hash=True)
class Package:
    """
    A top-level package of the CRAM repository.
    """

    name: str
    """
    Directory name, e.g. ``coraplex``.
    """

    description: str
    """
    One-line description (curated, or the first README line).
    """

    module_count: int
    """
    Number of Python modules in the package.
    """

    class_count: int
    """
    Number of classes defined in the package.
    """


@dataclass(unsafe_hash=True)
class SubPackage:
    """
    A qualified subpackage, e.g. ``coraplex.plans``.
    """

    name: str
    """
    Qualified name, e.g. ``coraplex.plans``.
    """

    package: str
    """
    The top-level package this subpackage belongs to.
    """

    module_count: int
    """
    Number of modules in the subpackage.
    """

    class_count: int
    """
    Number of classes defined in the subpackage.
    """


@dataclass(unsafe_hash=True)
class PythonClass:
    """
    A class found by the static scan of the CRAM repository.
    """

    name: str
    """
    Class name.
    """

    package: str
    """
    Top-level package the class is defined in.
    """

    subpackage: str
    """
    Qualified subpackage (equal to ``package`` for top-level modules).
    """

    module: str
    """
    Repository-relative module path.
    """

    bases: tuple
    """
    Names of the direct base classes.
    """

    methods: int
    """
    Number of methods defined on the class.
    """

    doc: str
    """
    First docstring line, or ``''``.
    """


# %% scan the CRAM architecture --------------------------------------------------
def _cram_root() -> str:
    """
    The CRAM repository the architecture graph is scanned from.
    """
    return str(paths.architecture_root())


def _architecture_cache() -> str:
    """
    Path of the scan cache — always in the writable data directory, because the scenes
    checkout may be read-only.
    """
    return os.path.join(str(paths.data_dir()), "arch_cache.json")


#: directories never descended into during the architecture scan
SKIP_DIRS = {
    "__pycache__",
    "node_modules",
    "doc",
    "docs",
    "resources",
    "build",
    "dist",
    "plugins",
}

#: curated one-line descriptions for the well-known workspace packages
PKG_DESCRIPTIONS = {
    "krrood": "knowledge representation & reasoning through OO design (home of EQL)",
    "coraplex": "the plan executive: designators, plans, locations",
    "pycram": "legacy plan executive (resources/demos)",
    "giskardpy": "constraint-based motion planning and control",
    "robokudo": "perception framework",
    "semantic_digital_twin": "semantic world model / digital twin",
    "segmind": "segmentation / vision models",
    "probabilistic_model": "probabilistic models and inference",
    "random_events": "sigma-algebra & random events for probabilistic reasoning",
    "physics_simulators": "physics simulator bindings",
    "experiments": "experiment scripts (incl. EQL experiments)",
    "test": "the test suites of all packages",
    "scripts": "maintenance scripts",
    "root": "top-level demo scripts (sterility test, wind turbine…)",
}


def _first_readme_line(directory: str) -> str:
    """
    The first non-empty line of a directory's README, or ``''``.
    """
    for name in ("README.md", "readme.md"):
        readme_path = os.path.join(directory, name)
        if os.path.exists(readme_path):
            for line in open(readme_path, encoding="utf-8", errors="replace"):
                line = line.strip().lstrip("#").strip()
                if line:
                    return line[:120]
    return ""


def scan_architecture() -> (
    Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Tuple[str, str]]]
):
    """
    Statically scan the CRAM repository for its architecture graph.

    :return: (packages, classes, import edges between packages). A pure ``ast`` parse —
        nothing is imported.
    """
    packages: List[Dict[str, Any]] = []
    classes: List[Dict[str, Any]] = []
    imports: Dict[str, set] = {}
    cram_root = _cram_root()
    if not os.path.isdir(cram_root):
        return packages, classes, []

    package_dirs = {"root": cram_root}
    for entry in sorted(os.listdir(cram_root)):
        directory = os.path.join(cram_root, entry)
        if (
            os.path.isdir(directory)
            and not entry.startswith(".")
            and entry not in SKIP_DIRS
            and "egg-info" not in entry
        ):
            package_dirs[entry] = directory
    package_names = set(package_dirs)

    modules_per_package: Dict[str, int] = {}
    for package, base in package_dirs.items():
        module_count = 0
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [
                name
                for name in dirnames
                if not name.startswith(".") and name not in SKIP_DIRS
            ]
            if package == "root":
                dirnames[:] = []  # root package = top-level scripts only
            for filename in filenames:
                if not filename.endswith(".py"):
                    continue
                path = os.path.join(dirpath, filename)
                try:
                    tree = ast.parse(
                        open(path, encoding="utf-8", errors="replace").read()
                    )
                except SyntaxError:
                    continue
                module_count += 1
                module = os.path.relpath(path, cram_root)[:-3].replace(os.sep, ".")
                _collect_classes_and_imports(
                    tree, package, module, package_names, classes, imports
                )
        modules_per_package[package] = module_count

    class_counts = Counter(entry["package"] for entry in classes)
    for package in package_dirs:
        description = PKG_DESCRIPTIONS.get(package) or _first_readme_line(
            package_dirs[package]
        )
        packages.append(
            dict(
                name=package,
                description=description,
                module_count=modules_per_package.get(package, 0),
                class_count=class_counts.get(package, 0),
            )
        )
    dependency_edges = sorted(
        (source, target) for source, targets in imports.items() for target in targets
    )
    return packages, classes, dependency_edges


def _collect_classes_and_imports(
    tree: ast.Module,
    package: str,
    module: str,
    package_names: set,
    classes: List[Dict[str, Any]],
    imports: Dict[str, set],
) -> None:
    """
    Collect class definitions and cross-package imports from one module.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            bases = tuple(
                (
                    base.id
                    if isinstance(base, ast.Name)
                    else (base.attr if isinstance(base, ast.Attribute) else "?")
                )
                for base in node.bases
            )
            doc = (ast.get_docstring(node) or "").strip().split("\n")[0][:140]
            methods = sum(
                1
                for member in node.body
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))
            )
            classes.append(
                dict(
                    name=node.name,
                    package=package,
                    module=module,
                    bases=list(bases),
                    methods=methods,
                    doc=doc,
                )
            )
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                roots = [alias.name.split(".")[0] for alias in node.names]
            elif node.level == 0:
                roots = [(node.module or "").split(".")[0]]
            else:
                roots = []
            for root in roots:
                if root in package_names and root != package:
                    imports.setdefault(package, set()).add(root)


def _load_architecture_cache(cram_root: str, require_classes: bool) -> Optional[tuple]:
    """
    The cached scan if it is usable, else None.

    A cache written for another repository root is not trusted (unless no repository
    exists at all, in which case any cache beats nothing).
    """
    try:
        cached = json.load(open(_architecture_cache()))
    except (OSError, ValueError):
        return None
    if cached.get("version") != 2:
        return None
    if os.path.isdir(cram_root) and cached.get("cram_root") != cram_root:
        return None
    if require_classes and not cached.get("classes"):
        return None
    return (
        cached["packages"],
        cached["classes"],
        [tuple(edge) for edge in cached["deps"]],
    )


def load_architecture() -> (
    Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Tuple[str, str]]]
):
    """
    :func:`scan_architecture` behind a JSON disk cache.

    A full scan takes seconds, so results are cached in the data directory, keyed by the
    scanned root; a cache from another root is rescanned.
    """
    cram_root = _cram_root()
    cached = _load_architecture_cache(cram_root, require_classes=False)
    if cached is not None:
        return cached
    if not os.path.isdir(cram_root):
        return [], [], []
    packages, classes, dependency_edges = scan_architecture()
    if not classes:
        # a checkout exists but yielded nothing (empty or partial clone) — fall
        # back to the cache rather than losing the architecture graph
        return _load_architecture_cache(cram_root, require_classes=True) or (
            packages,
            classes,
            dependency_edges,
        )
    os.makedirs(os.path.dirname(_architecture_cache()), exist_ok=True)
    json.dump(
        {
            "version": 2,
            "cram_root": cram_root,
            "packages": packages,
            "classes": classes,
            "deps": dependency_edges,
        },
        open(_architecture_cache(), "w"),
    )
    return packages, classes, dependency_edges


def _side_of_name(name: str) -> str:
    """
    Body side encoded in a part/link name: 'left', 'right' or ``''``.
    """
    lowered = name.lower()
    if "left" in lowered or lowered.startswith("l_"):
        return "left"
    if "right" in lowered or lowered.startswith("r_"):
        return "right"
    return ""


class KB:
    """
    The recorded episode as EQL-queryable entities.

    Built once from the active scene bundle plus a static scan of the CRAM repository;
    every attribute is a plain list of dataclass instances that EQL variables range
    over.
    """

    def __init__(self, selected_scene: Optional[str] = None) -> None:
        """
        Build every entity list from the active scene bundle and a static scan of the
        CRAM architecture.
        """
        scene, trajectory = load_scene(selected_scene)
        frames_per_second = scene.get("fps", 30)
        parts = (scene.get("robot") or {}).get("parts") or {}
        robot_name = (scene.get("robot") or {}).get("name", "robot")
        robot_prefix = (scene.get("robot") or {}).get("prefix", "")

        self.objects = self._build_objects(scene)
        objects_by_id = {entity.name: entity for entity in self.objects}
        place_area = objects_by_id.get("place_area")

        self.grippers, self.arms = self._build_arms(parts, robot_name)
        self.robot = Robot(robot_name, arm_count=len(self.arms))
        self.episodes = self._build_episodes(
            scene, frames_per_second, objects_by_id, place_area
        )
        self.joints = self._build_joint_motions(trajectory, parts, robot_prefix)

        packages, classes, dependency_edges = load_architecture()
        self.packages = [Package(**entry) for entry in packages]
        self.classes = [
            PythonClass(
                name=entry["name"],
                package=entry["package"],
                subpackage=self._subpackage_of(entry["package"], entry["module"]),
                module=entry["module"],
                bases=tuple(entry["bases"]),
                methods=entry["methods"],
                doc=entry["doc"],
            )
            for entry in classes
        ]
        self.package_deps = dependency_edges
        self.subpackages = self._build_subpackages(self.classes)

        self.action_phase_sequences = list(
            action_knowledge.ACTION_PHASE_KNOWLEDGE.values()
        )
        self.action_phases = [
            phase
            for sequence in self.action_phase_sequences
            for phase in sequence.phases
        ]
        self.concepts = self._build_concepts(self.action_phases)

        self.action_knowledge_entries = list(action_knowledge.ACTION_KNOWLEDGE.values())
        self.entity_knowledge_entries = list(entity_knowledge.ENTITY_KNOWLEDGE.values())
        self.ontology_classes = self._build_ontology_classes(
            self.action_knowledge_entries, self.entity_knowledge_entries
        )
        self.roles = [
            role for entry in self.action_knowledge_entries for role in entry.roles
        ]
        self.parameters = [
            parameter
            for entry in self.action_knowledge_entries
            for parameter in entry.parameters
        ]
        self.preconditions = [
            precondition
            for entry in self.action_knowledge_entries
            for precondition in entry.preconditions
        ]
        self.goals = [
            goal for entry in self.action_knowledge_entries for goal in entry.goals
        ]
        self.properties = [
            property_
            for entry in self.entity_knowledge_entries
            for property_ in entry.properties
        ]
        self.qualities = [
            quality
            for entry in self.entity_knowledge_entries
            for quality in entry.qualities
        ]
        self.affordances = [
            affordance
            for entry in self.entity_knowledge_entries
            for affordance in entry.affordances
        ]
        self.entity_relations = [
            relation
            for entry in self.entity_knowledge_entries
            for relation in entry.relations
        ]

        # empty until an instruction has been generated: the pouring variables range
        # over whatever the model produced, and there is nothing before that
        active_hypothesis = hypothesis_knowledge.active()
        view = active_hypothesis.view if active_hypothesis else None
        structure = view.root if view else None
        # in the order the phases run, so an answer listing phases reads as the
        # sequence of the action
        self.pouring_phases = view.phases() if view else []
        self.pouring_roles = list(structure.defines_role) if structure else []
        self.pouring_event_types = (
            list(structure.defines_event_type) if structure else []
        )
        self.pouring_force_dynamic_events = (
            [dynamics.event for dynamics in view.force_dynamics()] if view else []
        )
        self.pouring_transitions = (
            list(structure.defines_transition) if structure else []
        )
        self.pouring_control_points = (
            list(structure.defines_control_point) if structure else []
        )
        self.pouring_force_couplings = (
            list(structure.defines_force_coupling)
            if isinstance(structure, prehension.PrehensionPhaseStructure)
            else []
        )
        self.pouring_continuous_predictions = (
            list(structure.defines_continuous_prediction) if structure else []
        )
        self.pouring_parameters = list(structure.defines_parameter) if structure else []
        self.pouring_sensory_predictions = (
            list(structure.defines_prediction) if structure else []
        )
        self.pouring_comparisons = (
            list(structure.defines_comparison) if structure else []
        )
        self.pouring_corrections = (
            list(structure.defines_correction) if structure else []
        )
        self.pouring_gaze_fixations = (
            list(structure.defines_gaze_fixation) if structure else []
        )
        condition_requirements_by_identity = {}
        for phase in self.pouring_phases:
            for requirement in phase.has_condition_requirement:
                condition_requirements_by_identity[id(requirement)] = requirement
        self.pouring_condition_requirements = list(
            condition_requirements_by_identity.values()
        )

    @staticmethod
    def _build_concepts(
        phases: List[action_knowledge.ActionPhase],
    ) -> List[action_knowledge.Concept]:
        """
        Every distinct Concept referenced by any phase's ``maintains``/``targets``.
        """
        by_name: Dict[str, action_knowledge.Concept] = {}
        for phase in phases:
            for concept in (*phase.maintains, *phase.targets):
                by_name.setdefault(concept.name, concept)
        return list(by_name.values())

    @staticmethod
    def _build_ontology_classes(
        action_entries: List[action_knowledge.ActionKnowledge],
        entity_entries: List[entity_knowledge.EntityKnowledge],
    ) -> List[action_knowledge.OntologyClass]:
        """
        Every distinct OntologyClass referenced by any action or entity knowledge entry.

        Classes are shared graph nodes (see :func:`action_knowledge.class_node_id`), so
        the same class referenced from two entries is one entity here too.
        """
        by_name: Dict[str, action_knowledge.OntologyClass] = {}
        for entry in (*action_entries, *entity_entries):
            for ontology_class in entry.classes:
                by_name.setdefault(ontology_class.name, ontology_class)
        return list(by_name.values())

    @staticmethod
    def _build_objects(scene: Dict[str, Any]) -> List[BenchObject]:
        """
        Scene objects (spawn poses recorded at frame 0) plus the place area.
        """
        objects = []
        for entry in scene.get("objects") or []:
            objects.append(
                BenchObject(
                    name=entry["id"],
                    kind="object",
                    label=entry["id"].replace("_", " ").title(),
                    height_m=0.1,
                    position=Position(
                        *[round(value, 3) for value in entry["spawn"][:3]]
                    ),
                )
            )
        if scene.get("placeTarget"):
            target = scene["placeTarget"]
            objects.append(
                BenchObject(
                    name="place_area",
                    kind="location",
                    label="Place area",
                    height_m=0.0,
                    position=Position(
                        round(target["pos"][0], 3),
                        round(target["pos"][1], 3),
                        target.get("z", 0),
                    ),
                )
            )
        return objects

    @staticmethod
    def _build_arms(
        parts: Dict[str, Any], robot_name: str
    ) -> Tuple[List[Gripper], List[Arm]]:
        """
        Arms and grippers from the recorded robot annotation parts.

        Gripper keywords take precedence — robot names can contain 'arm' themselves, so
        'arm' alone must not decide.
        """
        gripper_parts = [
            part
            for part in parts
            if any(
                keyword in part.lower() for keyword in ("gripper", "hand", "effector")
            )
        ]
        arm_parts = [
            part
            for part in parts
            if part not in gripper_parts and "arm" in part.lower()
        ]
        grippers, arms = [], []
        for arm_part in sorted(arm_parts):
            side = _side_of_name(arm_part) or "n/a"
            gripper_part = next(
                (part for part in gripper_parts if _side_of_name(part) == side), None
            )
            gripper = Gripper(gripper_part or (arm_part + "_ee"), side)
            grippers.append(gripper)
            arms.append(Arm(arm_part, side, robot_name, gripper))
        return grippers, arms

    def _build_episodes(
        self,
        scene: Dict[str, Any],
        frames_per_second: int,
        objects_by_id: Dict[str, BenchObject],
        place_area: Optional[BenchObject],
    ) -> List[ActionEpisode]:
        """
        Action episodes from the recorded plan segments.
        """

        def arm_for(segment: Dict[str, Any]) -> Optional[Arm]:
            """
            The arm matching the segment's recorded side hint, falling back to the first
            arm if the segment picks something but names no side.
            """
            hint = (segment.get("arm") or "").lower()
            for arm in self.arms:
                if arm.side and arm.side in hint:
                    return arm
            return self.arms[0] if self.arms and segment.get("picks") else None

        episodes = []
        for index, segment in enumerate(scene.get("segments") or []):
            picks = objects_by_id.get(segment.get("picks"))
            episodes.append(
                ActionEpisode(
                    name=segment["step"],
                    index=index,
                    start_frame=segment["start"],
                    end_frame=segment["end"],
                    duration_s=round(
                        (segment["end"] - segment["start"]) / max(1, frames_per_second),
                        1,
                    ),
                    performed_by=arm_for(segment) if picks else None,
                    picks=picks,
                    places_at=place_area if picks else None,
                )
            )
        return episodes

    @staticmethod
    def _build_joint_motions(
        trajectory: Dict[str, Any], parts: Dict[str, Any], robot_prefix: str
    ) -> List[JointMotion]:
        """
        Per-joint motion statistics over the whole recorded trajectory.
        """
        minimum: Dict[str, float] = {}
        maximum: Dict[str, float] = {}
        for frame in trajectory.get("frames") or []:
            for joint, value in frame.items():
                if joint not in minimum or value < minimum[joint]:
                    minimum[joint] = value
                if joint not in maximum or value > maximum[joint]:
                    maximum[joint] = value

        link_to_part = {link: part for part, links in parts.items() for link in links}

        def side_of(key: str) -> str:
            """
            Which arm side a prefixed joint key belongs to, or "environment"/ "body"
            when it isn't part of an arm.
            """
            prefix, _, joint_name = key.partition("/")
            if "/" not in key:
                prefix, joint_name = "", key
            if robot_prefix and prefix != robot_prefix:
                return "environment"
            part = link_to_part.get(joint_name.replace("_joint", "_link"))
            if part and _side_of_name(part):
                return _side_of_name(part)
            return _side_of_name(joint_name) or "body"

        return [
            JointMotion(
                name=key.partition("/")[2] or key,
                arm_side=side_of(key),
                min_rad=round(minimum[key], 3),
                max_rad=round(maximum[key], 3),
                range_rad=round(maximum[key] - minimum[key], 3),
            )
            for key in sorted(minimum)
        ]

    @staticmethod
    def _subpackage_of(package: str, module: str) -> str:
        """
        Qualified subpackage of a module path.

        ``coraplex.src.coraplex.plans.designator`` → ``coraplex.plans``; top-level
        modules collapse onto the package itself.
        """
        segments = module.split(".")
        if segments and segments[0] == package:
            segments = segments[1:]
        while segments and segments[0] in ("src", package):
            segments = segments[1:]
        return package + "." + segments[0] if len(segments) >= 2 else package

    @staticmethod
    def _build_subpackages(classes: List[PythonClass]) -> List[SubPackage]:
        """
        Subpackage entities aggregated from the scanned classes.
        """
        modules = defaultdict(set)
        class_counts = defaultdict(int)
        for entry in classes:
            if entry.subpackage != entry.package:
                modules[(entry.package, entry.subpackage)].add(entry.module)
                class_counts[entry.subpackage] += 1
        return [
            SubPackage(
                name=subpackage,
                package=package,
                module_count=len(modules[(package, subpackage)]),
                class_count=class_counts[subpackage],
            )
            for (package, subpackage) in sorted(modules)
        ]


@cache
def get_kb(selected_scene: Optional[str] = None) -> KB:
    """
    The requested scene's knowledge base, built on first use.
    """
    return KB(selected_scene)


def reset_kb() -> None:
    """
    Drop the cached KB (tests point CRAM_VIZ_SCENES at fixtures).
    """
    get_kb.cache_clear()


# %% EQL session -----------------------------------------------------------------
#: EQL factories re-exported into every query namespace
_FACTORY_NAMES = [
    "entity",
    "set_of",
    "variable",
    "an",
    "a",
    "the",
    "and_",
    "or_",
    "not_",
    "contains",
    "in_",
    "exists",
    "for_all",
    "count",
    "count_all",
    "average",
    "sum",
    "min",
    "max",
    "mode",
    "distinct",
    "flat_variable",
    "variable_from",
]


def fresh_namespace(selected_scene: Optional[str] = None) -> Dict[str, Any]:
    """
    A namespace for evaluating one EQL query (fresh variables each time).
    """
    kb = get_kb(selected_scene)
    namespace = {
        name: getattr(eql_factories, name)
        for name in _FACTORY_NAMES
        if hasattr(eql_factories, name)
    }
    namespace.update(
        Position=Position,
        Gripper=Gripper,
        Arm=Arm,
        Robot=Robot,
        BenchObject=BenchObject,
        ActionEpisode=ActionEpisode,
        JointMotion=JointMotion,
        Package=Package,
        SubPackage=SubPackage,
        PythonClass=PythonClass,
        Concept=action_knowledge.Concept,
        ActionPhase=action_knowledge.ActionPhase,
        ActionPhaseSequence=action_knowledge.ActionPhaseSequence,
        OntologyClass=action_knowledge.OntologyClass,
        Role=action_knowledge.Role,
        Parameter=action_knowledge.Parameter,
        Precondition=action_knowledge.Precondition,
        Goal=action_knowledge.Goal,
        ActionKnowledge=action_knowledge.ActionKnowledge,
        Property=entity_knowledge.Property,
        Quality=entity_knowledge.Quality,
        Affordance=entity_knowledge.Affordance,
        EntityRelation=entity_knowledge.EntityRelation,
        EntityKnowledge=entity_knowledge.EntityKnowledge,
        RegulationMode=core.RegulationMode,
        CouplingRegime=core.CouplingRegime,
        ConditionMode=core.ConditionMode,
        ComparisonOutcome=core.ComparisonOutcome,
        ForceKind=core.ForceKind,
        HasType=HasType,
        core=core,
        prehension=prehension,
        pouring=pouring,
        gaze=gaze,
        objects=kb.objects,
        episodes=kb.episodes,
        arms=kb.arms,
        grippers=kb.grippers,
        joints=kb.joints,
        robots=[kb.robot],
        packages=kb.packages,
        subpackages=kb.subpackages,
        classes=kb.classes,
        concepts=kb.concepts,
        phases=kb.action_phases,
        phase_sequences=kb.action_phase_sequences,
        ontology_classes=kb.ontology_classes,
        roles=kb.roles,
        parameters=kb.parameters,
        preconditions=kb.preconditions,
        goals=kb.goals,
        action_knowledge_entries=kb.action_knowledge_entries,
        properties=kb.properties,
        qualities=kb.qualities,
        affordances=kb.affordances,
        entity_relations=kb.entity_relations,
        entity_knowledge_entries=kb.entity_knowledge_entries,
        pouring_phases=kb.pouring_phases,
        pouring_roles=kb.pouring_roles,
        pouring_event_types=kb.pouring_event_types,
        pouring_force_dynamic_events=kb.pouring_force_dynamic_events,
        pouring_transitions=kb.pouring_transitions,
        pouring_control_points=kb.pouring_control_points,
        pouring_force_couplings=kb.pouring_force_couplings,
        pouring_continuous_predictions=kb.pouring_continuous_predictions,
    )
    # ready-made query variables so one-liners stay short
    namespace["obj"] = eql_factories.variable(BenchObject, domain=kb.objects)
    namespace["ep"] = eql_factories.variable(ActionEpisode, domain=kb.episodes)
    namespace["arm"] = eql_factories.variable(Arm, domain=kb.arms)
    namespace["j"] = eql_factories.variable(JointMotion, domain=kb.joints)
    namespace["rob"] = eql_factories.variable(Robot, domain=[kb.robot])
    namespace["pkg"] = eql_factories.variable(Package, domain=kb.packages)
    namespace["sub"] = eql_factories.variable(SubPackage, domain=kb.subpackages)
    namespace["cls"] = eql_factories.variable(PythonClass, domain=kb.classes)
    namespace["concept"] = eql_factories.variable(
        action_knowledge.Concept, domain=kb.concepts
    )
    namespace["phase"] = eql_factories.variable(
        action_knowledge.ActionPhase, domain=kb.action_phases
    )
    namespace["phase_sequence"] = eql_factories.variable(
        action_knowledge.ActionPhaseSequence, domain=kb.action_phase_sequences
    )
    namespace["oclass"] = eql_factories.variable(
        action_knowledge.OntologyClass, domain=kb.ontology_classes
    )
    namespace["role"] = eql_factories.variable(action_knowledge.Role, domain=kb.roles)
    namespace["parameter"] = eql_factories.variable(
        action_knowledge.Parameter, domain=kb.parameters
    )
    namespace["precondition"] = eql_factories.variable(
        action_knowledge.Precondition, domain=kb.preconditions
    )
    namespace["goal"] = eql_factories.variable(action_knowledge.Goal, domain=kb.goals)
    namespace["property"] = eql_factories.variable(
        entity_knowledge.Property, domain=kb.properties
    )
    namespace["quality"] = eql_factories.variable(
        entity_knowledge.Quality, domain=kb.qualities
    )
    namespace["affordance"] = eql_factories.variable(
        entity_knowledge.Affordance, domain=kb.affordances
    )
    namespace["relation"] = eql_factories.variable(
        entity_knowledge.EntityRelation, domain=kb.entity_relations
    )
    namespace["pouring_phase"] = eql_factories.variable(
        core.ActionPhase, domain=kb.pouring_phases
    )
    namespace["pouring_role"] = eql_factories.variable(
        core.ManipulationRole, domain=kb.pouring_roles
    )
    namespace["pouring_event_type"] = eql_factories.variable(
        core.PhaseBoundaryEventType, domain=kb.pouring_event_types
    )
    namespace["pouring_force_dynamic_event"] = eql_factories.variable(
        core.MechanicalEventType, domain=kb.pouring_force_dynamic_events
    )
    namespace["pouring_transition"] = eql_factories.variable(
        core.PhaseTransition, domain=kb.pouring_transitions
    )
    namespace["pouring_control_point"] = eql_factories.variable(
        core.SensorimotorControlPoint, domain=kb.pouring_control_points
    )
    namespace["pouring_coupling"] = eql_factories.variable(
        prehension.GripLoadCoupling, domain=kb.pouring_force_couplings
    )
    namespace["pouring_continuous_prediction"] = eql_factories.variable(
        core.ContinuousSignalPrediction,
        domain=kb.pouring_continuous_predictions,
    )
    namespace["pouring_condition_requirement"] = eql_factories.variable(
        core.ConditionRequirement, domain=kb.pouring_condition_requirements
    )
    namespace["pouring_parameter"] = eql_factories.variable(
        core.Parameter, domain=kb.pouring_parameters
    )
    namespace["pouring_sensory_prediction"] = eql_factories.variable(
        core.SensoryEventPrediction, domain=kb.pouring_sensory_predictions
    )
    namespace["pouring_comparison"] = eql_factories.variable(
        core.PredictionComparison, domain=kb.pouring_comparisons
    )
    namespace["pouring_correction"] = eql_factories.variable(
        core.CorrectiveAction, domain=kb.pouring_corrections
    )
    namespace["pouring_gaze_fixation"] = eql_factories.variable(
        gaze.GazeFixation, domain=kb.pouring_gaze_fixations
    )
    return namespace


def _entity_name(value: Any) -> Optional[str]:
    """
    The entity's name attribute, or None for non-entities.

    A knowledge fragment of the generated vocabulary has no ``name`` field -- its
    ``label`` plays the same "human-readable display name" role, and its class names
    it when the label is empty. It is handled explicitly instead of by a generic
    ``label``/``name`` fallback, which would misfire on a type such as ``BenchObject``
    that carries both a ``name`` (the identifier every other part of the graph
    highlights by) and an unrelated ``label``.
    """
    if isinstance(value, DescriptionEntity):
        return value.label or type(value).__name__
    return getattr(value, "name", None)


def _jsonable(value: Any) -> Any:
    """
    A JSON-serializable rendering of one query result value.

    An ontology term -- an enumerated individual or a concept a fragment cites -- is
    written with its namespace's prefix, such as ``core:Coupled``.
    """
    if isinstance(value, (tuple, list)):
        # a tuple of plain strings is content a
        # query answer names, not metadata to hide -- render it as a real list
        # rather than a Python repr string, which reads as debug output. Checked
        # before the dataclass case: a fragment's list field is a list that is also
        # a dataclass, whose repr would hide its members.
        return [_jsonable(element) for element in value]
    if isinstance(value, StrEnum):
        return OntologyBacking().prefixed(value.value)
    if isinstance(value, ConceptReference):
        return OntologyBacking().prefixed(value.iri)
    if is_dataclass(value) and not isinstance(value, type):
        return _entity_name(value) or repr(value)
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return repr(value)


def run_query(
    code: str, limit: int = 200, selected_scene: Optional[str] = None
) -> Dict[str, Any]:
    """
    Execute an EQL query string and return a JSON-able result payload.

    The last expression of ``code`` is the query; preceding statements are executed as
    setup.

    :param code: The EQL query source.
    :param limit: Maximum number of result rows to return.
    :param selected_scene: Scene whose entities form the query domain.
    """
    namespace = fresh_namespace(selected_scene)
    tree = ast.parse(code, mode="exec")
    if not tree.body:
        raise ValueError("empty query")
    last = tree.body[-1]
    if isinstance(last, ast.Expr):
        if len(tree.body) > 1:
            preamble = ast.Module(body=tree.body[:-1], type_ignores=[])
            exec(compile(preamble, "<eql>", "exec"), namespace)
        result = eval(compile(ast.Expression(last.value), "<eql>", "eval"), namespace)
    else:
        exec(compile(tree, "<eql>", "exec"), namespace)
        result = namespace.get("result")

    if hasattr(result, "evaluate"):
        result = result.evaluate()
    rows, highlight, more = _result_rows(result, limit)
    kind = "rows" if rows and "__entity__" not in rows[0] else "entities"
    return {
        "ok": True,
        "kind": kind,
        "rows": rows,
        "count": len(rows),
        "more": more,
        "highlight": sorted(set(highlight)),
    }


def _result_rows(
    result: Any, limit: int
) -> Tuple[List[Dict[str, Any]], List[str], bool]:
    """
    Render a query result into (answer rows, highlight ids, truncated).
    """
    rows: List[Dict[str, Any]] = []
    highlight: List[str] = []
    if result is None:
        return rows, highlight, False
    if isinstance(result, (str, int, float, bool)):
        rows.append({"value": _jsonable(result)})
        return rows, highlight, False
    if is_dataclass(result) and not isinstance(result, type):
        rows.append(_entity_row(result, highlight))
        return rows, highlight, False
    try:
        iterator = iter(result)
    except TypeError:
        rows.append({"value": _jsonable(result)})
        return rows, highlight, False
    for item in iterator:
        if len(rows) >= limit:
            return rows, highlight, True
        rows.append(_item_row(item, highlight))
    return rows, highlight, False


#: item type -> (its knowledge-category key, the field holding its owner's name), for
#: every entity registered under an owner (an action verb or a BenchObject name) but
#: drawn as a graph node shared globally by name across every owner that has one — see
#: :func:`_append_owned_highlight`
_OWNED_HIGHLIGHT_CATEGORY: Dict[type, Tuple[str, str]] = {
    action_knowledge.ActionPhase: ("phase", "action_verb"),
    action_knowledge.Role: ("roles", "action_verb"),
    action_knowledge.Parameter: ("parameters", "action_verb"),
    action_knowledge.Precondition: ("preconditions", "action_verb"),
    action_knowledge.Goal: ("goals", "action_verb"),
    entity_knowledge.Property: ("properties", "entity_name"),
    entity_knowledge.Quality: ("qualities", "entity_name"),
    entity_knowledge.Affordance: ("affordances", "entity_name"),
    entity_knowledge.EntityRelation: ("relations", "entity_name"),
}


def _append_owned_highlight(
    highlight: List[str], category: str, owner_field: str, item: Any
) -> None:
    """
    Append the real, globally shared graph node id for one owned knowledge item, if it
    was actually registered under some owner (auto-stamped by the owning
    ActionKnowledge/EntityKnowledge's ``__post_init__``).
    """
    if getattr(item, owner_field):
        highlight.append(action_knowledge.member_node_id(category, item.name))


def _entity_row(item: Any, highlight: List[str]) -> Dict[str, Any]:
    """
    One entity as an answer row; collects the ids to highlight.
    """
    name = _entity_name(item)
    _append_highlight(item, highlight)
    row = {"__entity__": name or repr(item), "__type__": type(item).__name__}
    if isinstance(item, DescriptionEntity):
        # a real knowledge fragment's other fields are relational (a phase's own
        # events, roles, conditions...), not flat data -- dumping them here repeats
        # what a list-valued field already renders as an unreadable repr(), and
        # duplicates what the Hypothesis Graph tab exists to explore by expanding
        # the node itself. The EQL answer names the matching entities; the graph
        # is where their structure is examined.
        return row
    for entity_field in fields(item):
        if entity_field.name != "name":
            row[entity_field.name] = _jsonable(getattr(item, entity_field.name))
    return row


def _append_highlight(item: Any, highlight: List[str]) -> None:
    """
    Collect the ids of the graph nodes standing for one answered entity.
    """
    name = _entity_name(item)
    if name:
        highlight.append(name)
    owned_category = _OWNED_HIGHLIGHT_CATEGORY.get(type(item))
    if isinstance(item, PythonClass):
        # classes aren't graph nodes — light up their subpackage + package instead
        highlight.append(item.subpackage)
        highlight.append(item.package)
    elif isinstance(item, action_knowledge.Concept):
        # concept nodes are id-prefixed (globally shared across phases) — the bare
        # name above never matches a graph node, so highlight the real node too
        highlight.append(action_knowledge.concept_node_id(item))
    elif isinstance(item, action_knowledge.OntologyClass):
        # ontology classes are globally shared graph nodes too
        highlight.append(action_knowledge.class_node_id(item))
    elif isinstance(item, action_knowledge.ActionPhaseSequence):
        # the sequence itself has no dedicated hub node any more (each of its
        # phases links directly to its verb) -- the verb's own bare node is its
        # anchor in the domain view
        highlight.append(item.name)
    elif isinstance(item, DescriptionEntity):
        # the display name above is the human-readable label, not the graph node
        # id -- the Hypothesis Graph tab keys its nodes by their projected identifier
        node_identifier = hypothesis_knowledge.node_identifier_for(item)
        if node_identifier is not None:
            highlight.append(node_identifier)
    elif owned_category:
        category, owner_field = owned_category
        _append_owned_highlight(highlight, category, owner_field, item)


def _item_row(item: Any, highlight: List[str]) -> Dict[str, Any]:
    """
    One arbitrary query result item as an answer row.
    """
    if is_dataclass(item) and not isinstance(item, type):
        return _entity_row(item, highlight)
    if hasattr(item, "items"):  # UnificationDict from set_of()
        row = {}
        for key, value in item.items():
            for answered in value if isinstance(value, list) else [value]:
                if is_dataclass(answered) and not isinstance(answered, type):
                    _append_highlight(answered, highlight)
            row[str(key)] = _jsonable(value)
        return row
    return {"value": _jsonable(item)}


def _edge_payload(edge: action_knowledge.KnowledgeGraphEdge) -> Dict[str, Any]:
    """
    The graph-payload dict shape of one
    :class:`~cram_viz.action_knowledge.KnowledgeGraphEdge`.
    """
    return {
        "from": edge.source,
        "to": edge.target,
        "kind": edge.kind,
        "label": edge.relation.value,
    }


def _merge_knowledge_fragment(
    add: Callable[[str, str, str, List[str]], None],
    edges: List[Dict[str, Any]],
    details: Dict[str, Any],
    anchor_name: str,
    fragment: Any,
) -> None:
    """
    Render one GraphFragment's own nodes/edges (guarded per-node, since a phase,
    role, ... shared by several owners must still be added only once) and its direct
    anchor edges to ``anchor_name``.
    """
    nodes, fragment_edges = fragment.nodes_and_edges()
    for node in nodes:
        if node.identifier not in details:
            add(node.identifier, node.label, node.group, list(node.lines))
    for edge in fragment_edges:
        edges.append(_edge_payload(edge))
    for edge in fragment.anchor_edges(anchor_name):
        edges.append(_edge_payload(edge))


def _merge_action_phase_knowledge(
    add: Callable[[str, str, str, List[str]], None],
    edges: List[Dict[str, Any]],
    details: Dict[str, Any],
    episode_name: str,
    sequences: List[action_knowledge.ActionPhaseSequence],
) -> None:
    """
    Attach any registered phase knowledge whose action verb matches this episode.

    Every phase is a node shared globally by name across every sequence that
    includes it, so it is added the first time any matching sequence renders it;
    :meth:`~cram_viz.action_knowledge.ActionPhaseGraphFragment.anchor_edges` links
    this episode straight to every one of its own phases.
    """
    for sequence in sequences:
        if sequence.matches(episode_name):
            _merge_knowledge_fragment(
                add,
                edges,
                details,
                episode_name,
                action_knowledge.ActionPhaseGraphFragment(sequence),
            )


def _merge_action_knowledge(
    add: Callable[[str, str, str, List[str]], None],
    edges: List[Dict[str, Any]],
    details: Dict[str, Any],
    episode_name: str,
    entries: List[action_knowledge.ActionKnowledge],
) -> None:
    """
    Attach any registered action knowledge (classification, roles, parameters,
    preconditions, goals) whose verb matches this episode.

    A role/parameter/precondition/goal is a node shared globally by name across
    every verb that has one with that name; this episode links directly to every
    one of its own.
    """
    for entry in entries:
        if entry.matches(episode_name):
            _merge_knowledge_fragment(
                add,
                edges,
                details,
                episode_name,
                action_knowledge.ActionKnowledgeGraphFragment(entry),
            )


def _merge_entity_knowledge(
    add: Callable[[str, str, str, List[str]], None],
    edges: List[Dict[str, Any]],
    details: Dict[str, Any],
    object_name: str,
    entries: List[entity_knowledge.EntityKnowledge],
) -> None:
    """
    Attach any registered entity knowledge (classification, properties, qualities,
    affordances, relations) registered for this object.

    A property/quality/affordance/relation is a node shared globally by name across
    every entity that has one with that name; this object links directly to every
    one of its own.
    """
    for entry in entries:
        if entry.matches(object_name):
            _merge_knowledge_fragment(
                add,
                edges,
                details,
                object_name,
                entity_knowledge.EntityKnowledgeGraphFragment(entry),
            )


# %% the UI graph -----------------------------------------------------------------
def _bench_object_detail_lines(bench_object: BenchObject) -> List[str]:
    """
    Detail-panel lines for one scene object's own node, shared by the Knowledge tab
    and the Domain Knowledge tab (see :func:`_domain_knowledge_view`), so an object
    anchored in both looks like the same real thing, not two different stand-ins.
    """
    return [
        "a BenchObject",
        "kind: " + bench_object.kind,
        "position: " + repr(bench_object.position),
        "height: %.2f m" % bench_object.height_m,
    ]


def _episode_detail_lines(episode: ActionEpisode) -> List[str]:
    """
    Detail-panel lines for one recorded episode's own node, shared by the Knowledge
    tab and the Domain Knowledge tab (see :func:`_domain_knowledge_view`).
    """
    return (
        [
            "an ActionEpisode",
            "frames %d–%d" % (episode.start_frame, episode.end_frame),
            "duration: %.1f s" % episode.duration_s,
        ]
        + (["picks: " + episode.picks.name] if episode.picks else [])
        + (["places at: " + episode.places_at.name] if episode.places_at else [])
    )


def graph_payload(selected_scene: Optional[str] = None) -> Dict[str, Any]:
    """
    The knowledge-graph overview: nodes, edges, details and presets.
    """
    kb = get_kb(selected_scene)
    nodes, edges, details = [], [], {}

    def add(node_id: str, label: str, group: str, lines: List[str]) -> None:
        """
        Append one graph node and its detail-panel entry.
        """
        nodes.append(
            {
                "id": node_id,
                "label": label,
                "group": group,
                "title": "\n".join([label] + lines),
            }
        )
        details[node_id] = {"label": label, "group": group, "lines": lines}

    rob = kb.robot.name
    add(
        rob,
        rob,
        "robot",
        [
            "a Robot",
            "%d arm%s" % (kb.robot.arm_count, "" if kb.robot.arm_count == 1 else "s"),
            "double-click: full URDF tree",
        ],
    )
    for arm in kb.arms:
        add(
            arm.name,
            arm.name.replace("_", " "),
            "robot",
            ["an Arm", "side: " + arm.side, "gripper: " + arm.gripper.name],
        )
        edges.append({"from": rob, "to": arm.name, "kind": "prop", "label": "has part"})
        add(
            arm.gripper.name,
            arm.gripper.name.replace("_", " "),
            "robot",
            [
                "a Gripper",
                "side: " + arm.gripper.side,
                "opening: %.3f m" % arm.gripper.opening_m,
            ],
        )
        edges.append(
            {
                "from": arm.name,
                "to": arm.gripper.name,
                "kind": "prop",
                "label": "has part",
            }
        )

    for bench_object in kb.objects:
        add(
            bench_object.name,
            bench_object.label,
            "object",
            _bench_object_detail_lines(bench_object),
        )

    previous = None
    for episode in kb.episodes:
        add(episode.name, episode.name, "event", _episode_detail_lines(episode))
        if previous:
            edges.append(
                {
                    "from": previous,
                    "to": episode.name,
                    "kind": "type",
                    "label": "precedes",
                }
            )
        previous = episode.name
        # the robot performs the episode (with its arm); don't wire the episode
        # straight to the arm — the arm hangs off the robot, so the chain reads
        # transport_milk → pr2 → left_arm → left_gripper
        if episode.performed_by:
            edges.append(
                {
                    "from": episode.name,
                    "to": episode.performed_by.robot,
                    "kind": "prop",
                    "label": "performed by",
                }
            )
        if episode.picks:
            edges.append(
                {
                    "from": episode.name,
                    "to": episode.picks.name,
                    "kind": "prop",
                    "label": "picks",
                }
            )
        if episode.places_at:
            edges.append(
                {
                    "from": episode.name,
                    "to": episode.places_at.name,
                    "kind": "prop",
                    "label": "places at",
                }
            )

    # the CRAM architecture cluster: repo root → packages, plus import edges
    if kb.packages:
        add(
            "cram",
            "CRAM architecture",
            "root",
            [
                "~/cognitive_robot_abstract_machine",
                "%d packages · %d Python classes" % (len(kb.packages), len(kb.classes)),
            ],
        )
        for package in kb.packages:
            add(
                package.name,
                package.name,
                "concept",
                [
                    "a Package",
                    package.description,
                    "%d modules · %d classes"
                    % (package.module_count, package.class_count),
                    "double-click to open",
                ],
            )
            edges.append(
                {
                    "from": "cram",
                    "to": package.name,
                    "kind": "prop",
                    "label": "contains",
                }
            )
        for subpackage in kb.subpackages:
            add(
                subpackage.name,
                subpackage.name.split(".", 1)[1],
                "klass",
                [
                    "a SubPackage of " + subpackage.package,
                    "%d modules · %d classes"
                    % (subpackage.module_count, subpackage.class_count),
                    "double-click to open",
                ],
            )
            edges.append(
                {
                    "from": subpackage.package,
                    "to": subpackage.name,
                    "kind": "prop",
                    "label": "contains",
                }
            )
        for source, target in kb.package_deps:
            edges.append(
                {"from": source, "to": target, "kind": "type", "label": "imports"}
            )

        # ground the demo in the architecture at the SUBPACKAGE that actually
        # realises each part (only wire to a node that exists in this view)
        def link(src: str, dst: str, label: str) -> None:
            """
            Add an edge, but only if dst is actually a node in this view.
            """
            if any(n["id"] == dst for n in nodes):
                edges.append({"from": src, "to": dst, "kind": "type", "label": label})

        # anchor one representative manipulation episode (they share the stack)
        anchor = next((episode.name for episode in kb.episodes if episode.picks), None)
        if anchor:
            link(anchor, "coraplex.plans", "planned by")  # plan / designator layer
            link(anchor, "giskardpy.motion_statechart", "motion by")  # motion execution
        # every physical thing in the scene is modelled in the semantic digital twin
        link(rob, "semantic_digital_twin", "modelled in")
        for bench_object in kb.objects:
            link(bench_object.name, "semantic_digital_twin", "modelled in")

    # the executed plan tree (captured from the real PlanNode graph)
    scene, _ = load_scene(selected_scene)
    if scene.get("planTrees"):
        node_count = sum(_count_plan_nodes(tree) for tree in scene["planTrees"])
        add(
            "plan",
            "executed plan",
            "goal",
            [
                "the plan tree the demo actually executed",
                "%d nodes" % node_count,
                "double-click to open",
            ],
        )
        edges.append(
            {"from": "plan", "to": rob, "kind": "prop", "label": "executed by"}
        )
        for episode in kb.episodes:
            edges.append(
                {"from": "plan", "to": episode.name, "kind": "type", "label": "spans"}
            )

    status = "EQL ready · %d graph nodes · %d joints · %d CRAM classes" % (
        len(nodes),
        len(kb.joints),
        len(kb.classes),
    )
    return {
        "ok": True,
        "status": status,
        "nodes": nodes,
        "edges": edges,
        "details": details,
        "presets": get_presets(selected_scene),
    }


# %% drill-down subgraphs -----------------------------------------------------
# Double-clicking a node in the UI asks for its inside view: package → its
# subpackages + top-level classes, subpackage → its classes (with inheritance
# edges), class → its base classes and every subclass in the repo.

#: at most this many classes are drawn in one drill-down view
CLASS_CAP = 150


def _view() -> tuple:
    """
    Fresh (nodes, edges, details, add) accumulators for one subgraph.
    """
    nodes, edges, details = [], [], {}

    def add(
        node_id: str, label: str, group: str, lines: List[str], **extra: Any
    ) -> None:
        """
        Append one graph node (plus arbitrary extra fields) and its detail entry.
        """
        node = {
            "id": node_id,
            "label": label,
            "group": group,
            "title": "\n".join([label] + lines),
        }
        node.update(extra)
        nodes.append(node)
        details[node_id] = {"label": label, "group": group, "lines": lines}

    return nodes, edges, details, add


# %% the graph-panel tabs ---------------------------------------------------------
def view_payload(name: str, selected_scene: Optional[str] = None) -> Dict[str, Any]:
    """
    One tab of the graph panel.

    ``knowledge`` is the entity graph (the default, with drill-down); the others are
    structural views of the same demo that the UI can overlay with live status from the
    bridge (see :mod:`cram_viz.live.http`, ``/plan`` and ``/chart``).
    """
    kb = get_kb(selected_scene)
    if name == "knowledge":
        return graph_payload(selected_scene)
    if name == "kinematics":
        return _urdf_view(kb, selected_scene)
    if name == "plan":
        return _plan_view(kb, selected_scene)
    if name == "chart":
        return _chart_view(kb)
    if name == "domain":
        return _domain_knowledge_view(kb)
    if name == "hypothesis":
        return _hypothesis_graph_view()
    if name == "ontology":
        return _ontology_graph_view()
    if name == "designator":
        return _designator_view(kb)
    return {"ok": False, "error": "unknown view: %s" % name}


def _domain_knowledge_view(kb: KB) -> Dict[str, Any]:
    """
    The static domain-knowledge ontology, anchored on the same real scene entities the
    Knowledge tab shows wherever one exists: every action verb's phases,
    classification, roles, parameters, preconditions and goals attach directly to its
    matching recorded episode(s), and every registered entity's classification,
    properties, qualities, affordances and relations attach to its matching
    :class:`BenchObject`.

    This still shows the whole :data:`~cram_viz.action_knowledge.ACTION_PHASE_KNOWLEDGE`
    / :data:`~cram_viz.action_knowledge.ACTION_KNOWLEDGE` /
    :data:`~cram_viz.entity_knowledge.ENTITY_KNOWLEDGE` registries regardless of what
    was actually recorded -- a verb or entity with no match in the current scene falls
    back to its own bare name, exactly as before -- but a match makes this tab an
    extension of the same scene graph rather than an unconnected, parallel one.
    """
    nodes, edges, details, add = _view()
    verbs = sorted(
        {sequence.name for sequence in kb.action_phase_sequences}
        | {entry.name for entry in kb.action_knowledge_entries}
    )
    for verb in verbs:
        matching_episodes = [
            episode
            for episode in kb.episodes
            if action_knowledge.verb_matches_episode(verb, episode.name)
        ]
        anchors = [episode.name for episode in matching_episodes] or [verb]
        for episode in matching_episodes:
            add(episode.name, episode.name, "event", _episode_detail_lines(episode))
        if not matching_episodes:
            add(verb, verb, "event", ["a registered action verb"])
        for anchor in anchors:
            _merge_action_phase_knowledge(
                add, edges, details, anchor, kb.action_phase_sequences
            )
            _merge_action_knowledge(
                add, edges, details, anchor, kb.action_knowledge_entries
            )
    for entry in kb.entity_knowledge_entries:
        match = next(
            (
                bench_object
                for bench_object in kb.objects
                if entry.matches(bench_object.name)
            ),
            None,
        )
        if match:
            add(match.name, match.label, "object", _bench_object_detail_lines(match))
        else:
            add(entry.name, entry.name, "object", ["a registered entity"])
        anchor = match.name if match else entry.name
        _merge_entity_knowledge(
            add, edges, details, anchor, kb.entity_knowledge_entries
        )
    return {
        "ok": True,
        "crumb": "domain knowledge",
        "nodes": nodes,
        "edges": edges,
        "details": details,
        "empty": "No action or entity knowledge registered yet.",
    }


#: legend rows for the Designator tab -- overrides graph.js's default group labels
#: with what these groups mean in this view specifically
_DESIGNATOR_LEGEND = [
    {"group": "root", "label": "Designator"},
    {"group": "concept", "label": "Value slot (a model may generate this)"},
    {"group": "goal", "label": "Referent slot (must be a real individual)"},
    {
        "group": "object",
        "label": "Illustrative candidate (a scene object, not a real argument)",
    },
    {"group": "upper", "label": "Action-frame property group (from Hypothesis Graph)"},
    {"group": "ind", "label": "Property value"},
]


def _designator_slot_lines(slot: Any, resolved_for_instruction: str) -> List[str]:
    """
    Detail-panel lines for one reflected designator slot.
    """
    lines = ["type: " + slot.type_name, "kind: " + slot.kind.name.lower()]
    if slot.docstring:
        lines.append(slot.docstring)
    if slot.has_default:
        lines.append("default: " + repr(slot.default))
    if slot.role is not None:
        lines.append("fills the role: " + slot.role.__name__)
    resolved = slot.resolved_candidate
    if resolved is not None:
        lines.append(
            "Resolved illustratively to %r for the instruction %r, matching the "
            "hypothesis's %s filler -- still not a live resolver run."
            % (resolved.label, resolved_for_instruction, slot.role.__name__)
        )
    elif slot.kind is SlotKind.REFERENT:
        lines.append(
            "No live World is attached, so no real domain can be listed -- the "
            "candidates below are the scene's own objects, offered only as "
            "illustrative stand-ins."
        )
    return lines


def _designator_candidate_lines(
    candidate: Any, resolved_for_instruction: str
) -> List[str]:
    """
    Detail-panel lines for one referent slot's illustrative candidate.
    """
    if candidate.resolved:
        return ["Resolved for the instruction %r." % resolved_for_instruction]
    return ["an illustrative stand-in -- not a real Body/Tool individual"]


def _designator_lisp(knowledge: Any) -> Dict[str, Any]:
    """
    The Designator tab's left pane: the reflected designator as Lisp keyword syntax,
    one slot per line, each carrying the graph node id its value stands for so the
    frontend can highlight and focus that node when the line is clicked, and the
    ontology role it fills.
    """
    return {
        "head": knowledge.designator_name,
        "slots": [
            {
                "nodeId": "%s.%s" % (knowledge.designator_name, slot.name),
                "keyword": slot.lisp_keyword,
                "value": slot.lisp_value,
                "kind": slot.kind.name.lower(),
                "role": slot.role.__name__ if slot.role else None,
                "resolved": slot.resolved_candidate is not None,
            }
            for slot in knowledge.slots
        ],
    }


def _designator_view(kb: KB) -> Dict[str, Any]:
    """
    The Designator tab: :class:`~coraplex.robot_plans.actions.composite.tool_based.
    PouringAction`'s own declared fields, reflected directly and classified value
    versus referent (see :mod:`cram_viz.designator_knowledge`).

    Static and illustrative throughout: no live World is touched and no resolver is
    called. A referent slot's candidates are the current scene's own bench objects
    rather than real ``Body``/``Tool`` individuals, and the one reading like the active
    hypothesis's filler of the slot's role is marked resolved.
    """
    active_hypothesis = hypothesis_knowledge.active()
    knowledge = designator_knowledge.reflect_pouring_action(
        kb.objects, active_hypothesis.view if active_hypothesis else None
    )
    nodes, edges, details, add = _view()
    add(
        knowledge.designator_name,
        knowledge.designator_name,
        "root",
        [knowledge.docstring] if knowledge.docstring else [],
    )
    for slot in knowledge.slots:
        slot_id = "%s.%s" % (knowledge.designator_name, slot.name)
        group = "goal" if slot.kind is SlotKind.REFERENT else "concept"
        add(
            slot_id,
            slot.name,
            group,
            _designator_slot_lines(slot, knowledge.resolved_for_instruction),
        )
        edges.append(
            {
                "from": knowledge.designator_name,
                "to": slot_id,
                "kind": "prop",
                "label": "slot",
            }
        )
        for candidate in slot.candidates:
            candidate_id = "%s::%s" % (slot_id, candidate.identifier)
            add(
                candidate_id,
                candidate.label,
                "object",
                _designator_candidate_lines(
                    candidate, knowledge.resolved_for_instruction
                ),
            )
            edges.append(
                {
                    "from": slot_id,
                    "to": candidate_id,
                    "kind": "isa" if candidate.resolved else "type",
                    "label": "resolved" if candidate.resolved else "e.g.",
                }
            )
    return {
        "ok": True,
        "crumb": "PouringAction designator (illustrative)",
        "nodes": nodes,
        "edges": edges,
        "details": details,
        "legend": _DESIGNATOR_LEGEND,
        "lisp": _designator_lisp(knowledge),
    }


def _hypothesis_node_lines(node: HypothesisGraphNode) -> List[str]:
    """
    Detail-panel lines for one projected node.
    """
    return [
        "kind: " + node.kind.value,
        "type: " + node.type_name,
        "identifier: " + node.identifier,
    ]


def _hypothesis_graph_view() -> Dict[str, Any]:
    """
    The Hypothesis Graph tab: a live projection of the active pouring structure
    (real, EQL-queryable), presented phase first (see
    :mod:`cram_viz.hypothesis_knowledge`).

    Every node and edge of the projection is included; the frontend (see
    ``hypothesis-graph.js``) owns its own rendering engine and starts the view
    collapsed to just the instruction node, progressively revealing children as the
    user clicks through it -- see that file for why this tab does not go through the
    shared ``graph.js``/``window.Graph`` wrapper the other tabs use.

    The tab is empty until an instruction has been generated.
    """
    return _projection_view(
        crumb="hypothesis graph",
        node_label=lambda node: node.label,
        relation_label=lambda relation: relation.predicate,
        node_lines=_hypothesis_node_lines,
    )


def _ontology_graph_view() -> Dict[str, Any]:
    """
    The Ontology Graph tab: the Hypothesis Graph with the ontology class of each
    fragment beneath its label and the ontology property stating each link beneath its
    field; links only the viewer draws say so. A node's details name its class, the
    CRAM-AICOR and DUL terms the class specialises, and the ontology every value comes
    from (see :mod:`cram_viz.hypothesis_knowledge.ontology_backing`).
    """
    backing = OntologyBacking()
    return _projection_view(
        crumb="ontology graph",
        node_label=backing.node_label,
        relation_label=backing.relation_label,
        node_lines=backing.node_lines,
    )


def _projection_view(
    crumb: str,
    node_label: Callable[[HypothesisGraphNode], str],
    relation_label: Callable[[HypothesisGraphRelation], str],
    node_lines: Callable[[HypothesisGraphNode], List[str]],
) -> Dict[str, Any]:
    """
    A tab drawing the active hypothesis's projection, with the given node labels, link
    labels and detail lines; empty until an instruction has been generated.
    """
    active_hypothesis = hypothesis_knowledge.active()
    if active_hypothesis is None:
        return {
            "ok": True,
            "crumb": crumb,
            "root": None,
            "nodes": [],
            "edges": [],
            "details": {},
            "empty": (
                "Nothing generated yet — enter an instruction in the Generate bar "
                "above the EQL query box."
            ),
        }
    projection = active_hypothesis.projection
    nodes = [
        {
            "id": node.identifier,
            "label": node_label(node),
            "kind": node.kind.value,
        }
        for node in projection.nodes
    ]
    edges = [
        {
            "from": relation.source.identifier,
            "to": relation.target.identifier,
            "label": relation_label(relation),
        }
        for relation in projection.relations
    ]
    details = {
        node.identifier: {
            "label": node.label,
            "group": "hypothesis",
            "lines": node_lines(node),
        }
        for node in projection.nodes
    }
    return {
        "ok": True,
        "crumb": crumb,
        "nodes": nodes,
        "edges": edges,
        "details": details,
        "root": projection.instruction.identifier,
        "empty": "No hypothesis knowledge graph available.",
    }


def _chart_view(kb: KB) -> Dict[str, Any]:
    """
    The (live-only) statechart tab.

    Motion statecharts only exist while giskardpy executes them: one is
    compiled per merged motion group and thrown away afterwards, and nothing
    of it is recorded into the bundle — the UI fills this view from the
    bridge's ``/chart`` while attached.
    """
    return {
        "ok": True,
        "crumb": "motion statechart",
        "nodes": [],
        "edges": [],
        "details": {},
        "layout": "hier",
        "live": "chart",
        "empty": "Motion statecharts are built and ticked at execution time. "
        "Start the demo with cram-viz-live and press ◉ Live — "
        "the statechart of the running motion group appears here, "
        "coloured by its node life cycle.",
    }


def _class_id(python_class: PythonClass) -> str:
    """
    Graph node id of a scanned class (module-qualified).
    """
    return python_class.module + "." + python_class.name


def _class_lines(python_class: PythonClass, drill_hint: bool = True) -> List[str]:
    """
    Detail lines shown for a class node.
    """
    lines = [
        "a PythonClass",
        "package: " + python_class.package,
        "module: " + python_class.module,
        "methods: %d" % python_class.methods,
    ]
    if python_class.bases:
        lines.append("bases: " + ", ".join(python_class.bases))
    if python_class.doc:
        lines.append(python_class.doc)
    if drill_hint:
        lines.append("double-click: inheritance view")
    return lines


def _add_classes(
    add: Any,
    edges: List[Dict[str, Any]],
    parent_id: str,
    shown: List[PythonClass],
    total: int,
) -> List[str]:
    """
    Add class nodes plus their on-screen inheritance edges to a view.

    :return: Extra detail lines for the parent (a truncation notice, if any).
    """
    name_to_id: Dict[str, str] = {}
    for python_class in shown:
        class_id = _class_id(python_class)
        add(class_id, python_class.name, "pyclass", _class_lines(python_class))
        edges.append(
            {"from": parent_id, "to": class_id, "kind": "prop", "label": "defines"}
        )
        name_to_id.setdefault(python_class.name, class_id)
    for python_class in shown:
        for base in python_class.bases:
            if base in name_to_id and name_to_id[base] != _class_id(python_class):
                edges.append(
                    {
                        "from": _class_id(python_class),
                        "to": name_to_id[base],
                        "kind": "type",
                        "label": "inherits",
                    }
                )
    if total > len(shown):
        return [
            "showing the %d largest of %d classes (by method count)"
            % (len(shown), total)
        ]
    return []


def _count_plan_nodes(tree: Dict[str, Any]) -> int:
    """
    Number of nodes in a serialized plan tree.
    """
    return 1 + sum(_count_plan_nodes(child) for child in tree.get("children", []))


def expand_node(
    node_id: str, selected_scene: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    The inside view of a double-clicked node, or None if not drillable.
    """
    kb = get_kb(selected_scene)
    if node_id == kb.robot.name:  # robot → full URDF kinematic tree
        return _urdf_view(kb, selected_scene)
    if node_id == "plan":  # → the executed plan tree
        return _plan_view(kb, selected_scene)
    package = next((entry for entry in kb.packages if entry.name == node_id), None)
    if package:
        return _package_view(kb, package)
    subpackage = next(
        (entry for entry in kb.subpackages if entry.name == node_id), None
    )
    if subpackage:
        return _subpackage_view(kb, subpackage)
    python_class = next(
        (entry for entry in kb.classes if _class_id(entry) == node_id), None
    )
    if python_class:
        return _class_view(kb, python_class)
    return None


#: plan-node kind → node colour group of the graph panel
PLAN_GROUPS = {
    "ActionNode": "event",
    "MotionNode": "robot",
    "ConditionNode": "goal",
    "AttachmentNode": "object",
    "DetachmentNode": "object",
}

#: legend rows of the plan view
PLAN_LEGEND = [
    {"group": "event", "label": "Action"},
    {"group": "robot", "label": "Motion"},
    {"group": "goal", "label": "Condition"},
    {"group": "object", "label": "Attach / detach"},
    {"group": "ind", "label": "Other plan node"},
]


def _plan_view(kb: KB, selected_scene: Optional[str] = None) -> Dict[str, Any]:
    """
    The executed plan as a tree, one node per plan node the demo ran.

    The recorded statuses are thin on purpose: coraplex performs only the
    plan *root* (``Plan.perform`` → ``root.perform``), while
    ``ActionNode.notify`` merely expands its children into the merged motion
    statechart. So every inner node of a recorded tree reads ``CREATED``, and
    real per-step progress only shows up while the live bridge is attached
    (it derives it from the statechart life cycle).
    """
    scene, _ = load_scene(selected_scene)
    trees = scene.get("planTrees") or []
    nodes, edges, details, add = _view()
    counter = [0]

    def walk(tree: Dict[str, Any], parent: Optional[str]) -> None:
        """
        Add this plan node (with a freshly assigned id) and recurse into its children.
        """
        node_id = "pn%d" % counter[0]
        counter[0] += 1
        status = tree.get("status") or "CREATED"
        lines = ["a " + tree.get("kind", "PlanNode"), "status: " + status]
        if tree.get("arm"):
            lines.append("arm: " + tree["arm"])
        if tree.get("target"):
            lines.append("target: " + tree["target"])
        label = tree.get("label", "?").replace("Action", "")
        add(
            node_id,
            label,
            PLAN_GROUPS.get(tree.get("kind"), "ind"),
            lines,
            status=status,
        )
        if parent:
            edges.append(
                {"from": parent, "to": node_id, "kind": "prop", "label": "has step"}
            )
        for child in tree.get("children", []):
            walk(child, node_id)

    for tree in trees:
        walk(tree, None)
    return {
        "ok": True,
        "crumb": "executed plan",
        "nodes": nodes,
        "edges": edges,
        "details": details,
        "legend": PLAN_LEGEND,
        "layout": "hier",
        "live": "plan",
        "statusLegend": True,
        "empty": "No plan tree in this bundle — re-run cram-viz-onboard.",
    }


def _urdf_view(kb: KB, selected_scene: Optional[str] = None) -> Dict[str, Any]:
    """
    The scene robot's URDF as a kinematic tree.

    Every link is a node, every joint an edge (parent → child); movable joints are solid
    edges, fixed ones dashed. Links are coloured by robot part from the recorded
    annotation.
    """
    links, joints = load_urdf(selected_scene)
    nodes, edges, details, add = _view()
    if not links:
        return {
            "ok": True,
            "crumb": kb.robot.name + " · URDF (not found)",
            "nodes": [],
            "edges": [],
            "details": {},
        }

    scene, _ = load_scene(selected_scene)
    parts = (scene.get("robot") or {}).get("parts") or {}
    link_to_part = {
        link: part for part, part_links in parts.items() for link in part_links
    }

    def chain_group(link_name: str) -> str:
        """
        The visual group (colour) a kinematic-chain link is bucketed into.
        """
        part = link_to_part.get(link_name, "").lower()
        if "gripper" in part or "hand" in part or "effector" in part:
            return "object"  # grippers (teal)
        if "left" in part:
            return "robot"  # left arm (pink)
        if "right" in part:
            return "event"  # right arm (purple)
        lowered = link_name.lower()
        if any(
            keyword in lowered
            for keyword in ("head", "stereo", "sensor", "kinect", "camera", "laser")
        ):
            return "goal"  # head / sensors (amber)
        return "concept"  # base, torso, casters (green)

    # which joint drives each link (child link → its parent joint), for tooltips
    parent_joint = {joint["child"]: joint for joint in joints}
    for link in links:
        joint = parent_joint.get(link)
        lines = ["a URDF Link"]
        if joint:
            lines.append("joint: %s (%s)" % (joint["name"], joint["type"]))
            lines.append("parent link: " + joint["parent"])
        else:
            lines.append("root link")
        add("urdf:" + link, link, chain_group(link), lines)
    for joint in joints:
        if ("urdf:" + joint["parent"]) in details and (
            "urdf:" + joint["child"]
        ) in details:
            movable = joint["type"] not in ("fixed",)
            edges.append(
                {
                    "from": "urdf:" + joint["parent"],
                    "to": "urdf:" + joint["child"],
                    "kind": "prop" if movable else "type",
                    "label": "%s (%s)" % (joint["name"], joint["type"]),
                }
            )
    revolute_count = sum(1 for joint in joints if joint["type"] == "revolute")
    details["urdf:" + links[0]]["lines"].append(
        "%d links · %d joints (%d movable)" % (len(links), len(joints), revolute_count)
    )
    legend = [
        {"group": "concept", "label": "Base / torso"},
        {"group": "robot", "label": "Left arm"},
        {"group": "event", "label": "Right arm"},
        {"group": "object", "label": "Grippers"},
        {"group": "goal", "label": "Head / sensors"},
    ]
    # force-directed, not hierarchical: the chains read better when the arms and
    # the sensor head spread out around the base than as one wide LR tree
    return {
        "ok": True,
        "crumb": kb.robot.name + " · URDF",
        "nodes": nodes,
        "edges": edges,
        "details": details,
        "legend": legend,
    }


def _package_view(kb: KB, package: Package) -> Dict[str, Any]:
    """
    Inside view of a package: its subpackages and top-level classes.
    """
    nodes, edges, details, add = _view()
    subpackages = [entry for entry in kb.subpackages if entry.package == package.name]
    top_level = sorted(
        (
            entry
            for entry in kb.classes
            if entry.package == package.name and entry.subpackage == package.name
        ),
        key=lambda entry: -entry.methods,
    )
    add(
        package.name,
        package.name,
        "concept",
        [
            "a Package",
            package.description,
            "%d modules · %d classes" % (package.module_count, package.class_count),
        ],
    )
    for subpackage in subpackages:
        add(
            subpackage.name,
            subpackage.name.split(".", 1)[1],
            "klass",
            [
                "a SubPackage of " + subpackage.package,
                "%d modules · %d classes"
                % (subpackage.module_count, subpackage.class_count),
                "double-click to open",
            ],
        )
        edges.append(
            {
                "from": package.name,
                "to": subpackage.name,
                "kind": "prop",
                "label": "contains",
            }
        )
    note = _add_classes(add, edges, package.name, top_level[:CLASS_CAP], len(top_level))
    if note:
        details[package.name]["lines"] += note
    return {
        "ok": True,
        "crumb": package.name,
        "nodes": nodes,
        "edges": edges,
        "details": details,
    }


def _subpackage_view(kb: KB, subpackage: SubPackage) -> Dict[str, Any]:
    """
    Inside view of a subpackage: its classes with inheritance edges.
    """
    nodes, edges, details, add = _view()
    classes = sorted(
        (entry for entry in kb.classes if entry.subpackage == subpackage.name),
        key=lambda entry: -entry.methods,
    )
    add(
        subpackage.name,
        subpackage.name.split(".", 1)[1],
        "klass",
        [
            "a SubPackage of " + subpackage.package,
            "%d modules · %d classes"
            % (subpackage.module_count, subpackage.class_count),
        ],
    )
    note = _add_classes(add, edges, subpackage.name, classes[:CLASS_CAP], len(classes))
    if note:
        details[subpackage.name]["lines"] += note
    return {
        "ok": True,
        "crumb": subpackage.name.split(".", 1)[1],
        "nodes": nodes,
        "edges": edges,
        "details": details,
    }


#: at most this many subclasses are drawn in a class inheritance view
SUBCLASS_CAP = 80


def _class_view(kb: KB, python_class: PythonClass) -> Dict[str, Any]:
    """
    Inheritance view of one class: bases above, repo subclasses below.
    """
    nodes, edges, details, add = _view()
    class_id = _class_id(python_class)
    add(
        class_id,
        python_class.name,
        "pyclass",
        _class_lines(python_class, drill_hint=False),
    )
    # direct base classes: resolve inside the repo (same package preferred),
    # otherwise show them as external
    for base in python_class.bases:
        candidates = [entry for entry in kb.classes if entry.name == base]
        pick = next(
            (entry for entry in candidates if entry.package == python_class.package),
            candidates[0] if candidates else None,
        )
        if pick:
            base_id = _class_id(pick)
            if base_id not in details:
                add(base_id, pick.name, "pyclass", _class_lines(pick))
        else:
            base_id = "ext:" + base
            if base_id not in details:
                add(base_id, base, "upper", ["external base class (outside the repo)"])
        edges.append(
            {"from": class_id, "to": base_id, "kind": "type", "label": "inherits"}
        )
    # every subclass in the repo (matched by base name)
    subclasses = [
        entry
        for entry in kb.classes
        if python_class.name in entry.bases and _class_id(entry) != class_id
    ]
    for subclass in subclasses[:SUBCLASS_CAP]:
        subclass_id = _class_id(subclass)
        if subclass_id not in details:
            add(subclass_id, subclass.name, "pyclass", _class_lines(subclass))
        edges.append(
            {"from": subclass_id, "to": class_id, "kind": "type", "label": "inherits"}
        )
    if len(subclasses) > SUBCLASS_CAP:
        details[class_id]["lines"].append(
            "showing %d of %d subclasses" % (SUBCLASS_CAP, len(subclasses))
        )
    return {
        "ok": True,
        "crumb": python_class.name,
        "nodes": nodes,
        "edges": edges,
        "details": details,
    }


#: static presets for the architecture side of the graph
ARCH_PRESETS = [
    {
        "text": "CRAM packages by size",
        "code": "set_of(pkg.name, pkg.class_count).ordered_by(pkg.class_count, descending=True)",
    },
    {
        "text": "all Designator classes",
        "code": "an(entity(cls).where(cls.name.endswith('Designator')))",
    },
    {
        "text": "where does EQL live?",
        "code": "set_of(cls.name, cls.module).where(in_('entity_query_language', cls.module)).limit(15)",
    },
    {
        "text": "subclasses of Symbol",
        "code": "an(entity(cls).where(in_('Symbol', cls.bases)))",
    },
    {
        "text": "inside coraplex",
        "code": "an(entity(sub).where(sub.package == 'coraplex'))",
    },
]


def get_presets(selected_scene: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Ready-made queries for the EQL panel.

    Scene presets are generated from the loaded scene, so they stay valid for any
    onboarded robot/environment; the architecture presets are static.
    """
    kb = get_kb(selected_scene)
    presets = [
        {"text": "which robot is this?", "code": "the(entity(rob))"},
        {"text": "which arms does it have?", "code": "an(entity(arm))"},
        {"text": "each arm and its gripper", "code": "set_of(arm.side, arm.gripper)"},
        {"text": "what is in the scene?", "code": "an(entity(obj))"},
        {
            "text": "what gets moved?",
            "code": "an(entity(ep.picks).where(ep.picks != None))",
        },
    ]
    first_object = next((entry for entry in kb.objects if entry.kind == "object"), None)
    if first_object:
        presets.append(
            {
                "text": "the %s" % first_object.label.lower(),
                "code": "the(entity(obj).where(obj.name == '%s'))" % first_object.name,
            }
        )
    manipulation = next((episode for episode in kb.episodes if episode.picks), None)
    if manipulation:
        if manipulation.places_at:
            presets.append(
                {
                    "text": "where does it place them?",
                    "code": "the(entity(ep.places_at).where(ep.name == '%s'))"
                    % manipulation.name,
                }
            )
        if manipulation.performed_by:
            presets.append(
                {
                    "text": "which arm does '%s'?" % manipulation.name,
                    "code": "the(entity(ep.performed_by).where(ep.name == '%s'))"
                    % manipulation.name,
                }
            )
    presets += _scene_verb_presets(kb)
    presets += _domain_knowledge_presets(kb)
    return _hypothesis_graph_presets(kb) + presets + ARCH_PRESETS


def _example_action_verb(kb: KB) -> Optional[str]:
    """
    A representative registered action verb, preferring one with full ActionKnowledge
    (roles/parameters/preconditions/goals) over one with only phase knowledge.
    """
    if kb.action_knowledge_entries:
        return kb.action_knowledge_entries[0].name
    if kb.action_phase_sequences:
        return kb.action_phase_sequences[0].name
    return None


def _domain_knowledge_presets(kb: KB) -> List[Dict[str, str]]:
    """
    Ready-made queries over the static action/entity knowledge (the Domain Knowledge
    tab), generated from whichever verbs and entities are actually registered -- like
    the scene presets above, these stay valid for any registry content.
    """
    presets: List[Dict[str, str]] = []
    example_verb = _example_action_verb(kb)
    example_action_knowledge = (
        kb.action_knowledge_entries[0] if kb.action_knowledge_entries else None
    )
    if example_action_knowledge:
        presets.append(
            {
                "text": "who/what is involved in '%s'?" % example_verb,
                "code": "an(entity(role).where(role.action_verb == '%s'))"
                % example_verb,
            }
        )
        presets.append(
            {
                "text": "what must be true before '%s' can start?" % example_verb,
                "code": (
                    "an(entity(precondition).where(precondition.action_verb == '%s'))"
                    % example_verb
                ),
            }
        )
        presets.append(
            {
                "text": "what does '%s' achieve?" % example_verb,
                "code": "an(entity(goal).where(goal.action_verb == '%s'))"
                % example_verb,
            }
        )
        if example_action_knowledge.parameters:
            presets.append(
                {
                    "text": "how is '%s' typically parameterized?" % example_verb,
                    "code": (
                        "an(entity(parameter).where(parameter.action_verb == '%s'))"
                        % example_verb
                    ),
                }
            )
        if any(
            role.name.startswith("agent:") for role in example_action_knowledge.roles
        ):
            presets.append(
                {
                    "text": "who is the agent that carries out '%s'?" % example_verb,
                    "code": (
                        "the(entity(role).where(role.action_verb == '%s')"
                        ".where(role.name.startswith('agent:')))"
                    )
                    % example_verb,
                }
            )
        if example_action_knowledge.classes:
            presets.append(
                {
                    "text": "what kind of action is '%s'?" % example_verb,
                    "code": "the(entity(oclass).where(oclass.name == '%s'))"
                    % example_action_knowledge.classes[-1].name,
                }
            )
    if example_verb and any(
        sequence.name == example_verb and sequence.phases
        for sequence in kb.action_phase_sequences
    ):
        presets.append(
            {
                "text": "what phases does '%s' go through?" % example_verb,
                "code": (
                    "an(entity(phase).where(phase.action_verb == '%s'))"
                    ".ordered_by(phase.order)"
                )
                % example_verb,
            }
        )
    pickup_knowledge = next(
        (entry for entry in kb.action_knowledge_entries if entry.name == "pickup"),
        None,
    )
    place_knowledge = next(
        (entry for entry in kb.action_knowledge_entries if entry.name == "place"),
        None,
    )
    if pickup_knowledge and place_knowledge:
        shared_facts = {goal.name for goal in pickup_knowledge.goals} & {
            precondition.name for precondition in place_knowledge.preconditions
        }
        if shared_facts:
            presets.append(
                {
                    "text": "what does 'pickup' leave true for 'place' to rely on?",
                    "code": (
                        "set_of(goal.name, precondition.name)"
                        ".where(goal.action_verb == 'pickup')"
                        ".where(precondition.action_verb == 'place')"
                        ".where(goal.name == precondition.name)"
                    ),
                }
            )
    example_entity = (
        kb.entity_knowledge_entries[0] if kb.entity_knowledge_entries else None
    )
    if example_entity:
        presets.append(
            {
                "text": "what can the %s be used for?" % example_entity.name,
                "code": (
                    "an(entity(affordance).where(affordance.entity_name == '%s'))"
                    % example_entity.name
                ),
            }
        )
    return presets


def _hypothesis_graph_presets(kb: KB) -> List[Dict[str, str]]:
    """
    Ready-made questions over the active pouring structure: its phases, the events
    changing its forces, the things taking part in it, how its phases follow each
    other, what must hold during it, what the robot expects to sense, and what can go
    wrong.

    Each answer names the fragments it involves and highlights them in the Hypothesis
    Graph tab, where their own structure is explored by expanding the node.

    There are none until an instruction has been generated, since every one of them
    would match nothing.
    """
    if hypothesis_knowledge.active() is None:
        return []
    return [
        {
            "text": "what are the different action phases involved in the action?",
            "code": "an(entity(pouring_phase))",
        },
        {
            "text": "what are the force dynamic events involved in the action?",
            "code": "an(entity(pouring_force_dynamic_event))",
        },
        {
            "text": "what is the source container the substance is poured from?",
            "code": (
                "an(entity(pouring_role)"
                ".where(HasType(pouring_role, pouring.SourceContainer)))"
            ),
        },
        {
            "text": "what is the destination the substance is poured into?",
            "code": (
                "an(entity(pouring_role)"
                ".where(HasType(pouring_role, pouring.TargetReceptacle)))"
            ),
        },
        {
            "text": "what substance is poured?",
            "code": (
                "an(entity(pouring_role)"
                ".where(HasType(pouring_role, pouring.PouredSubstance)))"
            ),
        },
        {
            "text": "which effector grasps and moves the source container?",
            "code": (
                "an(entity(pouring_role)"
                ".where(HasType(pouring_role, prehension.GraspingEffector)))"
            ),
        },
        {
            "text": "how do the phases follow each other, and which event triggers each step?",
            "code": (
                "set_of(pouring_transition.from_phase, pouring_transition.to_phase,"
                " pouring_transition.gated_by)"
            ),
        },
        {
            "text": "what conditions must hold during the action?",
            "code": (
                "set_of(pouring_condition_requirement.has_condition_type,"
                " pouring_condition_requirement.has_condition_mode)"
            ),
        },
        {
            "text": "what sensory events does the robot expect during the action?",
            "code": (
                "set_of(pouring_sensory_prediction.predicts_event,"
                " pouring_sensory_prediction.has_modality)"
            ),
        },
        {
            "text": "which failures can occur during the action?",
            "code": (
                "an(entity(pouring_event_type)"
                ".where(HasType(pouring_event_type, core.FailureEventType)))"
            ),
        },
    ]


def _scene_verb_presets(kb: KB) -> List[Dict[str, str]]:
    """
    Ready-made queries that connect the recorded episodes of this demo to their
    action verb's identity -- e.g. every ``transport_*`` episode side by side, not
    just the single representative episode the plain scene presets above use.
    """
    presets: List[Dict[str, str]] = []
    verb = _example_action_verb(kb)
    if not verb:
        return presets
    matching_episodes = [
        episode for episode in kb.episodes if episode.name.startswith(verb + "_")
    ]
    if len(matching_episodes) < 2:
        return presets
    presets.append(
        {
            "text": "every '%s' action in this demo" % verb,
            "code": "an(entity(ep).where(ep.name.startswith('%s_')))" % verb,
        }
    )
    presets.append(
        {
            "text": "what does each '%s' move, and for how long?" % verb,
            "code": (
                "set_of(ep.name, ep.picks, ep.duration_s)"
                ".where(ep.name.startswith('%s_'))"
            )
            % verb,
        }
    )
    return presets


if __name__ == "__main__":
    # smoke test: run every preset against the active scene
    configure_console_logging()
    for preset in get_presets():
        try:
            result = run_query(preset["code"])
            logging.info("OK   %-32s -> %d rows", preset["text"], result["count"])
        except Exception as error:
            logging.error(
                "FAIL %-32s -> %s: %s", preset["text"], type(error).__name__, error
            )
