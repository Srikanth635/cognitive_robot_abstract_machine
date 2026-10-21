"""
What the add-on's routes do: offering the designators a scene has, resolving one with
llm_reasoner, generating hypothesis knowledge, showing a run, and performing one.

The state the routes share, which cram_viz kept in module-level caches, is held here by
one object the server owns.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import StrEnum
from functools import cached_property

from typing_extensions import Any

from cramera_resolution import hypothesis_knowledge, paths
from cramera_resolution.exceptions import EmptyInstruction
from cramera_resolution.hypothesis_knowledge.generation import (
    HypothesisGenerationService,
    ModelHypothesisSource,
)
from cramera_resolution.live import LiveRunLauncher
from cramera_resolution.payload import PayloadKey
from cramera_resolution.query_source import QuerySource
from cramera_resolution.resolution.action_vocabulary import ActionDesignatorVocabulary
from cramera_resolution.resolution.designators import (
    DesignatorQuery,
    OfferedDesignator,
    PouringDesignator,
    SceneNamespace,
)
from cramera_resolution.resolution.perspective_choice import (
    OfferedGeometryAdjustment,
    OfferedPerspective,
    OfferedPhaseStructure,
)
from cramera_resolution.resolution.resolver import DesignatorResolver, ResolutionRun
from cramera_resolution.resolution.runs import ResolutionRuns, RunDirectory, RunFile
from cramera_resolution.resolution.scene_world import SceneWorlds
from cramera_resolution.scenes import SceneBundle
from cramera_resolution.views import ViewName, view_payload


class ResolveRequestKey(StrEnum):
    """
    The keys of a request to resolve a designator.
    """

    QUERY = "query"
    """
    The designator's match query, as the search bar holds it.
    """

    SCENE = "scene"
    """
    The scene to resolve it in.
    """

    PERSPECTIVES = "perspectives"
    """
    The perspectives on the action core to consult beside the phase structure, by their
    modules' file names; none when left out.
    """

    PHASE_STRUCTURE = "phase_structure"
    """
    Whether to consult the phase structure describing the action; consulted when left
    out.
    """

    ADJUST_GEOMETRY = "adjust_geometry"
    """
    Whether a pour that misses may open the pour geometry the query leaves out; it may
    when left out.
    """


class GenerateRequestKey(StrEnum):
    """
    The keys of a request to generate hypothesis knowledge.
    """

    INSTRUCTION = "instruction"
    """
    The instruction to generate knowledge for.
    """


class ExecuteRequestKey(StrEnum):
    """
    The keys of a request to perform a resolution run.
    """

    RUN = "run"
    """
    The name of the run to perform.
    """


@dataclass
class ResolutionService:
    """
    The operations behind the add-on's routes, over the state they share: the world of the
    scene last asked for, the resolver, the generation of knowledge and the launcher of
    runs.

    Nothing that needs credentials is built until an operation that needs it is asked
    for, so the viewer starts without them.
    """

    scene_worlds: SceneWorlds = field(default_factory=SceneWorlds)
    """
    The world of the scene last offered or resolved in, shared by the designator menu and
    the resolver, so a scene's world is built once.
    """

    launcher: LiveRunLauncher = field(default_factory=LiveRunLauncher)
    """
    Starts the performance of a run, stopping the one it started before.
    """

    @cached_property
    def generation(self) -> HypothesisGenerationService:
        """
        The shared generation of hypothesis knowledge, with the record store indexed once.
        """
        return HypothesisGenerationService(source=ModelHypothesisSource())

    @cached_property
    def resolver(self) -> DesignatorResolver:
        """
        The shared designator resolver.
        """
        return DesignatorResolver.for_viewer(ModelHypothesisSource(), self.scene_worlds)

    @property
    def runs(self) -> ResolutionRuns:
        """
        The runs kept, in the data directory.
        """
        return ResolutionRuns(directory=paths.resolution_runs_dir())

    def resolver_for(self, request: dict[str, Any]) -> DesignatorResolver:
        """
        :param request: A resolve request, which may name perspectives to consult.
        :return: The shared resolver, consulting the action core and the perspectives the
            request chooses, none when it chooses none, and the phase structure unless the
            request leaves it out.
        :raises UnknownPerspective: When the request names a perspective the knowledge
            component does not register.
        """
        perspectives = OfferedPerspective.named(
            request.get(ResolveRequestKey.PERSPECTIVES) or []
        )
        return replace(
            self.resolver,
            perspectives=perspectives,
            consults_phase_structure=bool(
                request.get(ResolveRequestKey.PHASE_STRUCTURE, True)
            ),
            adjusts_pour_geometry=bool(
                request.get(ResolveRequestKey.ADJUST_GEOMETRY, True)
            ),
        )

    def offered_designators(self, requested_scene: str | None) -> dict[str, Any]:
        """
        :param requested_scene: The scene's name, or ``None`` for the default.
        :return: The predefined designators the scene offers, each as its match query,
            available or stating what the scene lacks, with the choices of knowledge to
            consult.
        :raises NoSuchScene: When the scene is not held.
        """
        bundle = SceneBundle.of(requested_scene)
        names = SceneNamespace(
            world=self.scene_worlds.world_for(bundle.name, bundle.directory)
        ).names()
        return {
            PayloadKey.OK: True,
            PayloadKey.SCENE: bundle.name,
            PayloadKey.DESIGNATORS: [
                OfferedDesignator.in_scene(identifier, names).payload()
                for identifier in PouringDesignator
            ],
            PayloadKey.ACTION_VOCABULARY: [
                entry.payload()
                for vocabulary in ActionDesignatorVocabulary.every()
                for entry in vocabulary.entries()
            ],
            PayloadKey.PHASE_STRUCTURE: OfferedPhaseStructure().payload(),
            PayloadKey.GEOMETRY_ADJUSTMENT: OfferedGeometryAdjustment().payload(),
            PayloadKey.PERSPECTIVES: [
                perspective.payload() for perspective in OfferedPerspective.every()
            ],
        }

    def resolve(self, request: dict[str, Any]) -> dict[str, Any]:
        """
        Resolve the designator query a request holds in a scene with llm_reasoner, make the
        hypothesis it consulted the one the views show, and answer with the resolution.

        :param request: The query and the scene, and what to consult.
        :raises NoSuchScene: When the scene is not held.
        """
        bundle = SceneBundle.of(request.get(ResolveRequestKey.SCENE))
        designator = DesignatorQuery(
            source=QuerySource(text=request.get(ResolveRequestKey.QUERY) or "")
        )
        run = self.resolver_for(request).resolve(
            designator, bundle.name, bundle.directory
        )
        return self.shown(run, bundle)

    def open_run(self, name: str, requested_scene: str | None) -> dict[str, Any]:
        """
        :param name: The name of a resolution run kept earlier.
        :param requested_scene: The scene's name, or ``None`` for the default.
        :return: The answer a resolution of that run gave, shown as it was then.
        :raises UnknownRun: When no run of that name is kept.
        """
        return self.shown(self.resolver.reopen(name), SceneBundle.of(requested_scene))

    def shown(self, run: ResolutionRun, bundle: SceneBundle) -> dict[str, Any]:
        """
        Make the hypothesis a run's resolution consulted what the views show, and answer
        with the resolution and that hypothesis.

        :param run: The run.
        :param bundle: The scene the run was made in.
        """
        hypothesis_knowledge.activate_resolution(run.resolution, run.knowledge)
        return {
            PayloadKey.OK: True,
            PayloadKey.RESOLUTION: run.resolution.payload(),
            PayloadKey.HYPOTHESIS: view_payload(ViewName.HYPOTHESIS, bundle.objects()),
            PayloadKey.VIEWER: paths.viewer_url(run.resolution.run),
        }

    def generate_hypothesis(self, request: dict[str, Any]) -> dict[str, Any]:
        """
        Generate knowledge for an instruction and answer with the refreshed hypothesis
        view.

        :param request: The instruction.
        :raises EmptyInstruction: When the instruction is empty.
        """
        instruction = (request.get(GenerateRequestKey.INSTRUCTION) or "").strip()
        if not instruction:
            raise EmptyInstruction()
        self.generation.generate(instruction)
        payload = view_payload(ViewName.HYPOTHESIS, [])
        payload[PayloadKey.INSTRUCTION] = instruction
        return payload

    def execute_run(self, request: dict[str, Any]) -> dict[str, Any]:
        """
        Perform a resolution run in the simulation, in the background, restarting whatever
        was performed before.

        :param request: The run's name.
        :raises UnknownRun: When no run of that name is kept.
        """
        self.launcher.launch(self.runs.named(request.get(ExecuteRequestKey.RUN) or ""))
        return {PayloadKey.OK: True}

    def kept_runs(self) -> dict[str, Any]:
        """
        :return: A summary of every run kept, the newest first.
        """
        directory = self.runs.directory
        kept = (
            [
                RunDirectory(directory=path)
                for path in directory.iterdir()
                if (path / RunFile.RESOLUTION).is_file()
            ]
            if directory.is_dir()
            else []
        )
        newest_first = sorted(
            kept,
            key=lambda run: (run.directory / RunFile.RESOLUTION).stat().st_mtime,
            reverse=True,
        )
        return {
            PayloadKey.OK: True,
            PayloadKey.RUNS: [self.summary_of(run) for run in newest_first],
        }

    @staticmethod
    def summary_of(run: RunDirectory) -> dict[str, Any]:
        """
        :param run: A run kept.
        :return: What the list of runs shows of it.
        """
        resolution = run.resolution()
        kept_at = (run.directory / RunFile.RESOLUTION).stat().st_mtime
        return {
            PayloadKey.NAME: run.name,
            PayloadKey.SCENE: run.environment().scene,
            PayloadKey.DESIGNATOR: resolution.designator,
            PayloadKey.QUERY: QuerySource(text=resolution.query).on_one_line,
            PayloadKey.COST: resolution.cost,
            PayloadKey.TRIALS: len(resolution.trials),
            PayloadKey.AT: datetime.fromtimestamp(kept_at).isoformat(
                timespec="seconds"
            ),
        }

    def view(self, name: str, requested_scene: str | None) -> dict[str, Any]:
        """
        :param name: The view's name.
        :param requested_scene: The scene's name, or ``None`` for the default.
        :return: The view's payload.
        :raises NoSuchScene: When the scene is not held.
        """
        return view_payload(name, SceneBundle.of(requested_scene).objects())
