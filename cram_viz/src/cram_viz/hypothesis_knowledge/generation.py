"""
Turning a natural-language instruction into displayed, queryable knowledge.

An instruction is minted into a task the microworld registry can profile, handed to
``llm_reasoner``'s caching generator, and the resulting record becomes the viewer's
active knowledge -- projected, overlaid, and exposed as EQL domains exactly like the
checked-in sample. An instruction the model cannot yet render admissibly falls back to
that same checked-in sample, so the viewer always has a clean, EQL-queryable structure
to show; see :meth:`HypothesisGenerationService.generate`.

Generation is cache-first: a repeated instruction is served from the store without a
model call, so re-running one from the viewer costs nothing.

.. note::
   The task is pinned to SOMA's *PouringInto* profile. The instruction text steers the
   model through ``action_label``, but choosing a different microworld from arbitrary
   English is a separate problem this does not attempt.
"""

from __future__ import annotations

import hashlib
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from llm_reasoner.clients import AnthropicClient
from llm_reasoner.clients.base import LanguageModelClient
from llm_reasoner.knowledge.generation import CachingGenerator, HypothesisGenerator
from llm_reasoner.knowledge.hypothesis.record import HypothesisRecord
from llm_reasoner.knowledge.hypothesis.store import FileStore
from llm_reasoner.knowledge.microworlds import flanagan
from llm_reasoner.knowledge.microworlds.registry import select_modules, task_reference
from llm_reasoner.knowledge.ontological.decorators import ConceptReference
from typing_extensions import Optional

from cram_viz import paths
from cram_viz.hypothesis_knowledge import (
    HypothesisKnowledge,
    activate,
    build_hypothesis_record,
)
from cram_viz.hypothesis_knowledge.pouring_structure import (
    build_pouring_phase_structure,
)

logger = logging.getLogger(__name__)

SOMA = "http://www.ease-crc.org/ont/SOMA.owl#"

#: the profiled task every generated instruction is placed under, so it inherits the
#: pouring microworld's vocabulary
POURING_TASK_ANCESTORS = frozenset(
    {SOMA + "Pouring", SOMA + "Actuating", SOMA + "PhysicalTask"}
)


#: how many times a rejected generation may be fed its own violations and retried
MAX_REPAIRS = 4


#: environment variable holding the key the generation client authenticates with
ANTHROPIC_API_KEY_NAME = "ANTHROPIC_API_KEY"


class HypothesisGenerationError(RuntimeError):
    """
    Raised when an instruction could not be turned into a usable hypothesis.
    """


class MissingCredentialsError(HypothesisGenerationError):
    """
    Raised when no API key is available to authenticate the generation client.
    """

    def __init__(self, variable_name: str, searched: Optional[Path]) -> None:
        location = f" or in {searched}" if searched else ""
        super().__init__(
            f"{variable_name} is not set{location}. "
            f"Generation needs it; export it or add it to a .env file, then retry."
        )


# %% credentials
def _find_environment_file() -> Optional[Path]:
    """
    The nearest ``.env``, searched from the working directory and then from this package
    upwards, so the viewer finds the workspace's file wherever it is started.
    """
    working_directory = Path.cwd().resolve()
    package_directory = Path(__file__).resolve()
    for directory in (
        working_directory,
        *working_directory.parents,
        *package_directory.parents,
    ):
        candidate = directory / ".env"
        if candidate.is_file():
            return candidate
    return None


def _load_environment_file(environment_file: Path) -> None:
    """
    Read ``KEY=value`` lines into the environment without overriding what is set.

    ``python-dotenv`` is used when available so behaviour matches the notebooks;
    otherwise the same simple format is parsed directly, since :mod:`cram_viz` keeps its
    own dependencies to the standard library.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        for line in environment_file.read_text().splitlines():
            entry = line.strip()
            if not entry or entry.startswith("#") or "=" not in entry:
                continue
            name, _, value = entry.partition("=")
            os.environ.setdefault(name.strip(), value.strip().strip("\"'"))
        return
    load_dotenv(environment_file, override=False)


# %% sources
class HypothesisSource(ABC):
    """
    Something that can produce a hypothesis record for an instruction.
    """

    @abstractmethod
    def record_for(self, instruction: str) -> HypothesisRecord:
        """
        The record describing what the instruction's action is generally like.
        """


@dataclass
class ModelHypothesisSource(HypothesisSource):
    """
    Produces records with a language model, consulting a store first.
    """

    store_directory: Optional[Path] = None
    """
    Where generated records are cached; :func:`~cram_viz.paths.hypothesis_store_dir`
    when unset.
    """

    client: Optional[LanguageModelClient] = None
    """
    The model client; an :class:`AnthropicClient` is built on demand when unset.
    """

    environment_file: Optional[Path] = None
    """
    The ``.env`` to read credentials from; searched for from the working directory
    upwards when unset.
    """

    _generator: Optional[CachingGenerator] = field(default=None, init=False)

    def load_credentials(self) -> None:
        """
        Make the API key visible to the client.

        The viewer is normally started from a shell that never sourced ``.env``, so the
        file the notebooks rely on is read here too. Values already exported win, and an
        injected client is assumed to carry its own credentials.

        :raises MissingCredentialsError: if no key is available.
        """
        if self.client is not None:
            return
        environment_file = self.environment_file or _find_environment_file()
        if environment_file is not None and environment_file.is_file():
            _load_environment_file(environment_file)
        if not os.environ.get(ANTHROPIC_API_KEY_NAME):
            raise MissingCredentialsError(ANTHROPIC_API_KEY_NAME, environment_file)

    @staticmethod
    def task_for(instruction: str) -> ConceptReference:
        """
        Mint a task for one instruction.

        The IRI is derived from the instruction so two different instructions get
        separate cache entries, while the inherited ancestors keep the registry
        selecting the pouring microworld.
        """
        digest = hashlib.sha256(instruction.strip().lower().encode()).hexdigest()[:16]
        return task_reference(
            f"urn:cram_viz:instruction:{digest}",
            POURING_TASK_ANCESTORS,
            label=instruction,
        )

    def record_for(self, instruction: str) -> HypothesisRecord:
        task = self.task_for(instruction)
        result = self._caching_generator().get(
            task, action_label=instruction, selection=select_modules(task)
        )
        if result.record.root is None:
            raise HypothesisGenerationError(
                f"Generation for {instruction!r} produced no usable structure "
                f"(status: {result.record.status})."
            )
        if not result.servable:
            # CachingGenerator.get deliberately returns a refuted record instead
            # of silently retrying (see its own docstring), so a refuted result
            # must be reported here rather than displayed as if it were valid.
            errors = "; ".join(result.record.validation.errors)
            raise HypothesisGenerationError(
                f"Generation for {instruction!r} was refuted "
                f"(status: {result.record.status}): {errors}"
            )
        return result.record

    def _caching_generator(self) -> CachingGenerator:
        """
        Build the generator on first use, so importing costs no credentials.
        """
        self.load_credentials()
        if self._generator is None:
            generator = HypothesisGenerator(
                client=self.client or AnthropicClient(),
                suite=flanagan.SUITE,
                max_repairs=MAX_REPAIRS,
            )
            self._generator = CachingGenerator(
                generator=generator,
                store=FileStore(
                    directory=self.store_directory or paths.hypothesis_store_dir()
                ),
            )
        return self._generator


# %% service
@dataclass
class HypothesisGenerationService:
    """
    Makes a generated hypothesis the knowledge the viewer shows and queries.
    """

    source: HypothesisSource
    """
    Where records come from.
    """

    def generate(self, instruction: str) -> HypothesisKnowledge:
        """
        Generate knowledge for an instruction and make it active.

        Falls back to the checked-in, hand-authored pouring sample when the source
        cannot produce an admissible structure, so the viewer always has a clean, EQL-
        queryable demo to show rather than an empty or broken one. The fallback is
        reported through :attr:`HypothesisKnowledge.generated` being ``False``, so a
        caller can still tell the two apart.
        """
        try:
            record = self.source.record_for(instruction)
            generated = True
        except HypothesisGenerationError as error:
            logger.warning(
                "generation for %r produced no admissible structure (%s); "
                "falling back to the checked-in pouring sample",
                instruction,
                error,
            )
            record = build_hypothesis_record(build_pouring_phase_structure())
            generated = False
        return activate(
            HypothesisKnowledge.from_structure(
                record.root,
                instruction=instruction,
                generated=generated,
                record=record,
            )
        )
