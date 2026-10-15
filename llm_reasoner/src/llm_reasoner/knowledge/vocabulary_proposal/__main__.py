"""
``python -m llm_reasoner.knowledge.vocabulary_proposal <perspective> --action <concept>
--frame <frame> --output <directory>``: ask a model for the vocabulary of a perspective,
and write the proposal as a module and a record of how it was made, for review.

It needs the linked ontologies the environment names (``CRAM_AICOR_ONTOLOGY``,
``DUL_ONTOLOGY``) and an Anthropic key; with owlready2 installed, HermiT checks every
answer against the action core, CRAM-AICOR and DUL.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from pathlib import Path

from typing_extensions import Any, Dict, List, Optional

from llm_reasoner.clients.anthropic_client import AnthropicClient
from llm_reasoner.clients.base import Exchange
from llm_reasoner.clients.usage import SONNET_5_5_PRICES
from llm_reasoner.knowledge.admission.hermit import HermitCheck
from llm_reasoner.knowledge.ontologies import ActionPerspectiveFile
from llm_reasoner.knowledge.ontologies.linked import OntologyFiles
from llm_reasoner.knowledge.vocabulary_proposal.catalogue import TermCatalogue
from llm_reasoner.knowledge.vocabulary_proposal.checks import Problem, ProposalChecker
from llm_reasoner.knowledge.vocabulary_proposal.perspective import Perspective
from llm_reasoner.knowledge.vocabulary_proposal.prompts import ProposalPrompt
from llm_reasoner.knowledge.vocabulary_proposal.proposer import (
    ProposalRejected,
    VocabularyProposer,
)
from llm_reasoner.observation.events import Component, ModelAnswered
from llm_reasoner.observation.observed import ObservedClient
from llm_reasoner.observation.sinks import MemorySink

logger = logging.getLogger(__spec__.name)
"""
This module's logger, named for the module even when ``python -m`` runs it.
"""

SONNET_5_5 = "claude-sonnet-5-5"
"""
The model asked unless the command line names another, whose prices are known.
"""

DEFAULT_MAX_TOKENS = 32000
"""
How many tokens an answer may have unless the command line says otherwise; a vocabulary
of a few dozen terms with their definitions needs some thousands.
"""

RECORD_SUFFIX = ".record.json"
"""
The suffix of the record written beside a proposed module.
"""

TURTLE_SUFFIX = ".ttl"
"""
The suffix of a proposed module.
"""


def arguments_of(argv: List[str]) -> argparse.Namespace:
    """
    :return: What the command line asks for.
    """
    parser = argparse.ArgumentParser(
        prog="python -m llm_reasoner.knowledge.vocabulary_proposal",
        description="Ask a model for the vocabulary of a perspective on an action.",
    )
    parser.add_argument(
        "perspective", choices=[perspective.value for perspective in Perspective]
    )
    parser.add_argument("--action", required=True, help="such as aicor:Pouring")
    parser.add_argument("--frame", required=True, help="such as core:PouringFrame")
    parser.add_argument("--output", required=True, type=Path, help="a directory")
    parser.add_argument("--model", default=SONNET_5_5)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--max-repairs", type=int, default=2)
    return parser.parse_args(argv)


def record_of(
    arguments: argparse.Namespace,
    exchanges: List[Exchange],
    problems: List[Problem],
    sink: MemorySink,
) -> Dict[str, Any]:
    """
    :return: How the proposal was made: what was asked, every exchange, the problems the
        last answer had, and what the answers cost.
    """
    answered = [event for event in sink.events if isinstance(event, ModelAnswered)]
    costs = [event.cost for event in answered if event.cost is not None]
    return {
        "perspective": arguments.perspective,
        "action": arguments.action,
        "frame": arguments.frame,
        "model": arguments.model,
        "accepted": not problems,
        "problems": [problem.to_json() for problem in problems],
        "cost": sum(costs) if costs else None,
        "usage": [
            dataclasses.asdict(event.usage)
            for event in answered
            if event.usage is not None
        ],
        "exchanges": [dataclasses.asdict(exchange) for exchange in exchanges],
    }


def main(argv: Optional[List[str]] = None) -> None:
    """
    Ask for the vocabulary, and write the proposed module and its record.
    """
    arguments = arguments_of(sys.argv[1:] if argv is None else argv)
    linked = OntologyFiles.linked_from_environment()
    if linked is None:
        raise SystemExit(
            "set CRAM_AICOR_ONTOLOGY and DUL_ONTOLOGY to the ontology files"
        )
    perspective = Perspective(arguments.perspective)
    sink = MemorySink()
    client = ObservedClient.of(
        AnthropicClient(model=arguments.model, max_tokens=arguments.max_tokens),
        sink,
        Component.KNOWLEDGE,
        prices=SONNET_5_5_PRICES if arguments.model == SONNET_5_5 else None,
    )
    catalogue = TermCatalogue.from_files(
        core=ActionPerspectiveFile.ACTION_CORE.path, linked=linked.paths
    )
    consistency = (
        HermitCheck(
            background_files=[ActionPerspectiveFile.ACTION_CORE.path, *linked.paths]
        )
        if HermitCheck.shipped_directory() is not None
        else None
    )
    proposer = VocabularyProposer(
        client=client,
        prompt=ProposalPrompt(catalogue=catalogue),
        checker=ProposalChecker(
            catalogue=catalogue, perspective=perspective, consistency=consistency
        ),
        max_repairs=arguments.max_repairs,
    )
    arguments.output.mkdir(parents=True, exist_ok=True)
    record_path = arguments.output / (perspective.value + RECORD_SUFFIX)
    try:
        outcome = proposer.propose(perspective, arguments.action, arguments.frame)
    except ProposalRejected as rejected:
        record_path.write_text(
            json.dumps(
                record_of(arguments, rejected.exchanges, rejected.problems, sink),
                indent=2,
            )
        )
        logger.error("%s; the record is in %s", rejected, record_path)
        raise SystemExit(1) from rejected
    (arguments.output / (perspective.value + TURTLE_SUFFIX)).write_text(outcome.turtle)
    record = record_of(arguments, outcome.exchanges, [], sink)
    record_path.write_text(json.dumps(record, indent=2))
    logger.info(
        "%d classes, %d properties and %d value partitions proposed in %d answers "
        "(HermiT %s), costing %s; written to %s",
        len(outcome.proposal.classes),
        len(outcome.proposal.properties),
        len(outcome.proposal.value_partitions),
        len(outcome.exchanges),
        "checked" if consistency is not None else "not available",
        "unknown" if record["cost"] is None else f"${record['cost']:.4f}",
        arguments.output,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
