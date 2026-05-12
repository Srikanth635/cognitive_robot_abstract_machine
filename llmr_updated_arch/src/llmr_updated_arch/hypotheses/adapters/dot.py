"""DOT and rendered graph export helpers for sg_model presentation graphs."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

from llmr_updated_arch.hypotheses.adapters.pydigraph import DerivedRelation, to_pydigraph
from llmr_updated_arch.hypotheses.entities.base import ClaimHypothesis, EvidenceHypothesis, Hypothesis
from llmr_updated_arch.hypotheses.entities.common import (
    Action,
    GroundingEvidence,
    Instruction,
    ReasonerRun,
    SlotEvidence,
)
from llmr_updated_arch.hypotheses.entities.flanagan import (
    FailureModeClaim,
    ForceDynamicEvent,
    GoalConditionClaim,
    PhaseClaim,
    PlanClaim,
    PreconditionClaim,
    RecoveryStrategyClaim,
)
from llmr_updated_arch.hypotheses.entities.framenet import FrameClaim, RoleClaim
from llmr_updated_arch.hypotheses.graph import HypothesisGraph
from llmr_updated_arch.hypotheses.meta import ClaimStatus, GroundingState


def to_dot(
    repository: HypothesisGraph,
    *,
    rankdir: str = "LR",
) -> str:
    """Return a DOT representation of the derived sg_model graph."""

    graph = to_pydigraph(repository)
    graph_attrs = {
        "rankdir": rankdir,
        "nodesep": "0.45",
        "ranksep": "0.7",
        "splines": "spline",
    }
    return _render_dot(graph, graph_attrs)


def _render_dot(graph: object, graph_attrs: dict[str, str]) -> str:
    """Render DOT directly instead of relying on rustworkx's callback wrapper."""

    lines = ["digraph {"]
    for key, value in graph_attrs.items():
        lines.append(f"  {key}={_dot_quote(value)};")

    for node_index in graph.node_indexes():
        node = graph.get_node_data(node_index)
        attrs = _format_attrs(_node_attrs(node))
        lines.append(f"  n{node_index} [{attrs}];")

    for source_index, target_index, edge in graph.weighted_edge_list():
        attrs = _format_attrs(_edge_attrs(edge))
        lines.append(f"  n{source_index} -> n{target_index} [{attrs}];")

    lines.append("}")
    return "\n".join(lines)


def _format_attrs(attrs: dict[str, str]) -> str:
    return ", ".join(f"{key}={_dot_quote(value)}" for key, value in attrs.items())


def _dot_quote(value: object) -> str:
    text = str(value)
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1]
    text = text.replace('"', r"\"").replace("\n", r"\n")
    return f'"{text}"'


def write_dot(
    repository: HypothesisGraph,
    filepath: str | Path,
    *,
    rankdir: str = "LR",
) -> Path:
    """Write the derived graph as a raw DOT file and return the path."""

    path = Path(filepath)
    if path.suffix != ".dot":
        path = path.with_suffix(".dot")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_dot(repository, rankdir=rankdir), encoding="utf-8")
    return path


def render_graph(
    repository: HypothesisGraph,
    filepath: str | Path,
    *,
    format_: str = "svg",
    rankdir: str = "LR",
) -> Path:
    """Render the derived graph for presentation use.

    If Graphviz/pydot rendering is unavailable, this falls back to writing a
    ``.dot`` file and returns that path instead.
    """

    path = Path(filepath)
    if path.suffix != f".{format_}":
        path = path.with_suffix(f".{format_}")
    path.parent.mkdir(parents=True, exist_ok=True)
    dot_str = to_dot(repository, rankdir=rankdir)

    try:
        import pydot
    except ImportError:
        return write_dot(repository, path.with_suffix(".dot"), rankdir=rankdir)

    try:
        dot_graph = pydot.graph_from_dot_data(dot_str)[0]
        dot_graph.write(path, format=format_)
        return path
    except Exception:
        dot_path = write_dot(repository, path.with_suffix(".dot"), rankdir=rankdir)
        dot_binary = shutil.which("dot")
        if dot_binary is None:
            return dot_path
        try:
            subprocess.run(
                [dot_binary, f"-T{format_}", str(dot_path), "-o", str(path)],
                check=True,
            )
            return path
        except Exception:
            return dot_path


def _node_attrs(node: Hypothesis) -> dict[str, str]:
    return {
        "label": _node_label(node),
        "shape": _node_shape(node),
        "style": '"filled,rounded"',
        "color": _node_border_color(node),
        "fillcolor": _node_fillcolor(node),
        "fontname": "Helvetica",
        "fontsize": "11",
        "penwidth": "1.3",
    }


def _edge_attrs(edge: DerivedRelation) -> dict[str, str]:
    return {
        "label": edge.label,
        "color": edge.color,
        "style": edge.style,
        "fontname": "Helvetica",
        "fontsize": "10",
    }


def _node_label(node: Hypothesis) -> str:
    if isinstance(node, Instruction):
        return f"Instruction\\n{_snippet(node.text, 44)}"
    if isinstance(node, Action):
        return f"Action\\n{node.action_type}"
    if isinstance(node, ReasonerRun):
        run = node.meta.short_run_id or node.run_id[:8]
        return f"Run\\n{node.reasoner_name}\\n{run}"
    if isinstance(node, FrameClaim):
        return f"Frame\\n{node.frame}\\n{node.lexical_unit}"
    if isinstance(node, RoleClaim):
        return f"Role: {node.role_name}\\n{_snippet(node.filler_text, 34)}"
    if isinstance(node, PlanClaim):
        return f"Plan\\n{node.action_type}\\n{node.phase_count} phases"
    if isinstance(node, ForceDynamicEvent):
        agent_part = node.agent or "—"
        return f"FDE [{node.role}]\\n{node.event_type}\\nagent: {agent_part}\\nobject: {_snippet(node.object_ref, 24)}"
    if isinstance(node, PhaseClaim):
        return f"[{node.phase_index}] {node.phase_name}\\n-> {_snippet(node.target_object, 28)}"
    if isinstance(node, PreconditionClaim):
        obj = f", {_snippet(node.object_ref, 18)}" if node.object_ref else ""
        subj = node.subject or "?"
        return f"Pre: {node.predicate_name}\\n({subj}{obj})\\n== {node.expected_value}"
    if isinstance(node, GoalConditionClaim):
        obj = f", {_snippet(node.object_ref, 18)}" if node.object_ref else ""
        subj = node.subject or "?"
        return f"Goal: {node.predicate_name}\\n({subj}{obj})\\n== {node.expected_value}"
    if isinstance(node, FailureModeClaim):
        return f"Failure\\n{node.name}\\n{_snippet(node.value_text or '', 28)}"
    if isinstance(node, RecoveryStrategyClaim):
        return f"Recovery\\n{node.name}\\n{_snippet(node.value_text or '', 28)}"
    if isinstance(node, SlotEvidence):
        return f"Slot\\n{node.slot_name}: {_snippet(node.value_repr, 30)}"
    if isinstance(node, GroundingEvidence):
        return f"Grounded\\n{_snippet(node.query_text, 28)}\\n{node.symbol_type}"
    return f"{type(node).__name__}\\n{node.short_id}"


def _node_shape(node: Hypothesis) -> str:
    if isinstance(node, (Instruction, Action, ReasonerRun)):
        return "box"
    if isinstance(node, EvidenceHypothesis):
        return "note"
    if isinstance(node, RoleClaim):
        return "ellipse"
    if isinstance(node, ForceDynamicEvent):
        return "diamond"
    if isinstance(node, (FrameClaim, PlanClaim)):
        return "box"
    if isinstance(node, PhaseClaim):
        return "box"
    if isinstance(
        node,
        (
            PreconditionClaim,
            GoalConditionClaim,
            FailureModeClaim,
            RecoveryStrategyClaim,
        ),
    ):
        return "ellipse"
    if isinstance(node, ClaimHypothesis):
        return "ellipse"
    return "box"


def _node_fillcolor(node: Hypothesis) -> str:
    if node.meta.grounding == GroundingState.SYMBOL_GROUNDED:
        return "lightgreen"
    if node.meta.grounding == GroundingState.SLOT_ALIGNED:
        return "khaki"
    if node.meta.status == ClaimStatus.SUPPORTED:
        return "lightblue"
    if node.meta.status == ClaimStatus.REFUTED:
        return "lightcoral"
    if node.meta.status == ClaimStatus.SUPERSEDED:
        return "gainsboro"
    if isinstance(node, ForceDynamicEvent):
        return "plum1"
    if isinstance(node, (Instruction, Action, ReasonerRun)):
        return "aliceblue"
    if isinstance(node, EvidenceHypothesis):
        return "cornsilk"
    return "white"


def _node_border_color(node: Hypothesis) -> str:
    if isinstance(node, ForceDynamicEvent):
        return "darkorchid4"
    if isinstance(node, EvidenceHypothesis):
        return "darkgoldenrod"
    if isinstance(node, (Instruction, Action, ReasonerRun)):
        return "steelblue4"
    return "black"


def _snippet(text: str, limit: int) -> str:
    compact = " ".join(text.split()).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."
