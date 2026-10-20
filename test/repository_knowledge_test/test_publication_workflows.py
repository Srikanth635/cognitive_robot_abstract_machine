"""
Publication workflows keep proposed data away from checker execution.
"""

from __future__ import annotations

from pathlib import Path
import os
import shutil
import subprocess

import yaml
import pytest

# %% workflow trust boundaries


@pytest.mark.parametrize(
    ("fixture_filename", "returncode"),
    [("activation.env", 0), ("activation-injection.env", 1)],
)
def test_activation_file_is_validated_as_data(
    tmp_path: Path, fixture_filename: str, returncode: int
) -> None:
    """
    Owner-selected pins are copied literally and unknown keys cannot execute.
    """
    root = Path(__file__).parents[2]
    shell = (
        "C:/Program Files/Git/bin/bash.exe"
        if os.name == "nt"
        else (shutil.which("bash") or "/usr/bin/bash")
    )
    if not Path(shell).is_file():
        pytest.skip("Bash is required to exercise the workflow's data boundary.")
    workflow = yaml.load(
        (root / ".github/workflows/knowledge-trust.yml").read_text(),
        Loader=yaml.BaseLoader,
    )
    script = workflow["jobs"]["pins"]["steps"][-1]["run"]
    data = (Path(__file__).parent / "dataset/knowledge" / fixture_filename).read_text()
    configuration = tmp_path / "activation/repository_knowledge/policy/activation.env"
    configuration.parent.mkdir(parents=True)
    configuration.write_text(data, newline="\n")
    output = tmp_path / "outputs.env"
    environment = dict(os.environ, GITHUB_OUTPUT=output.as_posix())
    result = subprocess.run(
        [shell, "--noprofile", "--norc", "-euo", "pipefail", "-c", script],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == returncode, result.stderr
    assert (tmp_path / "injected").exists() is False
    if returncode == 0:
        assert output.read_text() == data


def test_hosted_consumption_uses_explicit_artifact_and_source() -> None:
    """
    Hosted verification checks immutable publication IDs against requested source.
    """
    root = Path(__file__).parents[2]
    workflow = yaml.load(
        (root / ".github/workflows/knowledge-verify.yml").read_text(),
        Loader=yaml.BaseLoader,
    )
    assert workflow["permissions"] == {"contents": "read", "actions": "read"}
    inputs = workflow["on"]["workflow_dispatch"]["inputs"]
    assert inputs["artifact_id"]["required"] == "true"
    assert inputs["source_commit"]["required"] == "true"
    job = workflow["jobs"]["verify"]
    assert "github.sha == needs.trust.outputs.checker_revision" in job["if"]
    run = next(
        step
        for step in job["steps"]
        if step.get("env", {}).get("OPERATION") == "consume"
    )
    assert (
        run["env"]["ARTIFACT_IDENTIFIER"]
        == "${{ inputs.artifact_id || needs.trust.outputs.artifact_id }}"
    )
    assert run["env"]["KNOWLEDGE_IDENTITY_TOKEN"] == "${{ github.token }}"
    runner = (root / "repository_knowledge/scripts/publication.sh").read_text()
    assert 'arguments+=(--artifact-id "$ARTIFACT_IDENTIFIER")' in runner


def test_runtime_build_uses_only_pinned_trusted_inputs() -> None:
    """
    Build a dependency runtime without allowing a proposal to select its build code.
    """
    root = Path(__file__).parents[2]
    workflow = yaml.load(
        (root / ".github/workflows/knowledge-runtime.yml").read_text(),
        Loader=yaml.BaseLoader,
    )
    job = workflow["jobs"]["runtime"]
    assert "github.sha == inputs.checker_revision" in job["if"]
    assert workflow["permissions"] == {"contents": "read", "packages": "write"}
    dockerfile = (root / "repository_knowledge/runtime/Dockerfile").read_text()
    assert "@sha256:" in next(
        line for line in dockerfile.splitlines() if line.startswith("FROM ")
    )
    assert "--require-hashes" in dockerfile
    assert "--no-deps" in dockerfile
    requirements = (root / "repository_knowledge/runtime/requirements.txt").read_text()
    packages = [
        line for line in requirements.splitlines() if line and not line.startswith("#")
    ]
    assert all("==" in line and "--hash=sha256:" in line for line in packages)
    assert all(
        len(step["uses"].split("@")[1]) == 40 for step in job["steps"] if "uses" in step
    )


def test_admission_uses_unprivileged_pinned_checker() -> None:
    """
    Pull requests cannot select their executable checker or receive credentials.
    """
    root = Path(__file__).parents[2]
    workflow = yaml.load(
        (root / ".github/workflows/knowledge-admission.yml").read_text(),
        Loader=yaml.BaseLoader,
    )
    assert set(workflow["on"]) == {"pull_request_target", "workflow_dispatch", "push"}
    assert workflow["permissions"] == {"contents": "read"}
    job = workflow["jobs"]["admission"]
    assert job["name"] == "knowledge-admission"
    checkouts = [
        step
        for step in job["steps"]
        if step.get("uses", "").startswith("actions/checkout@")
    ]
    assert all(step["with"]["persist-credentials"] == "false" for step in checkouts)
    assert checkouts[0]["with"]["ref"] == "${{ needs.trust.outputs.checker_revision }}"
    run = job["steps"][-1]
    assert run["env"]["OPERATION"] == "check"
    assert "KNOWLEDGE_IDENTITY_TOKEN" not in run["env"]
    assert (
        run["run"]
        == 'bash "$GITHUB_WORKSPACE/checker/repository_knowledge/scripts/publication.sh"\n'
    )


def test_publisher_requires_no_human_environment_approval() -> None:
    """
    The owner dispatches a pinned publisher with a read-only provider token.
    """
    root = Path(__file__).parents[2]
    workflow = yaml.load(
        (root / ".github/workflows/knowledge-publish.yml").read_text(),
        Loader=yaml.BaseLoader,
    )
    assert set(workflow["on"]) == {"workflow_dispatch", "push"}
    assert workflow["permissions"] == {"contents": "read", "actions": "read"}
    job = workflow["jobs"]["publish"]
    assert "environment" not in job
    assert (
        workflow["on"]["workflow_dispatch"]["inputs"]["source_commit"]["required"]
        == "true"
    )
    assert "github.sha == needs.trust.outputs.checker_revision" in job["if"]
    run = next(
        step
        for step in job["steps"]
        if step.get("env", {}).get("OPERATION") == "publish"
    )
    assert run["env"]["KNOWLEDGE_IDENTITY_TOKEN"] == "${{ github.token }}"
    upload = job["steps"][-1]
    assert upload["with"]["name"] == "cram-knowledge"
    assert upload["with"]["if-no-files-found"] == "error"
    for step in job["steps"]:
        if "uses" in step:
            assert len(step["uses"].split("@")[1]) == 40
