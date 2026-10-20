"""
Inspect optional repository knowledge through its selected interpreter.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from threading import Event, Timer

# %% external process contract


class KnowledgeSetting(StrEnum):
    """
    Optional knowledge settings supplied by the session operator.
    """

    INTERPRETER = "CRAM_KNOWLEDGE_PYTHON"
    DATABASE = "CRAM_KNOWLEDGE_DATABASE"


class ProcessArgument(StrEnum):
    """
    The read-only report executable and its source selectors.
    """

    MODULE_FLAG = "-m"
    MODULE = "repository_knowledge.interfaces.session"
    REPOSITORY = "--repository"
    DATABASE = "--database"
    REVISION = "--revision"
    CHECKOUT = "HEAD"


class ReportField(StrEnum):
    """
    Fields required from the independently installed service.
    """

    TYPE = "__json_type__"
    REPORT = "repository_knowledge.clients.SessionReport"
    VERSION = "protocol_version"
    SUMMARY = "summary"
    ENTRYPOINT = "wiki_entrypoint"


class StartupState(StrEnum):
    """
    The optional inspection outcome visible to the session.
    """

    DISABLED = "disabled"
    REPORTED = "reported"
    UNAVAILABLE = "unavailable"
    TIMED_OUT = "timed-out"


@dataclass(frozen=True)
class StartupReport:
    """
    Knowledge guidance that cannot stop ordinary session startup.
    """

    state: StartupState
    """
    Whether the optional service returned usable startup guidance.
    """

    summary: str
    """
    The source status or reason ordinary source navigation is needed.
    """

    @classmethod
    def from_json(
        cls, payload: dict[str, object], protocol_version: int = 1
    ) -> StartupReport:
        """
        Accept only the supported external report shape and version.
        """
        summary = payload.get(ReportField.SUMMARY)
        if (
            payload.get(ReportField.TYPE) != ReportField.REPORT
            or type(payload.get(ReportField.VERSION)) is not int
            or payload.get(ReportField.VERSION) != protocol_version
            or not isinstance(summary, str)
            or not isinstance(payload.get(ReportField.ENTRYPOINT), str)
        ):
            return cls(
                StartupState.UNAVAILABLE,
                "Unsupported knowledge report; use source navigation.",
            )
        return cls(StartupState.REPORTED, summary)


# %% deadline-bound subprocess inspection


@dataclass
class KnowledgeStartup:
    """
    Request a small report without importing the CRAM package into basstler.
    """

    repository_root: Path
    """
    The checkout whose source status is requested.
    """

    interpreter: Path | None = None
    """
    The explicitly selected interpreter containing repository knowledge.
    """

    database_path: Path | None = None
    """
    An optional selected committed-source cache.
    """

    timeout_seconds: float = 5.0
    """
    The maximum time allowed for optional inspection.
    """

    maximum_output_bytes: int = 16384
    """
    The maximum response read into the basstler process.
    """

    def inspect(self) -> StartupReport:
        """
        Observe source freshness with a deadline and bounded response reading.
        """
        if self.interpreter is None:
            return StartupReport(
                StartupState.DISABLED,
                "Knowledge inspection disabled; use source navigation.",
            )
        if (
            not self.interpreter.is_file()
            or self.timeout_seconds <= 0
            or self.maximum_output_bytes < 1
        ):
            return StartupReport(
                StartupState.UNAVAILABLE,
                "Knowledge interpreter or inspection budget unavailable; use source navigation.",
            )
        arguments = [
            str(self.interpreter),
            ProcessArgument.MODULE_FLAG,
            ProcessArgument.MODULE,
            ProcessArgument.REPOSITORY,
            str(self.repository_root),
            ProcessArgument.REVISION,
            ProcessArgument.CHECKOUT,
        ]
        if self.database_path is not None:
            arguments.extend([ProcessArgument.DATABASE, str(self.database_path)])
        timed_out = Event()
        with tempfile.TemporaryFile() as output:
            process = subprocess.Popen(
                arguments,
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.DEVNULL,
                cwd=self.repository_root,
            )

            def expire() -> None:
                """
                Stop an optional child that exceeded its startup allowance.
                """
                timed_out.set()
                process.kill()

            deadline = Timer(self.timeout_seconds, expire)
            deadline.daemon = True
            deadline.start()
            returncode = process.wait()
            deadline.cancel()
            if timed_out.is_set():
                return StartupReport(
                    StartupState.TIMED_OUT,
                    "Knowledge inspection timed out; use source navigation.",
                )
            if returncode != 0 or output.tell() > self.maximum_output_bytes:
                return StartupReport(
                    StartupState.UNAVAILABLE,
                    "Knowledge service unavailable; use source navigation.",
                )
            output.seek(0)
            payload = json.loads(output.read(self.maximum_output_bytes).decode("utf-8"))
        if not isinstance(payload, dict):
            return StartupReport(
                StartupState.UNAVAILABLE,
                "Unsupported knowledge response; use source navigation.",
            )
        return StartupReport.from_json(payload)


# %% session hook entry


def main() -> int:
    """
    Print optional guidance using the operator-selected CRAM interpreter.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(ProcessArgument.REPOSITORY, type=Path, required=True)
    arguments = parser.parse_args()
    interpreter = os.environ.get(KnowledgeSetting.INTERPRETER)
    database = os.environ.get(KnowledgeSetting.DATABASE)
    result = KnowledgeStartup(
        arguments.repository,
        Path(interpreter) if interpreter else None,
        Path(database) if database else None,
    ).inspect()
    print(f"repository knowledge: {result.summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
