"""Check generated packages in a fresh interpreter before publishing them."""

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


class GeneratedVerificationError(ValueError):
    """A generated package failed isolated verification."""


@dataclass
class PackageVerifier:
    """Verify imports, descriptor registration and optional individual preflight."""

    timeout_seconds: int = 45

    def verify(
        self, directory: Path, package_name: str, preflight: bool
    ) -> dict[str, object]:
        worker = Path(__file__).with_name("verification_worker.py")
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-B",
                    str(worker),
                    str(directory),
                    package_name,
                    str(Path(__file__).parents[1]),
                    "true" if preflight else "false",
                ],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            raise GeneratedVerificationError(
                "Isolated verification timed out"
            ) from error
        if result.returncode:
            raise GeneratedVerificationError(
                f"Isolated verification failed:\n{result.stderr[-8000:]}"
            )
        return {
            "isolated_import": "passed",
            "individual_preflight": "passed" if preflight else "omitted",
            "timeout_seconds": self.timeout_seconds,
        }
