"""
Change source between the two observations of a working-file snapshot.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from repository_knowledge.adapters.working_tree import WorkingCapture, WorkingTreeSource

# %% capture interleaving


@dataclass
class ChangingCapture(WorkingTreeSource):
    """
    Replace a selected file after the first capture observation.
    """

    replacement_path: Path | None = None
    """
    The new source bytes introduced during capture.
    """

    changed_path: str = ""
    """
    The repository-relative file changed between observations.
    """

    observation_count: int = 0
    """
    The number of completed observations.
    """

    def capture_files(self) -> WorkingCapture:
        """
        Change source after the first :meth:`WorkingTreeSource.capture_files`
        observation.
        """
        observation = super().capture_files()
        self.observation_count += 1
        if self.observation_count == 1 and self.replacement_path is not None:
            shutil.copyfile(
                self.replacement_path, self.repository_root / self.changed_path
            )
        return observation
