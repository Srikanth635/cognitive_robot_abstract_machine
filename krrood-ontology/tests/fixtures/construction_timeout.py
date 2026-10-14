
def _construction_timeout(self, *args, **kwargs):
    import os
    import time
    from pathlib import Path

    (Path(__file__).parent / "timeout-marker.txt").write_text(str(os.getpid()))
    time.sleep(60)


globals()["Entity"].__init__ = _construction_timeout
