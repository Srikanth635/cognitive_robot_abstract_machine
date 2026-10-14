
def _construction_failure(self, *args, **kwargs):
    raise RuntimeError("private fixture detail")


globals()["Entity"].__init__ = _construction_failure
