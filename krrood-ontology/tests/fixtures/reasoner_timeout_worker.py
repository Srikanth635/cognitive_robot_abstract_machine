"""Trusted fault fixture: embed a real JVM, record PID, then block."""

import os
import time
from pathlib import Path

import jpype

jpype.startJVM(jpype.getDefaultJVMPath(), "-Xmx128m")
Path(__file__).with_suffix(".pid").write_text(str(os.getpid()))
time.sleep(60)
