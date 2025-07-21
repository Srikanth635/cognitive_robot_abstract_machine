__version__ = "0.0.2"


import logging


logger = logging.Logger("semantic_world")
logger.setLevel(logging.INFO)

from .connections import *
from .views import *
from .robots import *
