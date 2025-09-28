"""
FDTD_FUN
=====

Provides
  1. An fdtd simulation of the electromagnetic field and current density over a cuboid grid
  2. A matplotlib visualization helper for the simulation

>>>> say something about the readme I think? also obviously please rename the package

Available subpackages
---------------------
visualization - visualization helpers for the simulation
"""
# region set up logging
import logging
import sys

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")
class CustomFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(levelname)s:%(name)s: %(message)s"
    #format = "%(levelname)s:%(asctime)s:%(name)s: %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

ch = logging.StreamHandler(stream=sys.stdout)
ch.setLevel(logging.DEBUG)
ch.setFormatter(CustomFormatter())
logger.addHandler(ch)
logger.propagate = False
#endregion

from . import mylogging
from .grid import Grid
from .detector import Detector
from .conductor import Conductor
from .source import Source
from .typing_ import Field, Comp