"""
FDTDCurrent
=====

Provides
  1. An fdtd simulation of the electromagnetic field and current density over a cuboid grid
  2. A matplotlib visualization helper for the simulation

>>>> say something about the readme I think?

Available subpackages
---------------------
visualization - visualization helpers for the simulation
"""
# region set up logging
from . import _mylogging
from enum import Enum


class LoggingLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARN"
    ERROR = "ERROR"


logger = _mylogging.setupLogging(__name__)


def configureLogging(loggingLevel: LoggingLevel):  # TODO: is it better to have an enum or use .logging stuff?
    logger.setLevel(loggingLevel.value)


# endregion
from .grid import Grid, Side, BoundaryType
from .detector import Detector
from .conductor import Conductor
from .source import Source
from .typing_ import Field, Comp
from . import constants
