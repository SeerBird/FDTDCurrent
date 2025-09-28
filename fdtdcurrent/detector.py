from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy import ndarray
from enum import Enum
from fdtd_fun.grid_object import GridObject
from fdtd_fun.typing_ import Field, Key, Comp
import logging
import numpy as np

logger: logging.Logger = logging.getLogger(__name__)


class Detectable(Enum):  # stupid name
    # components of vector fields or scalar fields
    E = Field.E
    Ex = (Field.E, Comp.x)
    Ey = (Field.E, Comp.y)
    Ez = (Field.E, Comp.z)
    B = Field.B
    Bx = (Field.B, Comp.x)
    By = (Field.B, Comp.y)
    Bz = (Field.B, Comp.z)
    J = Field.J
    Jx = (Field.J, Comp.x)
    Jy = (Field.J, Comp.y)
    Jz = (Field.J, Comp.z)
    V = "V"


scalar_obss = [Detectable.V]
magnitudes = [Detectable.E, Detectable.B, Detectable.J]


class Detector(GridObject):
    def __init__(self, name: str, toRead: list[Detectable]):
        """
        A detector GridObject that allows to repeatedly access the field values in an arbitrarily shaped portion of
         the grid. Access Detector.values to read the field values corresponding to the detectables in toRead.
        :param name: yup.
        :param toRead: a list of detectables (scalar fields) to be read every time self.read() is called
        """
        self.toRead: list[Detectable] = toRead
        # noinspection PyTypeChecker
        self.values: list[ndarray] = [None] * len(toRead)
        super().__init__(name)

    def _validate_position(self, x: Key, y: Key, z: Key):
        if self.toRead.__contains__(Detectable.V):
            shape = self._grid[x, y, z].shape[1:]
            if len(shape) - shape.count(1) != 1:
                logger.warning("Can only read potential with 1D detectors. I think.")
                self.toRead.remove(Detectable.V)

    def read(self):
        """
        Read current grid state into the detector
        """
        grid_subset = self._grid[self.x,self.y,self.z]
        for i in range(len(self.toRead)):
            obs = self.toRead[i]
            if scalar_obss.__contains__(obs):
                if obs==Detectable.V:
                    # detector is assumed 1-D
                    E = grid_subset[Field.E.value]
                    E = np.moveaxis(E, 0, -1).reshape((-1, 3))
                    positions = np.moveaxis(self._grid._get_index(self.x, self.y, self.z), 0, -1).reshape((-1, 3)) * self._grid.ds
                    E = (E[1:] + E[:-1]) / 2
                    distances = positions[1:] - positions[:-1]
                    self.values[i] = np.cumsum(
                        -E[:, 0] * distances[:, 0] - E[:, 1] * distances[:, 1] - E[:, 2] * distances[:, 2])
            elif magnitudes.__contains__(obs):
                f = grid_subset[obs.value.value]
                self.values[i] = (f[0]**2+f[1]**2+f[2]**2)**0.5
            else:
                # noinspection PyUnresolvedReferences
                self.values[i]=grid_subset[obs.value[0].value,obs.value[1].value]
        # TODO: decide if we want to spend time copying from views
