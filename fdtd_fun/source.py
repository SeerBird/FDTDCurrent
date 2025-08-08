from __future__ import annotations
from typing import TYPE_CHECKING, Callable


if TYPE_CHECKING:
    from fdtd_fun import Grid
    from .grid import State
    from fdtd_fun.typing_ import Key

import numpy as np
from .grid_object import GridObject


## Object
class Source(GridObject):

    def __init__(self, name: str,
                 function: Callable[[np.ndarray, float], np.ndarray]):
        """

        :param name: yeah.
        :param function: function that, given a cartesian position ndarray, returns
         an emf vector ndarray of the same shape
        """
        super().__init__(name)
        self.function = function
        self.positions: np.ndarray | None = None

    def apply(self):
        self._grid.emf[:,self.x,self.y,self.z] = self.function(self.positions, self._grid.time())


    def _register_grid(self, grid: Grid, x: Key, y: Key, z: Key):
        super()._register_grid(grid, x, y, z)
        self.positions = self._grid[self.x, self.y, self.z] * self._grid.ds

    def _validate_position(self, x: Key, y: Key, z: Key):
        pass

    def __getstate__(self):
        _dict = super().__getstate__()
        _dict.pop("function")
        return _dict
