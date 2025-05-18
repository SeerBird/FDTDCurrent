from __future__ import annotations
from typing import TYPE_CHECKING, Callable


if TYPE_CHECKING:
    from fdtd_fun import Grid
    from .grid import State
    from fdtd_fun.typing_ import Index

import numpy as np
from .grid_object import GridObject


## Object
class Source(GridObject):

    def __init__(self, name: str,
                 function: Callable[[np.ndarray, float], State]):
        """

        :param name:
        :param function:
        """
        super().__init__(name)
        self.function = function
        self.lastState: State | None = None
        self.positions: np.ndarray | None = None

    def apply(self):
        self.lastState = self.function(self.positions, self._grid.time())
        if self.lastState.E is not None:
            self._grid.E[self.x,self.y,self.z]+=self.lastState.E
        if self.lastState.H is not None:
            self._grid.H[self.x,self.y,self.z]+=self.lastState.H
        if self.lastState.J is not None:
            self._grid.J[self.x,self.y,self.z]+=self.lastState.J
        if self.lastState.rho is not None:
            self._grid.rho[self.x,self.y,self.z]+=self.lastState.rho

    def cancel(self):
        if self.lastState.E is not None:
            self._grid.E[self.x, self.y, self.z] -= self.lastState.E
        if self.lastState.H is not None:
            self._grid.H[self.x, self.y, self.z] -= self.lastState.H
        if self.lastState.J is not None:
            self._grid.J[self.x, self.y, self.z] -= self.lastState.J
        if self.lastState.rho is not None:
            self._grid.rho[self.x, self.y, self.z] -= self.lastState.rho

    def _register_grid(self, grid: Grid, x: Index, y: Index, z: Index):
        super()._register_grid(grid, x, y, z)
        self.positions = self._grid.positions(self.x, self.y, self.z)

    def _validate_position(self, x: Index, y: Index, z: Index):
        pass
