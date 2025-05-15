from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fdtd_fun.grid import Grid
    from fdtd_fun.typing_ import Index
    from numpy import ndarray

from typing import Callable
from fdtd_fun.grid_object import GridObject


## Object
class Detector(GridObject):
    def __init__(self, name: str):
        # TODO: add parameters to control which fields this is taking and
        #  add temporal resolution
        # the fields measured by the conductor the last time the read() method was called
        self.E: ndarray  # shape starts with 3, so the first index selects the vector component
        self.B: ndarray
        self.J: ndarray
        self.rho: ndarray  # shape of the position array provided in _register_grid()
        super().__init__(name)

    def _validate_position(self, x: Index, y: Index, z: Index):
        pass

    def read(self):
        self.E = self._grid.E[:, self.x, self.y, self.z]
        self.J = self._grid.J[:, self.x, self.y, self.z]
        self.B = self._grid.H[:, self.x, self.y, self.z]
        self.rho = self._grid.rho[self.x, self.y, self.z]
        #TODO: decide if we want to spend time copying from views
        '''
        #like this:
        if self.E.base is not None:
            self.E = self.E.copy()
        #etc..
        '''
