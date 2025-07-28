from __future__ import annotations
from typing import TYPE_CHECKING

from fdtd_fun.constants import Field

if TYPE_CHECKING:
    from fdtd_fun.typing_ import Index
    from numpy import ndarray

from typing import Callable
from fdtd_fun.grid_object import GridObject


## Object
class Detector(GridObject):
    def __init__(self, name: str, toRead:dict[Field,bool]):
        # TODO: add temporal resolution?
        self.toRead:dict[Field,bool] = toRead
        self.E: ndarray | None = None  # shape starts with 3, so the first index selects the vector component
        self.H: ndarray | None = None
        self.J: ndarray | None = None
        self.rho: ndarray | None = None  # shape of the position array provided in _register_grid()
        super().__init__(name)

    def _validate_position(self, x: Index, y: Index, z: Index):
        pass

    def read(self):
        if self.toRead.get(Field.E):
            self.E = self._grid.E[:, self.x, self.y, self.z]
        if self.toRead.get(Field.J):
            self.J = self._grid.J[:, self.x, self.y, self.z]
        if self.toRead.get(Field.H):
            self.H = self._grid.B[:, self.x, self.y, self.z]
        if self.toRead.get(Field.rho):
            self.rho = self._grid.rho[self.x, self.y, self.z]
        # TODO: decide if we want to spend time copying from views
        '''
        #like this:
        if self.E.base is not None:
            self.E = self.E.copy()
        #etc..
        '''
