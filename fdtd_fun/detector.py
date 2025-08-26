from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fdtd_fun.typing_ import Key, Field
    from numpy import ndarray

from fdtd_fun.grid_object import GridObject
from fdtd_fun.typing_ import Field


## Object
class Detector(GridObject):
    def __init__(self, name: str, toRead:dict[Field,bool]):
        # TODO: add temporal resolution?
        self.toRead:dict[Field,bool] = toRead
        self.E: ndarray = None # shape starts with 3, so the first index selects the vector component
        self.B: ndarray = None
        self.J: ndarray = None
        self.rho: ndarray  = None # shape of the position array provided in _register_grid()
        super().__init__(name)

    def _validate_position(self, x: Key, y: Key, z: Key):
        pass

    def read(self):
        if self.toRead.get(Field.E):
            self.E = self._grid._get_value(Field.E, self.x, self.y, self.z) # (3,...)
        if self.toRead.get(Field.J):
            self.J = self._grid._get_value(Field.J, self.x, self.y, self.z)
        if self.toRead.get(Field.B):
            self.B = self._grid._get_value(Field.B, self.x, self.y, self.z)
        if self.toRead.get(Field.rho):
            self.rho = self._grid.rho[self.x, self.y, self.z]
        # TODO: decide if we want to spend time copying from views
        '''
        #like this:
        if self.E.base is not None:
            self.E = self.E.copy()
        #etc..
        '''
