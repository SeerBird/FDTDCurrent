from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy import ndarray

from fdtd_fun.grid_object import GridObject
from fdtd_fun.typing_ import Field, Key
import logging
import numpy as np

logger: logging.Logger = logging.getLogger(__name__)

## Object
class Detector(GridObject):
    def __init__(self, name: str, toRead: list[Field]):
        """
        A detector GridObject that allows to repeatedly access the field values in an arbitrarily shaped portion of
         the grid.
        :param name: yup.
        :param toRead: a list of fields to be read every time self.read() is called
        """
        self._toRead: list[Field] = toRead
        self.E: ndarray = None  # shape starts with 3, so the first index selects the vector component
        self.B: ndarray = None
        self.J: ndarray = None
        self.V: ndarray = None
        self.rho: ndarray = None  # shape of the position array provided in _register_grid()
        super().__init__(name)

    def _validate_position(self, x: Key, y: Key, z: Key):
        if self._toRead.__contains__(Field.V):
            shape = self._grid[x,y,z].shape[1:]
            if len(shape)-shape.count(1)!=1:
                logger.warning("Can only read potential with 1D detectors. I think.")
                self._toRead.remove(Field.V)

    def read(self):
        """
        Each field is array of the shape (3, ...) where the latter part of the shape corresponds to the key used
         for assigning this detector to the grid.
        """
        if self._toRead.__contains__(Field.E):
            self.E = self._grid._get_value(Field.E, self.x, self.y, self.z)  # (3,...)
        if self._toRead.__contains__(Field.J):
            self.J = self._grid._get_value(Field.J, self.x, self.y, self.z)
        if self._toRead.__contains__(Field.B):
            self.B = self._grid._get_value(Field.B, self.x, self.y, self.z)
        if self._toRead.__contains__(Field.V):
            if self.E is None:
                E = self._grid._get_value(Field.E, self.x, self.y, self.z)
            else:
                E = self.E
            E = np.moveaxis(E,0,-1).reshape((-1,3))
            positions = np.moveaxis(self._grid[self.x,self.y,self.z],0,-1).reshape((-1,3))*self._grid.ds
            E = (E[1:] + E[:-1]) / 2
            distances = positions[1:]-positions[:-1]
            self.V = np.cumsum(E[:,0]*distances[:,0]+E[:,1]*distances[:,1]+E[:,2]*distances[:,2])

        # TODO: decide if we want to spend time copying from views
