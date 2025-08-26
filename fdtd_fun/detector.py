from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy import ndarray

from fdtd_fun.grid_object import GridObject
from fdtd_fun.typing_ import Field


## Object
class Detector(GridObject, ABC):
    def __init__(self, name: str, toRead: list[Field]):
        """
        A detector GridObject that allows to repeatedly access the field values in an arbitrarily shaped portion of
         the grid.
        :param name: yup.
        :param toRead: a list of fields to be read every time self.read() is called
        """
        self.toRead: list[Field] = toRead
        self.E: ndarray = None  # shape starts with 3, so the first index selects the vector component
        self.B: ndarray = None
        self.J: ndarray = None
        self.rho: ndarray = None  # shape of the position array provided in _register_grid()
        super().__init__(name)

    def read(self):
        """
        :return: An array of the shape (3, ...) where the latter part of the shape corresponds to the key used
         for assigning this detector to the grid.
        """
        if self.toRead.__contains__(Field.E):
            self.E = self._grid._get_value(Field.E, self.x, self.y, self.z)  # (3,...)
        if self.toRead.__contains__(Field.J):
            self.J = self._grid._get_value(Field.J, self.x, self.y, self.z)
        if self.toRead.__contains__(Field.B):
            self.B = self._grid._get_value(Field.B, self.x, self.y, self.z)
        # TODO: decide if we want to spend time copying from views
