from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fdtd_fun.grid import Grid
    from fdtd_fun.typing_ import Index

from .grid_object import GridObject


class Boundary(GridObject):

    def __init__(self, name: str):
        super().__init__(name)

    def _validate_position(self, x: Index, y: Index, z: Index):
        super()._validate_position(x, y, z)

    @abstractmethod
    def update_some_field(self):
        #TODO: implement PML/Periodic/Reflecting Boundary classes that update the fields we decide we'll have
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(name={repr(self.name)})"

    def __str__(self):
        s = "    " + repr(self) + "\n"
