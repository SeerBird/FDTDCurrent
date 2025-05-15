from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fdtd_fun import Grid
    from fdtd_fun.typing_ import Index

from .grid_object import GridObject


class Material(GridObject):
    def __init__(self, name: str):
        super().__init__(name)

    def _validate_position(self, x: Index, y: Index, z: Index):
        pass

    def _update_J_and_rho(self):
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(name={repr(self.name)})"

    def __str__(self):
        s = "    " + repr(self) + "\n"
