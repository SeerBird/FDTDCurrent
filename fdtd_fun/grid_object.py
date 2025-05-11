from __future__ import annotations
from typing import TYPE_CHECKING
from abc import abstractmethod
if TYPE_CHECKING:
    from .grid import Grid
    from .typing_ import Index


class GridObject:
    name:str
    grid:Grid
    x:Index
    y:Index
    z:Index
    def __init__(self, name:str):
        self.name = name

    def _register_grid(self, grid: Grid, x: Index, y: Index, z: Index):
        self.grid = grid
        self._validate_position(x,y,z)
        self.x = x
        self.y = y
        self.z = z

    @abstractmethod
    def _validate_position(self, x: Index, y: Index, z: Index):
        pass