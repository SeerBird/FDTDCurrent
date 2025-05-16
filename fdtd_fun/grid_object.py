from __future__ import annotations
from typing import TYPE_CHECKING
from abc import abstractmethod

if TYPE_CHECKING:
    from .grid import Grid
    from .typing_ import Index


class GridObject:
    def __init__(self, name: str):
        self.name = name # name, unique per grid
        self._grid: Grid # the grid this grid object is registered with
        self.x: Index # x,y,z are the index positions that make up the grid subset
        # which this grid object is registered with
        self.y: Index
        self.z: Index

    def _register_grid(self, grid: Grid, x: Index, y: Index, z: Index):
        self._grid = grid
        self._validate_position(x, y, z)
        self.x = x
        self.y = y
        self.z = z

    def __getstate__(self):
        _dict = self.__dict__.copy()
        _dict.pop("_grid")
        return _dict

    @abstractmethod
    def _validate_position(self, x: Index, y: Index, z: Index):
        pass
