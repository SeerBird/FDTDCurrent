from __future__ import annotations
from typing import TYPE_CHECKING
from abc import abstractmethod, ABC

if TYPE_CHECKING:
    from .grid import Grid
    from .typing_ import Key


class GridObject(ABC):
    def __init__(self, name: str):
        self.name = name # name, unique per grid
        self._grid: Grid # the grid this grid object is registered with
        #TODO: potentially make GridObjects support multiple key assignments (e.g. grid[lalala] = a and then grid[brrr] = a)?
        self.x: Key # x,y,z are the index positions that make up the grid subset
        # which this grid object is registered with
        self.y: Key
        self.z: Key

    def _register_grid(self, grid: Grid, x: Key, y: Key, z: Key):
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
    def _validate_position(self, x: Key, y: Key, z: Key):
        pass #TODO: consider actually doing this
