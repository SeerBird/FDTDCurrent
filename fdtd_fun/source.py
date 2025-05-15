from __future__ import annotations
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from fdtd_fun import Grid
    from fdtd_fun.typing_ import Index

import numpy as np
from .grid_object import GridObject


## Object
class Source(GridObject):
    function: Callable[[np.ndarray, float], np.ndarray[tuple[int, ...], np.dtype[float]]]

    def __init__(self, name: str,
                 function: Callable[[np.ndarray, float], np.ndarray[tuple[int, ...], np.dtype[float]]]):
        """
        Args:
            name: name of the object (will become available as attribute to the grid)
        """
        super().__init__(name)
        self.function = function

    def apply(self):
        pass

    def cancel(self):
        pass

    def _validate_position(self, x: Index, y: Index, z: Index):
        pass


    def __repr__(self):
        return f"{self.__class__.__name__}(name={repr(self.name)})"

    def __str__(self):
        s = "    " + repr(self) + "\n"
