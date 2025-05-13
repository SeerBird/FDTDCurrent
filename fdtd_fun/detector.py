from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fdtd_fun.grid import Grid
    from fdtd_fun.typing_ import Index

from typing import Callable
from fdtd_fun.grid_object import GridObject


## Object
class Detector(GridObject):
    output: Callable[[object], None]  # figure out the shape of this

    def __init__(self, name: str,
                 output: Callable[[object], None]):
        #TODO: add parameters to control which fields this is taking and
        # add temporal resolution
        super().__init__(name)
        self.output = output

    def _validate_position(self, x: Index, y: Index, z: Index):
        pass

    def read(self):
        pass #

    def __repr__(self):
        return f"{self.__class__.__name__}(name={repr(self.name)})"

    def __str__(self):
        s = "    " + repr(self) + "\n"
