from numpy import ndarray
from enum import Enum
Vector3 = tuple[int, int, int] | tuple[float, float, float]
Key = slice | ndarray[tuple[int, ...], int] | int

class SliceAddInt: # just in case
    def __init__(self, adding:int):
        self.adding:int = adding
    def __add__(self, other):
        if not isinstance(other, slice):
            return self.adding + other
        else:
            start = None if not isinstance(other.start, int) else other.start + self.adding
            stop = None if not isinstance(other.stop, int) else other.stop + self.adding
            return slice(start, stop, other.step)

class Field(Enum):
    E = 0
    B = 1
    J = 2
    # only use these in detectors? find a way to make this nicer perhaps
    rho = -1
    V = -2

class Comp(Enum):
    x = 0
    y = 1
    z = 2
    abs = -1

