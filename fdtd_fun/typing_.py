from numpy import ndarray
Vector3 = tuple[int, int, int] | tuple[float, float, float]
Index = slice | ndarray[tuple[int, ...], int]|int

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

