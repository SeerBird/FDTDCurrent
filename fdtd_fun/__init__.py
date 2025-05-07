from typing import Callable

import numpy as np
from numpy import ndarray as array


# region wrappers
class State:
    D: array
    P: array
    H: array
    M: array
    rho: array

    def __init__(self):
        pass  # yeah just shove it there


class Material:
    region: np.ndarray  # I don't remember how the index array is typed


# endregion

class Grid:
    """DOCS"""
    indexing: array
    E: array

    def __init__(self, xwidth: float, ywidth: float, zwidth: float, ds: float, ):  # add boundaries
        """ """
        if xwidth < 0 or ywidth < 0 or zwidth < 0:
            raise Exception("I'm in hell actually")
        shape = (int(xwidth // ds + 1), int(ywidth // ds + 1), int(zwidth // ds + 1))
        x = np.arange(shape[0])
        y = np.arange(shape[1])
        z = np.arange(shape[2])
        self.indexing = np.asarray(np.meshgrid(x, y, z))
        self.E = np.zeros(shape)

        pass

    def add_material(self, pattern: Callable[[array], array[tuple[int, ...], np.dtype[bool]]],
                     free_charge_density: float,
                     spec_charge: float):  # probably more properties
        """ """
        region = self.indexing[:, pattern(self.indexing)]
        Ecut = self.E[region[0],region[1],region[2]]
        Ecut += 1 # doesn't change the initial E
        pass

    def add_detector(self, read_points: array,
                     output: Callable[[State], None]):
        """ """
        pass

    def add_source(self, pattern: Callable[[array], array[tuple[int, ...], np.dtype[bool]]],
                   rho: Callable[[array, float], array[tuple[int, ...], np.dtype[float]]]):
        pass

    def run(self, charge_dist: Callable[[array], array]):
        pass
