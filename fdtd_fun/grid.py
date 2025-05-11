from __future__ import annotations
from typing import Callable

import numpy as np
from numpy import ndarray

from . import constants as const
from .grid_object import GridObject
from .boundary import Boundary
from .detector import Detector
from .material import Material
from .source import Source
from .typing_ import Index


class Grid:
    """DOCS"""
    ds: float
    dt: float
    courant: float
    t: int
    boundaries: dict[str, Boundary] = {}
    detectors: dict[str, Detector] = {}
    materials: dict[str, Material] = {}
    sources: dict[str, Source] = {}
    E: ndarray  # some sort of field

    def __init__(self, shape: tuple[float | int, float | int, float | int], ds: float, courant=None):  # add boundaries
        """ """
        self.ds = ds
        self.t = 0
        self.Nx, self.Ny, self.Nz = self._handle_tuple(shape)
        if self.Nx < 0 or self.Ny < 0 or self.Nz < 0:
            raise ValueError("grid dimensions must be non-negative")
        D = int(self.Nx > 1) + int(self.Ny > 1) + int(self.Nz > 1)
        max_courant = const.stability * float(D) ** (-0.5)
        if courant is None:
            self.courant = max_courant
        elif courant > max_courant:
            raise ValueError(f"courant_number {courant} too high for "
                             f"a {D}D simulation, have to use {max_courant} or lower")
        else:
            self.courant = float(courant)
        self.dt = self.ds * self.courant / const.c

        vectorShape = (self.Nx, self.Ny, self.Nz, 3)
        self.E = np.zeros(vectorShape)

    def __setitem__(self, key, obj):
        # the grid supports indexing either by slices or by a 3-tuple of identically shaped ndarrays
        if not (isinstance(obj, GridObject)):
            raise TypeError("grid only accepts grid objects")
        if not isinstance(key, tuple):
            x, y, z = key, slice(None), slice(None)
        elif len(key) == 1:
            x, y, z = key[0], slice(None), slice(None)
        elif len(key) == 2:
            x, y, z = key[0], key[1], slice(None)
        elif len(key) == 3:
            x, y, z = key
        else:
            raise KeyError("maximum number of indices for the grid is 3")
        # region make sure ndarrays match in shape
        arrays: list[ndarray] = []
        if isinstance(x, ndarray):
            arrays.append(x)
        if isinstance(y, ndarray):
            arrays.append(y)
        if isinstance(z, ndarray):
            arrays.append(z)
        if len(arrays) != 0:
            shape = arrays[0].shape
            for array in arrays:
                if array.shape != shape:
                    raise ValueError("passed ndarrays must match in shape")
        # endregion
        obj._register_grid(
            grid=self,
            x=self._handle_single_key(x),
            y=self._handle_single_key(y),
            z=self._handle_single_key(z),
        )
        self._add_object(obj)

    def run(self, charge_dist: Callable[[ndarray], ndarray]):
        # equalize - how? antidivergence?
        # region cycle
        # region boundaries
        # endregion
        # region evolution rules
        # endregion
        # region sources
        # endregion
        # region detect
        # endregion
        # endregion
        pass

    def _add_object(self, obj: GridObject):
        if isinstance(obj, Boundary):
            dictionary = self.boundaries
        elif isinstance(obj, Detector):
            dictionary = self.detectors
        elif isinstance(obj, Material):
            dictionary = self.materials
        elif isinstance(obj, Source):
            dictionary = self.sources
        else:
            raise TypeError("Grid only accepts GridObjects")
        if dictionary.keys().__contains__(obj.name):
            raise KeyError("Object with this name is already on the grid")
        # TODO: maybe we need to make sure the objects don't intersect
        dictionary[obj.name] = obj

    def _handle_single_key(self, key) -> Index:
        if isinstance(key, ndarray):
            return self.handle_distance(key)  # maybe this won't be identical to float|int at some point. don't care rn.
        elif isinstance(key, slice):
            return self._handle_slice(key)
        elif isinstance(key, float | int):
            return self.handle_distance(key)
        else:
            raise TypeError("key must be ndarray, slice, float, or int")

    def _handle_slice(self, s: slice) -> slice:
        start = (
            s.start
            if isinstance(s.start, int) or s.start is None
            else self.handle_distance(s.start)
        )
        stop = (
            s.stop
            if isinstance(s.stop, int) or s.stop is None
            else self.handle_distance(s.stop)
        )
        step = (
            s.step
            if isinstance(s.step, int) or s.step is None
            else self.handle_distance(s.step)
        )
        return slice(start, stop, step)

    def _handle_tuple(self, shape: tuple[float | int, float | int, float | int]
                      ) -> tuple[int, int, int]:
        if len(shape) != 3:
            raise ValueError(
                f"invalid grid shape {shape}\n"
                f"grid shape should be a 3D tuple containing floats or ints"
            )
        x, y, z = shape
        x = self.handle_distance(x)
        y = self.handle_distance(y)
        z = self.handle_distance(z)
        return x, y, z

    def handle_distance(self, distance: float | int | ndarray):
        if isinstance(distance, int):
            return distance
        elif isinstance(distance, float):
            return int(distance / self.ds + 0.5)
        elif isinstance(distance, ndarray):
            if distance.dtype == float:  # get the actual float types here when testing
                return (distance / self.ds + 0.5).astype(int)
            elif distance.dtype == int:
                return distance
        raise TypeError("Distance values should be float, int, or ndarrays of float, int")
