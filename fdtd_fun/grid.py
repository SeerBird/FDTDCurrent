from __future__ import annotations
from typing import Callable

import numpy as np
import pickle
from numpy import ndarray

from . import constants as const
from .grid_object import GridObject
from .boundary import Boundary
from .detector import Detector
from .conductor import Conductor
from .source import Source
from .typing_ import Index
from enum import Enum

class Field(Enum):
    E = 1
    H = 2
    J = 3
    rho = 4


class State:
    def __init__(self, E: ndarray | None, H: ndarray | None, J: ndarray | None, rho: ndarray | None):
        self.E: ndarray | None = E
        self.H: ndarray | None = H
        self.J: ndarray | None = J
        self.rho: ndarray | None = rho


def _curl_E(E: ndarray) -> ndarray:
    # TODO: figure out what the grids actually look like and formalize all the differential ops
    curl = np.zeros_like(E)
    # dz/dy - dy/dz
    curl[0, :, :-1, :] += E[2, :, 1:, :] - E[2, :, :-1, :]
    curl[0, :, :, :-1] -= E[1, :, :, 1:] - E[1, :, :, :-1]
    # dx/dz-dz/dx
    curl[1, :, :, :-1] += E[0, :, :, 1:] - E[0, :, :, :-1]
    curl[1, :-1, :, :] -= E[2, 1:, :, :] - E[2, :-1, :, :]
    # dy/dx-dx/dy
    curl[2, :-1, :, :] += E[1, 1:, :, :] - E[1, :-1, :, :]
    curl[2, :, :-1, :] -= E[0, :, 1:, :] - E[0, :, :-1, :]
    return curl  # zero on the high edges of the region


def _curl_H(H: ndarray) -> ndarray:
    curl = np.zeros_like(H)
    # dz/dy - dy/dz
    curl[0, :, 1:, :] += H[2, :, 1:, :] - H[2, :, :-1, :]
    curl[0, :, :, 1:] -= H[1, :, :, 1:] - H[1, :, :, :-1]
    # dx/dz-dz/dx
    curl[1, :, :, 1:] += H[0, :, :, 1:] - H[0, :, :, :-1]
    curl[1, 1:, :, :] -= H[2, 1:, :, :] - H[2, :-1, :, :]
    # dy/dx-dx/dy
    curl[2, 1:, :, :] += H[1, 1:, :, :] - H[1, :-1, :, :]
    curl[2, :, 1:, :] -= H[0, :, 1:, :] - H[0, :, :-1, :]
    return curl  # zero on the low edges of the region


def _div_E(E: ndarray) -> ndarray:
    div = np.zeros((E.shape[1], E.shape[2], E.shape[3]))
    div[1:, :, :] += E[0, 1:, :, :] - E[0, :-1, :, :]
    div[:, 1:, :] += E[1, :, 1:, :] - E[1, :, :-1, :]
    div[:, :, 1:] += E[2, :, :, 1:] - E[2, :, :, :-1]
    return div


class Grid:
    """The FDTD grid - the core of this library. The intended use is to create a Grid object,
     assign GridObject objects to the grid by indexing the Grid object(see __setitem__ below),
      and then use the run() method below"""

    def __init__(self, name: str, shape: tuple[float | int, float | int, float | int], ds: float,
                 courant: float = None):  # add boundaries
        """

        :param shape: the dimensions of the grid, a float|int 3-tuple. int values will be used as indexes,
         and float values will be converted to indexes using the ds value given
        :param ds: the spacial step of the grid, in meters
        :param courant: the courant number for the simulation
        """
        self.file = None
        self.name: str = name
        self.boundaries: dict[str, Boundary] = {}
        self.detectors: dict[str, Detector] = {}
        self.conductors: dict[str, Conductor] = {}
        self.sources: dict[str, Source] = {}
        self.ds: float = ds  # space step
        self.dt: float  # time step
        self.courant: float  # the courant number, c*dt/ds
        self.t: int = 0  # current time index
        self.Nx, self.Ny, self.Nz = self._handle_tuple(shape)  # index dimensions of the grid
        self._positions: ndarray = self.ds * np.asarray(np.meshgrid(np.arange(self.Nx),
                                                                    np.arange(self.Ny),
                                                                    np.arange(self.Nz), indexing="ij"))
        if self.Nx < 0 or self.Ny < 0 or self.Nz < 0:
            raise ValueError("grid dimensions must be non-negative")
        dim = int(self.Nx > 1) + int(self.Ny > 1) + int(self.Nz > 1)
        max_courant = const.stability * float(dim) ** (-0.5)
        if courant is None:
            self.courant = max_courant
        elif courant > max_courant:
            raise ValueError(f"courant_number {courant} too high for "
                             f"a {dim}D simulation, have to use {max_courant} or lower")
        else:
            self.courant = float(courant)
        self.dt = self.ds * self.courant / const.c
        self.E: ndarray = np.zeros((3, self.Nx, self.Ny, self.Nz))
        self.H: ndarray = np.zeros((3, self.Nx, self.Ny, self.Nz))
        self.J: ndarray = np.zeros((3, self.Nx, self.Ny, self.Nz))
        self.rho: ndarray = np.zeros((self.Nx, self.Ny, self.Nz))
        self.materialMask = np.zeros((self.Nx, self.Ny, self.Nz), int)

    def __setitem__(self, key, obj):
        """
        Assign a GridObject to a subset of the grid
        :param key: a tuple of slices, numbers, or ndarrays. All ndarrays must be of the same shape,
        all int values will be treated as indexes and all float values will be treated as distances and converted to
        indexes. Indexing with ndarrays is very inefficient and should be avoided.
        :param obj: a GridObject object
        """
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

    def run(self, charge_dist: Callable[[ndarray], ndarray],
            time: float | int, save_path: str = None,
            trigger: Callable = None):
        starting_rho = charge_dist(self._positions)
        if starting_rho.shape != self.rho.shape:
            raise ValueError("charge_dist function must return blah blah blah")
        self.rho = starting_rho
        # equalize - how? antidivergence?
        if save_path is not None:
            self.file = open(save_path + f"{self.name}.dat", "wb")
            pickle.dump(self, self.file,protocol=-1)
        if isinstance(time, float):
            time = int(time / self.dt)
        while self.t < time:
            self._step()
            if trigger is not None:
                trigger()
            self.t += 1
        if self.file is not None:
            self.file.close()

    @classmethod
    def load_from_file(cls, save_path: str) -> Grid:
        """
        Loads a Grid object from a file, restores all GridObjects, and sets the state to the first recorded state
        :param save_path: specify format here
        :return: new Grid object loaded from the file
        """
        file = open(save_path, "rb")
        grid = pickle.load(file)
        if isinstance(grid, Grid):
            grid.file = file
            # maybe check out persistent ID pickle stuff
            return grid
        raise Exception("haha")

    def load_next_frame(self) -> bool:
        if self.file is None or not self.file.mode == "rb":
            raise NotImplementedError(
                "This method is only callable on a Grid object that has been loaded from a file - "
                "please use Grid.load_from_file()")
        try:
            state = pickle.load(self.file)
            if not isinstance(state, State):
                raise ValueError("The value read from the file was not a State object")
            self.E = state.E
            self.H = state.H
            self.J = state.J
            self.rho = state.rho
            for _, detector in self.detectors.items():
                detector.read()
            return False
        except EOFError:
            self.file.close()
            return True

    def __getstate__(self):
        _dict = self.__dict__.copy()
        _dict.pop("file")
        _dict.pop("E")
        _dict.pop("H")
        _dict.pop("J")
        _dict.pop("rho")
        return _dict

    def _step(self):
        for _, src in self.sources.items():
            src.apply()
        if self.file is not None:
            pickle.dump(State(self.E, self.H, self.J, self.rho), self.file,protocol = -1)
        self._update_E()
        self._update_H()
        for _, material in self.conductors.items():
            material._update_J()
            self.rho -= _div_E(self.J) * self.materialMask
        for _, det in self.detectors.items():
            det.read()
        for _, src in self.sources.items():
            src.cancel()

    def _update_E(self):
        for _, boundary in self.boundaries.items():
            boundary.update_phi_E()  # etc etc

        curl = _curl_H(self.H)
        self.E += self.courant * (curl - self.J) / np.sqrt(const.eps_0 / const.mu_0)

        for _, boundary in self.boundaries.items():
            boundary.update_E()  # etc etc

    def _update_H(self):
        for _, boundary in self.boundaries.items():
            boundary.update_phi_H()  # etc etc

        curl = _curl_E(self.E)
        self.H += self.courant * -curl * np.sqrt(const.eps_0 / const.mu_0)

        for _, boundary in self.boundaries.items():
            boundary.update_H()  # etc etc

    # region distance-Index helpers
    def _add_object(self, obj: GridObject):
        """Validate and add a GridObject"""
        if isinstance(obj, Boundary):
            dictionary = self.boundaries
        elif isinstance(obj, Detector):
            dictionary = self.detectors
        elif isinstance(obj, Conductor):
            dictionary = self.conductors
            self.materialMask[obj.x, obj.y, obj.z] = 1
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
            dist = self.handle_distance(key)
            return slice(dist, dist + 1, 1)
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

    def handle_distance(self, distance: float | int | ndarray):
        # TODO: make sure this is convenient for indexing(no holes, no overlaps with the most obvious
        # approaches etc.)
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

    def _handle_tuple(self, shape: tuple[float | int, float | int, float | int]) -> tuple[int, int, int]:
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

    def positions(self, x: Index, y: Index, z: Index):
        return self._positions[:, x, y, z]

    def time(self):
        return self.t * self.dt
    # endregion
