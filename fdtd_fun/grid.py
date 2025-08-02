from __future__ import annotations
from typing import Callable

import numpy as np
import pickle

import scipy.sparse.linalg
from numpy import ndarray
from scipy.sparse import csr_array, dia_array
from scipy.sparse.linalg import factorized

from . import constants as const
from .constants import c
from .grid_object import GridObject
from .boundary import Boundary
from .detector import Detector
from .conductor import Conductor
from .source import Source
from .typing_ import Key
from enum import Enum


class Field(Enum):
    E = 0
    B = 1
    J = 2
    rho = -1  # only use this in detectors


class Comp(Enum):
    x = 0
    y = 1
    z = 2


class State:
    def __init__(self, E: ndarray | None, B: ndarray | None, J: ndarray | None, rho: ndarray | None):
        self.E: ndarray | None = E
        self.B: ndarray | None = B
        self.J: ndarray | None = J
        self.rho: ndarray | None = rho


def _curl_E(E: ndarray) -> ndarray:
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
    div[:-1, :, :] += E[0, 1:, :, :] - E[0, :-1, :, :]
    div[:, :-1, :] += E[1, :, 1:, :] - E[1, :, :-1, :]
    div[:, :, :-1] += E[2, :, :, 1:] - E[2, :, :, :-1]
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
        if self.Nx < 1 or self.Ny < 1 or self.Nz < 1:
            raise ValueError("grid dimensions must be positive")
        # region determine the courant number
        dim = int(self.Nx > 1) + int(self.Ny > 1) + int(self.Nz > 1)
        max_courant = const.stability * float(dim) ** (-0.5)
        if courant is None:
            self.courant = max_courant
        elif courant > max_courant:
            raise ValueError(f"courant_number {courant} too high for "
                             f"a {dim}D simulation, have to use {max_courant} or lower")
        else:
            self.courant = float(courant)
        # endregion
        self.dt = self.ds * self.courant / const.c

        self.inner_indices: ndarray = (np.indices((self.Nx, self.Ny, self.Nz)))
        self.inner_positions: ndarray = self.ds * np.asarray(np.meshgrid(np.arange(self.Nx),
                                                                         np.arange(self.Ny),
                                                                         np.arange(self.Nz), indexing="ij"))
        self.border_indices: ndarray = (np.indices((self.Nx + 2, self.Ny + 2, self.Nz + 2))
                                        - np.asarray((1, 1, 1), int)[:, None]).reshape((3, -1))
        self.borderIndexSorter = np.argsort(self.border_indices, axis=1)
        self.material_indices: ndarray = np.zeros(
            (3, 0))
        self.materialIndexSorter = np.argsort(self.material_indices, axis=1)
        self.emf: ndarray = np.zeros((3, self.Nx, self.Ny, self.Nz))
        self.materialMask = np.zeros((self.Nx, self.Ny, self.Nz), int)  # TODO: maybe make this boolean?

        self.PML: csr_array = None  # converts S to boundary vector, rect
        # self.B: dia_array = None  # converts S to b, square
        self.solver: Callable[[np.ndarray], np.ndarray]
        # region stored state
        self.S: ndarray = np.zeros(
            self.inner_indices.shape[1] * self.inner_indices.shape[2] * self.inner_indices.shape[3]
            * 3 * 3)  # [Nx*Ny*Nz*3*3] (pos(3), E/B/J, x/y/z)
        self.b: ndarray = np.zeros(
            self.border_indices.shape[1] * 2 * 3)  # [nBorder*3*3] (borderIndex, E/B, x/y/z) NO J!
        self.rho: ndarray = np.zeros((self.Nx, self.Ny, self.Nz))
        # endregion

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
        x, y, z = self._handle_single_key(x), self._handle_single_key(y), self._handle_single_key(z)
        obj._register_grid(
            grid=self,
            x=x,
            y=y,
            z=z,
        )
        self._add_object(obj)

    def run(self, charge_dist: Callable[[ndarray], ndarray],
            time: float | int, save_path: str = None,
            trigger: Callable = None):
        starting_rho = charge_dist(self.inner_positions)
        if starting_rho.shape != self.rho.shape:
            raise ValueError("charge_dist function must return blah blah blah")
        self.rho = starting_rho
        # equalize - how? antidivergence?
        if save_path is not None:
            self.file = open(save_path + f"{self.name}.dat", "wb")
            pickle.dump(self, self.file, protocol=-1)
        if isinstance(time, float):
            time = int(time / self.dt)
        self._prep_solver()
        while self.t < time:
            self._step()
            if trigger is not None:
                trigger()
            self.t += 1
        if self.file is not None:
            self.file.close()

    # region file stuff
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
            self.B = state.B
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
        _dict.pop("B")
        _dict.pop("J")
        _dict.pop("rho")
        return _dict

    # endregion

    # region step stuff

    def _prep_solver(self):
        A = csr_array((self.S.shape[0], self.S.shape[0]))  # S -> [dt * F]_from_free_state
        I = dia_array((self.S.shape[0], self.S.shape[0]))  # identity.
        I.setdiag(1, 0)
        R = self._get_reflecting_boundary()  # S -> boundary
        B = csr_array((self.S.shape[0], self.b.shape[0]))  # boundary -> [dt * F]_from_boundary_conditions

        inner = self.inner_indices.reshape((3, -1))

        def add_eq(toIndices: ndarray, toField: Field, toComp: Comp, fromField: Field, fromComp: Comp,
                   shift: tuple[int, int, int],
                   value: float)->None:
            """

            :param toIndices: the position indices in F to which this transformation maps, a (3,n)-shaped array
            :param toField: the field to which this transformation maps
            :param toComp: the component to which this transformation maps
            :param fromField: the field from which this transformation maps
            :param fromComp: the component from which this transformation maps
            :param shift: the shift that is added to toIndices to get the position indices in S
             from which this transformation maps
            :param value: the proportionality constant
            """
            # assume toIndices are all in the free state
            fromIndices = toIndices + np.asarray(shift, int)[:, None]
            in_free_state = self.in_free_state(fromIndices[0], fromIndices[1], fromIndices[2], fromField)
            if fromField != Field.J:  # completely get rid of fromJ values outside of the free state as they are all zero
                rejection = fromIndices[:, ~in_free_state]  # supposed to be subset of border_indices
                B[(self.ravelSIndices(toIndices, toField, toComp),
                   self.ravelBIndices(rejection, fromField, fromComp))] = value
            fromIndices = fromIndices[:, in_free_state]
            A[(self.ravelSIndices(toIndices, toField, toComp),
               self.ravelSIndices(fromIndices, fromField, fromComp))] = value

        # region to E field
        # region curl B term
        curlBValue = c / 2 * self.courant
        # region dBz/dy - dBy/dz
        add_eq(inner, Field.E, Comp.x, Field.B, Comp.z, (0, 1, 0), curlBValue)
        add_eq(inner, Field.E, Comp.x, Field.B, Comp.z, (0, -1, 0), -curlBValue)
        add_eq(inner, Field.E, Comp.x, Field.B, Comp.y, (0, 0, 1), -curlBValue)
        add_eq(inner, Field.E, Comp.x, Field.B, Comp.y, (0, 0, -1), curlBValue)
        # endregion
        # region dBx/dz - dBz/dx
        add_eq(inner, Field.E, Comp.y, Field.B, Comp.x, (0, 0, 1), curlBValue)
        add_eq(inner, Field.E, Comp.y, Field.B, Comp.x, (0, 0, -1), -curlBValue)
        add_eq(inner, Field.E, Comp.y, Field.B, Comp.z, (1, 0, 0), -curlBValue)
        add_eq(inner, Field.E, Comp.y, Field.B, Comp.z, (-1, 0, 0), curlBValue)
        # endregion
        # region dBy/dx - dBx/dy
        add_eq(inner, Field.E, Comp.z, Field.B, Comp.y, (1, 0, 0), curlBValue)
        add_eq(inner, Field.E, Comp.z, Field.B, Comp.y, (-1, 0, 0), -curlBValue)
        add_eq(inner, Field.E, Comp.z, Field.B, Comp.x, (0, 1, 0), -curlBValue)
        add_eq(inner, Field.E, Comp.z, Field.B, Comp.x, (0, -1, 0), curlBValue)
        # endregion
        # endregion
        # region J term
        Jvalue = -c ** 2 * self.dt * const.mu_0
        add_eq(inner, Field.E, Comp.x, Field.J, Comp.x, (0, 0, 0), Jvalue)
        add_eq(inner, Field.E, Comp.y, Field.J, Comp.y, (0, 0, 0), Jvalue)
        add_eq(inner, Field.E, Comp.z, Field.J, Comp.z, (0, 0, 0), Jvalue)
        # endregion
        # endregion
        # region to B field
        # region curl E term
        curlEValue = -self.courant / 2 / c
        # region dEz/dy - dEy/dz
        add_eq(inner, Field.B, Comp.x, Field.E, Comp.z, (0, 1, 0), curlEValue)
        add_eq(inner, Field.B, Comp.x, Field.E, Comp.z, (0, -1, 0), -curlEValue)
        add_eq(inner, Field.B, Comp.x, Field.E, Comp.y, (0, 0, 1), -curlEValue)
        add_eq(inner, Field.B, Comp.x, Field.E, Comp.y, (0, 0, -1), curlEValue)
        # endregion
        # region dEx/dz - dEz/dx
        add_eq(inner, Field.B, Comp.y, Field.E, Comp.x, (0, 0, 1), curlEValue)
        add_eq(inner, Field.B, Comp.y, Field.E, Comp.x, (0, 0, -1), -curlEValue)
        add_eq(inner, Field.B, Comp.y, Field.E, Comp.z, (1, 0, 0), -curlEValue)
        add_eq(inner, Field.B, Comp.y, Field.E, Comp.z, (-1, 0, 0), curlEValue)
        # endregion
        # region dEy/dx - dEx/dy
        add_eq(inner, Field.B, Comp.z, Field.E, Comp.y, (1, 0, 0), curlEValue)
        add_eq(inner, Field.B, Comp.z, Field.E, Comp.y, (-1, 0, 0), -curlEValue)
        add_eq(inner, Field.B, Comp.z, Field.E, Comp.x, (0, 1, 0), -curlEValue)
        add_eq(inner, Field.B, Comp.z, Field.E, Comp.x, (0, -1, 0), curlEValue)
        # endregion
        # endregion
        # endregion
        # region to J field
        for _, material in self.conductors.items():
            Evalue = material.s * material.rho_f * self.dt
            Jvalue = - material.s * material.rho_f * self.dt / material.sigma
            matIndices = np.asarray(material.x, material.y, material.z)
            add_eq(matIndices, Field.J, Comp.x, Field.J, Comp.x, (0, 0, 0), Jvalue)
            add_eq(matIndices, Field.J, Comp.y, Field.J, Comp.y, (0, 0, 0), Jvalue)
            add_eq(matIndices, Field.J, Comp.z, Field.J, Comp.z, (0, 0, 0), Jvalue)
            add_eq(matIndices, Field.J, Comp.x, Field.E, Comp.x, (0, 0, 0), Evalue)
            add_eq(matIndices, Field.J, Comp.y, Field.E, Comp.y, (0, 0, 0), Evalue)
            add_eq(matIndices, Field.J, Comp.z, Field.E, Comp.z, (0, 0, 0), Evalue)

        # endregion

        solver = factorized(I - A - B @ R)  # TODO: check signs

    def ravelSIndices(self, posIndices: ndarray, field: Field, component: Comp):
        if field != Field.J:
            return np.ravel_multi_index((*posIndices, field.value, component.value),
                                        (self.Nx, self.Ny, self.Nz, 3, 3))
        else:  # posIndices are expected to be in the free state J (inside materials)
            offset = self.S.shape[0]
            material_indices = self.materialIndexSorter[np.searchsorted(self.material_indices, posIndices,
                                                                        sorter=self.materialIndexSorter)]
            return np.ravel_multi_index((material_indices, field.value, component.value),
                                        (self.material_indices.shape[1], 3)) + offset

    def ravelBIndices(self, posIndices: ndarray, field: Field, component: Comp):
        if field == Field.J:
            raise ValueError("J is not stored in the boundary vector as it is always zero in the boundary!")
        border_indices = self.borderIndexSorter[np.searchsorted(self.border_indices, posIndices,
                                                                sorter=self.borderIndexSorter)]
        return np.ravel_multi_index((border_indices, field.value, component.value),
                                    (self.border_indices.shape[1], 2, 3))

    def _step(self):
        print(np.sum(self.rho))
        for _, src in self.sources.items():
            src.apply()
        if self.file is not None:
            pickle.dump(State(self.E, self.B, self.J, self.rho), self.file, protocol=-1)
        self._update_H()
        for _, material in self.conductors.items():
            material._get_J()
        self._update_rho()
        self._update_E()

        for _, det in self.detectors.items():
            det.read()

    # endregion

    # region boundaries

    def _get_reflecting_boundary(self) -> csr_array:
        return csr_array((self.b.shape[0], self.S.shape[0]))  # yummy empty matrix

    # endregion

    def in_free_state(self, x: ndarray | int, y: ndarray | int, z: ndarray | int, field: Field):
        result = (x >= 0) & (x < self.Nx) & (y >= 0) & (y < self.Ny) & (z >= 0) & (z < self.Nz)
        if field == Field.J:
            result = result & (self.materialMask[x, y, z] == 1)
        return result

    def _update_E(self):
        for _, boundary in self.boundaries.items():
            boundary.update_phi_E()  # etc etc

        curl = _curl_H(self.B)
        self.E += (curl / self.ds - self.J) / const.eps_0 * self.dt

        for _, boundary in self.boundaries.items():
            boundary.update_E()  # etc etc

    def _update_H(self):
        self.lastH = self.B
        for _, boundary in self.boundaries.items():
            boundary.update_phi_H()  # etc etc

        curl = _curl_E(self.E)
        self.B += -curl / self.ds / const.mu_0 * self.dt

        for _, boundary in self.boundaries.items():
            boundary.update_H()  # etc etc

    def _update_rho(self):
        drho = -_div_E(self.J) * self.dt / self.ds
        mask_traversal = np.asarray((self.materialMask[2:, 1:-1, 1:-1],
                                     self.materialMask[0:-2, 1:-1, 1:-1],
                                     self.materialMask[1:-1, 2:, 1:-1],
                                     self.materialMask[1:-1, 0:-2, 1:-1],
                                     self.materialMask[1:-1, 1:-1, 2:],
                                     self.materialMask[1:-1, 1:-1, 0:-2])
                                    )
        mask_traversal[:, self.materialMask[1:-1, 1:-1, 1:-1] == 1] = 0
        self.rho += drho

    # region distance-Index helpers
    def _add_object(self, obj: GridObject):
        """Validate and add a GridObject"""
        if isinstance(obj, Boundary):
            dictionary = self.boundaries
        elif isinstance(obj, Detector):
            dictionary = self.detectors
        elif isinstance(obj, Conductor):
            dictionary = self.conductors
            self.material_indices = np.concatenate((self.material_indices,
                                                    self.inner_indices[:, obj.x, obj.y, obj.z].reshape(3, -1)), axis=1)
            self.materialIndexSorter = np.argsort(self.material_indices, axis=1)
            self.materialMask[obj.x, obj.y, obj.z] = 1
        elif isinstance(obj, Source):
            dictionary = self.sources
        else:
            raise TypeError("Grid only accepts GridObjects")
        if dictionary.keys().__contains__(obj.name):
            raise KeyError("Object with this name is already on the grid")
        # TODO: maybe we need to make sure the objects don't intersect
        dictionary[obj.name] = obj

    def _handle_single_key(self, key) -> Key:
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

    def positions(self, x: Key, y: Key, z: Key):
        return self.inner_positions[:, x, y,
               z]  # TODO: get rid of this and only calculate what is needed for sources etc. once

    def time(self):
        return self.t * self.dt
    # endregion
