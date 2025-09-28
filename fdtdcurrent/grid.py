from __future__ import annotations

import logging
import warnings
from typing import Callable

import numpy as np
import pickle

from numpy import ndarray
from scipy.sparse import csr_array, dia_array
from scipy.sparse.linalg import factorized

from . import constants as const
from .constants import c
from .grid_object import GridObject
from .detector import Detector
from .conductor import Conductor
from .mylogging import printProgressBar
from .source import Source
from .typing_ import Key, Field, Comp

logger = logging.getLogger(__name__)


class Grid:
    """
    The FDTD grid - the core of this library. The intended use is to create a Grid object,
     assign GridObject objects to the grid by indexing the Grid object(see __setitem__ below),
      and then use the run() method below
    """

    def __init__(self, name: str, shape: tuple[float | int, float | int, float | int], ds: float = None,
                 dt: float = None):
        """
        :param shape: the dimensions of the grid, a float|int 3-tuple. int values will be used as indexes,
         and float values will be converted to indexes using the ds value given
        :param ds: the spacial step of the grid, in meters
        :param dt: the time step of the simulation, in seconds
        """
        self.file = None  # is not None if this Grid is saving to a file or was loaded from a file
        self.save_path = None  # is not None if this Grid was loaded from a file
        self.tot_frames = None  # how many steps this grid ran for, is not None if this Grid was loaded from a file
        self.name: str = name
        self.detectors: dict[str, Detector] = {}
        self.conductors: dict[str, Conductor] = {}
        self.sources: dict[str, Source] = {}
        self.ds: float  # space step
        self.dt: float  # time step
        self.t: int = 0  # current time index
        self.Nx, self.Ny, self.Nz = self._handle_tuple(shape)  # index dimensions of the grid
        self.shape = (self.Nx, self.Ny, self.Nz)
        if self.Nx < 1 or self.Ny < 1 or self.Nz < 1:
            raise ValueError("grid dimensions must be positive")
        # region determine the steps
        dim = int(self.Nx > 1) + int(self.Ny > 1) + int(self.Nz > 1)
        if dim == 0:
            max_courant = 1
        else:
            max_courant = const.stability * float(dim) ** (-0.5)
        # TODO: do we care about the Courant–Friedrichs–Lewy condition now that we're doing Crank–Nicolson?
        if dt is None and ds is None:
            raise ValueError("Please define a time step or a space step value (or both) for the Grid")
        if ds is None:
            self.dt = dt
            self.ds = self.dt * const.c / max_courant
        elif dt is None:
            self.ds = ds
            self.dt = self.ds * max_courant / const.c
        elif const.c * dt / ds > max_courant:
            raise ValueError(f"Courant number (c*dt/ds) {const.c * dt / ds} is too high for "
                             f"a {dim}D simulation , have to use {max_courant} or lower")
        else:
            self.dt = dt
            self.ds = ds
        # endregion
        logger.info(f"Timestep: {self.dt:.2E} s, space step: {self.ds:.2e} m")
        self.inner_indices: ndarray = np.indices(self.shape, int)
        self.boundary_indices: ndarray = (np.indices((self.Nx + 2, self.Ny + 2, self.Nz + 2)).reshape((3, -1))
                                          - np.asarray((1, 1, 1), int)[:, None])
        self.boundary_indices = self.boundary_indices[:,
        ((self.boundary_indices[0] == -1) | (self.boundary_indices[0] == self.Nx)) |
        ((self.boundary_indices[1] == -1) | (self.boundary_indices[1] == self.Ny)) |
        ((self.boundary_indices[2] == -1) | (self.boundary_indices[2] == self.Nz))]
        self.cond_indices: ndarray = np.zeros(
            (3, 0), int)
        # self.R: csr_array  # converts S to boundary vector, rect
        self.B: dia_array  # converts S to b_linear, square
        self.solver: Callable[[np.ndarray], np.ndarray]  # solves the matrix equation for the step
        # region stored state
        self.State: ndarray = np.zeros((*self.shape, 3, 3))  # x,y,z,Field,Comp
        self.dof: int = self.Nx * self.Ny * self.Nz * 2 * 3  # degrees of freedom, this will increase as conductors are added
        self.EBdof: int = self.dof  # electric and magnetic field degrees of freedom
        self.Sprev: ndarray  # previous state [Nx*Ny*Nz*2*3 + NJ*3]
        # TODO: consider adding charge density tracking. to this charge redistribution sim.

        # endregion

    # region indexing the grid
    def _handle_key(self, key):
        # region pad the key with None slices or scream that the key is too long
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
        # endregion
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
                    raise ValueError("Ndarrays passed in the grid indexing key must match in shape")
        # endregion
        x, y, z = self._handle_single_key(x), self._handle_single_key(y), self._handle_single_key(z)
        return x, y, z

    def __setitem__(self, key, obj):
        """
        Assign a GridObject to a subset of the grid
        :param key: a tuple of slices, numbers, or ndarrays. All ndarrays must be of the same shape,
        all int values will be treated as indexes and all float values will be treated as coordinates in metres
         and converted to indexes. Indexing with ndarrays is very inefficient and should be avoided.
        :param obj: a GridObject object
        """
        if not (isinstance(obj, GridObject)):
            raise TypeError("Grid only accepts GridObjects")
        x, y, z = self._handle_key(key)
        obj._register_grid(
            grid=self,
            x=x,
            y=y,
            z=z,
        )
        self._add_object(obj)

    def _add_object(self, obj: GridObject):
        """Validate and add a GridObject"""
        if isinstance(obj, Detector):
            dictionary = self.detectors
        elif isinstance(obj, Conductor):
            dictionary = self.conductors
            cond_index = self._get_index(obj.x, obj.y, obj.z).reshape(3, -1)
            self.cond_indices = np.concatenate((self.cond_indices,
                                                cond_index), axis=1)
            self.dof += cond_index.size
        elif isinstance(obj, Source):
            dictionary = self.sources
        else:
            raise TypeError("Grid only accepts GridObjects")  # how did we get here?
        if dictionary.keys().__contains__(obj.name):
            raise KeyError("Object with this name is already on the grid")
        # TODO: maybe we need to make sure the objects don't intersect
        dictionary[obj.name] = obj

    def __getitem__(self, key):
        """
        :param key: a tuple of slices, numbers, or ndarrays. All ndarrays must be of the same shape,
        all int values will be treated as indexes and all float values will be treated as coordinates in metres
         and converted to indexes. Indexing with ndarrays is very inefficient and should be avoided.
        :return: state in the selected portion of the grid, (3,3,...) - shaped, with the first two indices being
         the field and the component
        """
        x, y, z = self._handle_key(key)
        return np.moveaxis(self.State[x, y, z], [-1, -2], [1, 0])

    def _get_index(self, x: Key, y: Key, z: Key):
        return self.inner_indices[:, x, y, z]

    # endregion

    def run(self, time: float | int, charge_dist: Callable[[ndarray], ndarray] = None, save: bool = False,
            trigger: Callable = None):
        """
        Run the Grid
        :param time: total time to run the Grid for, in seconds or steps
        :param charge_dist: the initial charge distribution, currently not used
        :param save: whether to save the simulation results to a file
        :param trigger: function to run before each step (plotting, saving a subset of the grid state etc.)
        """
        # TODO: consider actually using the starting charge distribution
        #  if we do we gotta find the starting fields, use antidivergence?
        if save:
            self.file = open(f"{self.name}.dat", "wb")
            # noinspection PyTypeChecker
            pickle.dump(self, self.file, protocol=-1)
        if isinstance(time, float):
            time = int(time / self.dt)
        logger.info(
            f"Running grid of shape ({self.shape[0]}, {self.shape[1]}, {self.shape[2]}) for {time} steps, {time * self.dt:.3E} s")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # warnings.resetwarnings()
            # TODO: remove/add back above line to suppress/return spare efficiency warnings
            self._prep_solver()
        # region find Sprev using an Euler step
        self.Sprev = np.zeros(self.dof)
        # TODO: make this more general as currently a lot of stuff is assumed zero and code is copied...
        # TODO: also consider doing a better step as this probably introduces more error
        # region get and add G
        G_J = np.zeros((*self.shape, 3))  # wasteful
        # region get K
        K = np.zeros((*self.shape, 3))
        for _, src in self.sources.items():
            K[src.x, src.y, src.z] += np.moveaxis(src.function(src.positions, self.time()), 0, -1)
        # endregion
        # region add K
        for _, mat in self.conductors.items():
            G_J[mat.x, mat.y, mat.z] += mat.s * mat.rho_f * K[mat.x, mat.y, mat.z]  # add K
        # endregion
        if self.cond_indices.size != 0:
            self.Sprev[self.EBdof:] -= G_J[*self.cond_indices].reshape((-1)) * self.dt
        # endregion
        # endregion
        while self.t < time:
            printProgressBar(self.t, time)
            if trigger is not None:
                for _, det in self.detectors.items():
                    det.read()
                trigger()
            self._step()
            if self.file is not None:
                # noinspection PyTypeChecker
                pickle.dump(self.State, self.file, protocol=-1)
                # TODO: am I wasting too much memory saving the whole state rather than just the free state?
                #  I think I can reduce the file size up to three times, but is it worth the inconvenience?
        if self.file is not None:
            self.file.close()

    # region file stuff
    @classmethod
    def load_from_file(cls, save_path: str) -> Grid:
        """
        Loads a Grid object from a file, restores all GridObjects, and sets the state to the initial state
        :param save_path: yep.
        :return: new Grid object loaded from the file
        """
        # region get total frames... this is painful
        logger.debug("Getting total frames")
        file = open(save_path, "rb")
        grid = pickle.load(file)
        if not isinstance(grid, Grid):
            raise ValueError(f"Expected Grid object as second pickle in save file, got {type(grid)}")
        grid.file = file
        grid.save_path = save_path
        while grid.load_next_frame():
            pass
        tot_frames = grid.t + 1
        # endregion
        # maybe check out persistent ID pickle stuff
        logger.debug("Loading grid from file")
        file.close()
        file = open(save_path, "rb")
        grid = pickle.load(file)
        if not isinstance(grid, Grid):
            raise ValueError(f"Expected Grid object as second pickle in save file, got {type(grid)}")
        grid.file = file
        grid.save_path = save_path
        grid.tot_frames = tot_frames
        for obj in [*grid.conductors.values(),*grid.detectors.values(),*grid.sources.values()]:
            obj._grid = grid
        return grid

    def load_next_frame(self) -> bool:
        """
        Loads the next frame of the simulation from the file if this Grid is loaded from a file.
        :return: True if successful, False if there were no more frames saved
        """
        if self.file is None or not self.file.mode == "rb":
            raise NotImplementedError(
                "This method is only callable on a Grid object that has been loaded from a file - "
                "please use Grid.load_from_file()")
        try:
            state = pickle.load(self.file)
            if not isinstance(state, ndarray):
                raise ValueError("The value read from the file was not a numpy array object - why?")
            self.State = state
            for _, detector in self.detectors.items():
                detector.read()
            self.t += 1
            return True
        except EOFError:
            self.file.close()
            return False

    def _reload(self):  # not sure if this is a good thing to make available
        self.file.close()
        file = open(self.save_path, "rb")
        grid = pickle.load(file)
        if isinstance(grid, Grid):
            self.file = file
            self.State = grid.State
            self.t = grid.t
            for _, detector in self.detectors.items():
                detector.read()
            return
        raise Exception("why is this not a grid")

    def __getstate__(self):
        # serialization cleaning
        _dict = self.__dict__.copy()
        _dict.pop("file")
        return _dict

    # endregion

    # region step stuff

    def _prep_solver(self):
        logger.debug("Prepping matrices")
        # TODO: is there a way to easily optimise this? probably doesn't matter though as
        #  this is often going to be negligible in comparison to running the sim
        raveler = np.zeros((*self.shape, 3, 3), int) - 1  # should not ever index the -1s
        raveler[:, :, :, :-1, :] = np.arange(self.EBdof).reshape((*self.shape, 2, 3))
        raveler[*self.cond_indices, Field.J.value, :] = np.arange(self.EBdof, self.dof).reshape((-1, 3))
        self.raveler = raveler
        A = csr_array((self.dof, self.dof))  # S -> [dt * F]_from_free_state
        I = dia_array((self.dof, self.dof))  # identity.
        I.setdiag(1, 0)
        R = self._get_wrap_boundary()  # S -> boundary
        # TODO: consider ignoring the boundary (effectively making it always fully reflective)
        #  and then adding a PML(which would just be a change to A and H)
        #  or adding a customizable boundary(PML of varying thickness/wrap/reflect)
        #  (customizable how? by side or by subset of the boundary?)
        H = csr_array(
            (self.dof, self.boundary_indices.shape[1] * 2 * 3))  # boundary -> [dt * F]_from_boundary_conditions

        inner = self.inner_indices.reshape((3, -1))

        def add_eq(toIndices: ndarray, toField: Field, toComp: Comp, fromField: Field, fromComp: Comp,
                   shift: tuple[int, int, int],
                   value: float) -> None:
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
            in_free_state = self._in_free_state(fromIndices[0], fromIndices[1], fromIndices[2], fromField)
            if fromField != Field.J:  # completely get rid of fromJ values outside of the free state as they are all zero
                rejection = fromIndices[:, ~in_free_state]  # supposed to be subset of border_indices
                H[(raveler[*toIndices[:, ~in_free_state], toField.value, toComp.value],
                   self._ravelBIndices(rejection, fromField, fromComp))] = value
            A[raveler[*toIndices[:, in_free_state], toField.value, toComp.value],
            raveler[*fromIndices[:, in_free_state], fromField.value, fromComp.value]] = value

        # region to E field
        # region curl B term
        curlBValue = c ** 2 / 2 * self.dt / self.ds
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
        curlEValue = -self.dt / self.ds / 2
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
        for _, cond in self.conductors.items():
            condIndices = self._get_index(cond.x, cond.y, cond.z).reshape((3, -1))
            # region J term
            Jvalue = - cond.s * cond.rho_f * self.dt / cond.sigma
            add_eq(condIndices, Field.J, Comp.x, Field.J, Comp.x, (0, 0, 0), Jvalue)
            add_eq(condIndices, Field.J, Comp.y, Field.J, Comp.y, (0, 0, 0), Jvalue)
            add_eq(condIndices, Field.J, Comp.z, Field.J, Comp.z, (0, 0, 0), Jvalue)
            # endregion
            # region E term
            Evalue = cond.s * cond.rho_f * self.dt
            add_eq(condIndices, Field.J, Comp.x, Field.E, Comp.x, (0, 0, 0), Evalue)
            add_eq(condIndices, Field.J, Comp.y, Field.E, Comp.y, (0, 0, 0), Evalue)
            add_eq(condIndices, Field.J, Comp.z, Field.E, Comp.z, (0, 0, 0), Evalue)
            # endregion

        # endregion
        logger.debug("Factorising matrices")
        F = A + H @ R
        self.solver = factorized(I - F)
        self.B = I + F
        del self.raveler

    def _step(self):
        # region get b
        b = np.zeros(self.dof)
        # region get and add G
        G_J = np.zeros((*self.shape, 3))
        # region get K
        K = np.zeros((*self.shape, 3))
        for _, src in self.sources.items():
            K[src.x, src.y, src.z] += np.moveaxis(src.function(src.positions, self.time()), 0, -1)
        # endregion
        # region J.nabla J
        J = self.State[:, :, :, Field.J.value, :]
        # region nabla J
        nablaJ = np.zeros((*self.shape, 3, 3))  # x,y,z, J component, derivative component
        # this is nablaJ * ds
        nablaJ[1:-1, :, :, Comp.x.value] = (J[2:, :, :] - J[:-2, :, :]) / 2
        nablaJ[0, :, :, Comp.x.value] = J[0, :, :]  # boundary J is 0
        nablaJ[-1, :, :, Comp.x.value] = -J[-1, :, :]

        nablaJ[:, 1:-1, :, Comp.y.value] = (J[:, 2:, :] - J[:, :-2, :]) / 2
        nablaJ[:, 0, :, Comp.y.value] = J[:, 0, :]  # boundary J is 0
        nablaJ[:, -1, :, Comp.y.value] = -J[:, -1, :]

        nablaJ[:, :, 1:-1, Comp.z.value] = (J[:, :, 2:] - J[:, :, :-2]) / 2
        nablaJ[:, :, 0, Comp.z.value] = J[:, :, 0]  # boundary J is 0
        nablaJ[:, :, -1, Comp.z.value] = -J[:, :, -1]
        # endregion
        JnabJ = np.einsum("ijkl,ijkyl->ijky", J, nablaJ)
        # endregion
        # region add K and J x B and J.nabla J
        for _, mat in self.conductors.items():
            G_J[mat.x, mat.y, mat.z] += mat.s * mat.rho_f * K[mat.x, mat.y, mat.z]  # add K
            J_mat = self.State[mat.x, mat.y, mat.z, Field.J.value, :]
            B_mat = self.State[mat.x, mat.y, mat.z, Field.B.value, :]
            none_key = [slice(None)] * (len(J_mat.shape) - 1)
            G_J[mat.x, mat.y, mat.z, 0] += mat.s * (
                    J_mat[*none_key, 1] * B_mat[*none_key, 2] - J_mat[*none_key, 2] * B_mat[*none_key, 1])
            G_J[mat.x, mat.y, mat.z, 1] += mat.s * (
                    J_mat[*none_key, 2] * B_mat[*none_key, 0] - J_mat[*none_key, 0] * B_mat[*none_key, 2])
            G_J[mat.x, mat.y, mat.z, 2] += mat.s * (
                    J_mat[*none_key, 0] * B_mat[*none_key, 1] - J_mat[*none_key, 1] * B_mat[*none_key, 0])
            G_J[mat.x, mat.y, mat.z] += -1 / self.ds / mat.rho_f * JnabJ[
                mat.x, mat.y, mat.z]  # TODO: is this negligible?

        # endregion
        if self.cond_indices.size != 0:
            b[self.EBdof:] += 2 * G_J[*self.cond_indices].reshape((-1)) * self.dt
        # endregion
        b += self.B @ self.Sprev  # get and add the S_n-1 term
        # endregion
        Snext = self.solver(b)
        # region set state to the next timestep
        self.Sprev = np.concatenate(
            (self.State[:, :, :, :-1, :].flatten(), self.State[*self.cond_indices, Field.J.value].flatten()))
        self.State[:, :, :, :-1, :] = Snext[:self.EBdof].reshape((*self.shape, 2, 3))
        self.State[*self.cond_indices, Field.J.value, :] = Snext[self.EBdof:].reshape((-1, 3))
        self.t += 1
        # endregion

    def _in_free_state(self, x: ndarray, y: ndarray, z: ndarray, field: Field):
        result = (x >= 0) & (x < self.Nx) & (y >= 0) & (y < self.Ny) & (z >= 0) & (z < self.Nz)
        if field == Field.J:
            if self.cond_indices.size == 0:
                return result & False
            mask = np.full(self.shape, False)
            mask[*self.cond_indices] = True
            result = result & mask[x, y, z]
        return result

    # endregion

    # region boundaries
    def _add_border_eq(self, R, toIndices: ndarray, toField: Field, toComp: Comp,
                       fromIndices: ndarray, fromField: Field, fromComp: Comp, value: float):
        R[self._ravelBIndices(toIndices, toField, toComp), self.raveler[
            *fromIndices, fromField.value, fromComp.value]] = value

    def _get_reflecting_boundary(self) -> csr_array:
        R = csr_array((self.boundary_indices.shape[1] * 2 * 3, self.dof))  # yummy empty matrix
        return R

    def _get_wrap_boundary(self):
        R = csr_array((self.boundary_indices.shape[1] * 2 * 3, self.dof))
        colon = slice(None)
        bi = self.boundary_indices
        xlowb = bi[0] == -1
        xhighb = bi[0] == self.Nx
        ylowb = bi[1] == -1
        yhighb = bi[1] == self.Ny
        zlowb = bi[2] == -1
        zhighb = bi[2] == self.Nz
        xlow = self._get_index(0, colon, colon).reshape(3, -1)
        xhigh = self._get_index(-1, colon, colon).reshape(3, -1)
        ylow = self._get_index(colon, 0, colon).reshape(3, -1)
        yhigh = self._get_index(colon, -1, colon).reshape(3, -1)
        zlow = self._get_index(colon, colon, 0).reshape(3, -1)
        zhigh = self._get_index(colon, colon, -1).reshape(3, -1)
        toSides = [bi[:, xlowb & ~ylowb & ~yhighb & ~zlowb & ~zhighb],
                   bi[:, xhighb & ~ylowb & ~yhighb & ~zlowb & ~zhighb],
                   bi[:, ylowb & ~xlowb & ~xhighb & ~zlowb & ~zhighb],
                   bi[:, yhighb & ~xlowb & ~xhighb & ~zlowb & ~zhighb],
                   bi[:, zlowb & ~xlowb & ~xhighb & ~ylowb & ~yhighb],
                   bi[:, zhighb & ~xlowb & ~xhighb & ~ylowb & ~yhighb]]
        fromSides = [xhigh, xlow, yhigh, ylow, zhigh, zlow]
        for i in range(len(toSides)):
            for f in [Field.E, Field.B]:
                for comp in [Comp.x, Comp.y, Comp.z]:
                    self._add_border_eq(R, toSides[i], f, comp, fromSides[i], f, comp, 1)
        return R

    # endregion

    # region ravel stuff

    def _unique_boundary_integer(self, posIndices: ndarray):
        return posIndices[0] + posIndices[1] * (self.Nx + 2) + posIndices[2] * (self.Nx + 2) * (self.Ny + 2)

    def _ravelBIndices(self, posIndices: ndarray, field: Field, component: Comp):
        if field == Field.J:
            raise ValueError("J is not stored in the boundary vector as it is always zero in the boundary!")
        unique_b_ints = self._unique_boundary_integer(self.boundary_indices)
        unique_arg_ints = self._unique_boundary_integer(posIndices)
        # noinspection PyTypeChecker
        return (np.searchsorted(unique_b_ints, unique_arg_ints, sorter=np.argsort(unique_b_ints)) * 6
                + field.value * 3 + component.value)

    # endregion

    # region distance-Index helpers and similar

    def _handle_single_key(self, key) -> Key:
        if isinstance(key, ndarray):
            return self.handle_distance(key)  # maybe this won't be identical to float|int at some point. don't care rn.
        elif isinstance(key, slice):
            return self._handle_slice(key)
        elif isinstance(key, float | int):
            dist = self.handle_distance(key)
            if dist < 0:
                return slice(dist, dist - 1, -1)
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
        #  approaches etc.)
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

    def time(self):
        return self.t * self.dt
    # endregion
