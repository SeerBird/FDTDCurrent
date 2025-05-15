from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import constants as const

if TYPE_CHECKING:
    from fdtd_fun import Grid
    from numpy import ndarray
    from fdtd_fun.typing_ import Index

from .grid_object import GridObject


def _J_cross_H(J: ndarray, H: ndarray):
    # TODO: figure out how to cross an E-type field and a B-type field
    cross = np.empty_like(J)  # make sure all values are set when using empty_like
    cross[0] = J[1] * H[2] - J[2] * H[1]
    cross[1] = J[2] * H[0] - J[0] * H[2]
    cross[2] = J[0] * H[1] - J[1] * H[0]
    return cross


class Conductor(GridObject):

    def __init__(self, name: str, rho_f: float, s: float, sigma: float):
        """

        :param name: the conductor's name... yeah...
        :param rho_f: free charge density of the conductor -
         the charge density of the charge carriers when the total charge density is zero
        :param s: the specific charge of the charge carriers
        :param sigma: the conductivity of the conductor
        """
        super().__init__(name)
        self.sigma = sigma
        self.rho_f = rho_f
        self.s = s

    def _validate_position(self, x: Index, y: Index, z: Index):
        pass

    def _update_J_and_rho(self) -> None:
        # TODO: consider moving everything to do with J and rho into the conductor code and
        #  splitting the field arrays into per-conductor arrays
        E = self._grid.E[:, self.x, self.y, self.z]
        H = self._grid.H[:, self.x, self.y, self.z]
        J = self._grid.J[:, self.x, self.y, self.z]
        self._grid.J[:, self.x, self.y, self.z] += self._grid.dt * (
                self.s * (self.rho_f * E + const.mu_0 * _J_cross_H(J, H) - self.rho_f / self.sigma * J))
