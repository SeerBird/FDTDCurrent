from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
from . import constants as const

if TYPE_CHECKING:
    from fdtd_fun import Grid
    from numpy import ndarray
    from fdtd_fun.typing_ import Index

from .grid_object import GridObject


def _E_cross_H(E: ndarray, H: ndarray):
    # TODO: figure out how to cross and dot an E-type field and a B-type field
    cross = np.empty_like(E)  # make sure all values are set when using empty_like
    cross[0] = E[1] * H[2] - E[2] * H[1]
    cross[1] = E[2] * H[0] - E[0] * H[2]
    cross[2] = E[0] * H[1] - E[1] * H[0]
    return cross


def _E_dot_H(E: ndarray, H: ndarray):
    return E[0] * H[0] + E[1] * H[1] + E[2] * H[2]  # TODO: fix this!


def _H_dot_H(H: ndarray):
    return H[0] ** 2 + H[1] ** 2 + H[2] ** 2


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
        if sigma <= 0:
            raise ValueError(f"Conductivity must be positive, {sigma} provided")
        if s == 0:
            raise ValueError(f"Specific charge must be non-zero")
        self.sigma = sigma
        self.rho_f = rho_f
        self.s = s

    def _validate_position(self, x: Index, y: Index, z: Index):
        pass

    def _get_J(self) -> None:
        E = self._grid.E[:, self.x, self.y, self.z] + self._grid.emf[:, self.x, self.y, self.z]
        H = (self._grid.B[:, self.x, self.y, self.z] + self._grid.lastH[:, self.x, self.y, self.z]) / 2
        J = self._grid.J[:, self.x, self.y, self.z]
        self._grid.J[:, self.x, self.y, self.z] += self._grid.dt * (
                self.s * (self.rho_f * E + const.mu_0 * _E_cross_H(J, H) - self.rho_f / self.sigma * J))
        '''
        self._grid.J[:, self.x, self.y, self.z] = ((sigma * rho ** 2 * E +
                                                    const.mu_0 * sigma ** 2 * rho * _E_cross_H(E, H) +
                                                    const.mu_0 ** 2 * sigma ** 3 * _E_dot_H(E, H) * H) /
                                                   (rho ** 2 + sigma ** 2 * _H_dot_H(H)))'''
        #self._grid.J[:, self.x, self.y, self.z] = self.sigma * E
