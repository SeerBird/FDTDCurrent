from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
from . import constants as const

if TYPE_CHECKING:
    from fdtd_fun import Grid
    from numpy import ndarray
    from fdtd_fun.typing_ import Key

from .grid_object import GridObject


def _E_cross_H(E: ndarray, H: ndarray):
    cross = np.empty_like(E)  # make sure all values are set when using empty_like
    cross[0] = E[1] * H[2] - E[2] * H[1]
    cross[1] = E[2] * H[0] - E[0] * H[2]
    cross[2] = E[0] * H[1] - E[1] * H[0]
    return cross


def _E_dot_H(E: ndarray, H: ndarray):
    return E[0] * H[0] + E[1] * H[1] + E[2] * H[2]


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

    def _validate_position(self, x: Key, y: Key, z: Key):
        pass
