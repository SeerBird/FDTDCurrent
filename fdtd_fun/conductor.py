from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

import numpy as np
from . import constants as const

if TYPE_CHECKING:
    from fdtd_fun.typing_ import Key

from .grid_object import GridObject


class Conductor(GridObject):

    def __init__(self, name: str, rho_f: float, s: float, sigma: float):
        """
        The Conductor GridObject that allows to define subspaces of the grid where charges can move.
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
        self.sigma = sigma #TODO: switch this to resistivity to make everything cleaner
        self.rho_f = rho_f
        self.s = s

    def _validate_position(self, x: Key, y: Key, z: Key):
        pass #TODO: make sure all positions provided are unique
