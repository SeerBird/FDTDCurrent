import numpy as np

from fdtdcurrent import *
from fdtdcurrent.detector import Detectable
from fdtdcurrent.visualization import animate
from main.Aerial.aerialConf import size
from main.util import gaussian, copper_rho_s_sigma

sigma = 1e-9


def my_emf(positions: np.ndarray, time: float):
    res = np.zeros_like(positions, float)
    res[0] = gaussian(sigma * 4, sigma, 1.0, time) - gaussian(sigma * 4, sigma, 1.0, 0)
    return res


mid = size // 2
grid = Grid("aerial", (size, size, size), dt=sigma)
grid[0:3, mid - 1:mid + 1, mid - 1:mid + 1] = Conductor("testConductor1", *copper_rho_s_sigma)
grid[0:3, mid - 1:mid + 1, mid - 1:mid + 1] = Source("testSource", my_emf)
grid.run(100, save=True)
