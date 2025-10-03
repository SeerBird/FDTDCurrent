import numpy as np

from fdtdcurrent import *
from fdtdcurrent.detector import Detectable
from fdtdcurrent.visualization import animate
from main.Aerial.aerialConf import size
from main.util import gaussian, copper_rho_s_sigma

sigma = 1e-9


def my_emf(positions: np.ndarray, time: float):
    res = np.zeros_like(positions, float)
    res[2] = gaussian(sigma * 4, sigma, 1.0, time) - gaussian(sigma * 4, sigma, 1.0, 0)
    return res


mid = size // 2
grid = Grid("aerial", (size, size, 1), dt=sigma)
grid[mid - 1:mid + 1, 0:3, :] = Conductor("testConductor1", *copper_rho_s_sigma)
grid[mid - 1:mid + 1, 0:3, :] = Source("testSource", my_emf)
grid.run(1000, save=True, boundary={
    Side.xlow: BoundaryType.wrap,
    Side.xhigh: BoundaryType.wrap,
    Side.ylow: BoundaryType.wrap,
    Side.yhigh: BoundaryType.wrap,
    Side.zlow: BoundaryType.wrap,
    Side.zhigh: BoundaryType.wrap})
