from functools import partial

import numpy as np
from numpy import ndarray

from fdtd_fun import Grid, Conductor, Source
from main.util import perrycioc, copper_rho_s_sigma
import matplotlib.pyplot as plt


def my_emf(positions: ndarray, time: float):
    res = np.zeros_like(positions, float)
    func = partial(perrycioc, 0, 1, 6e-10, 2e10, )
    res[2] = func(time) - func(0)
    return res



testtime = np.linspace(0,40e-10,5000)
plt.plot(testtime, perrycioc(0,1,6e-10,1e9,testtime))
plt.show()

d: int = 3
size = d * 5
grid = Grid("longPair", (60, d * 3, size), 1e-2)
# region experiment setup (baddddd resolution)
y = slice(d, 2 * d)
xslice = slice(d, grid.Nx - d)
grid[xslice, y, d:2 * d] = Conductor("wire1", *copper_rho_s_sigma)
grid[xslice, y, 3 * d:4 * d] = Conductor("wire2", *copper_rho_s_sigma)
grid[d:2 * d, y, 2 * d:3 * d] = Conductor("sourceWire", *copper_rho_s_sigma)
# endregion
grid[d:2 * d, y, 2 * d:3 * d] = Source("testSource", my_emf)
grid.run(400, save=True)
