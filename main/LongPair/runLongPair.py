from functools import partial

import numpy as np
from numpy import ndarray

from fdtd_fun import Grid, Conductor, Source
from main.util import perrycioc, copper_rho_s_sigma, gaussian
import matplotlib.pyplot as plt

func = partial(perrycioc, 0, 3e-9, 4e-8, 1e8, )
func = partial(gaussian,4e-8,2e-9,1)
def my_emf(positions: ndarray, time: float):
    res = np.zeros_like(positions, float)
    res[2] = func(time) - func(0)
    return res



testtime = np.linspace(0,100e-9,5000)
plt.plot(testtime, func(testtime))
plt.show()

d: int = 3
size = d * 5
grid = Grid("longPair", (100, d * 3, size), 4e-2)
# region experiment setup (baddddd resolution)
y = slice(d, 2 * d)
xslice = slice(d, grid.Nx - d)
grid[xslice, y, d:2 * d] = Conductor("wire1", *copper_rho_s_sigma)
grid[xslice, y, 3 * d:4 * d] = Conductor("wire2", *copper_rho_s_sigma)
grid[d:2 * d, y, 2 * d:3 * d] = Conductor("sourceWire", *copper_rho_s_sigma[:-1],5)
grid[grid.Nx-2*d:grid.Nx-d, y, 2 * d:3 * d] = Conductor("endWire", *copper_rho_s_sigma)
# endregion
grid[d:2 * d, y, 2 * d:3 * d] = Source("testSource", my_emf)
grid.run(4000, save=True)
