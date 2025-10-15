import numpy as np
from numpy import ndarray

from fdtdcurrent import Grid, Conductor, Source, Side, BoundaryType
from main.util import perrycioc

perryargs = (0,1,30e-10,2e9)
def my_emf(positions: ndarray, time: float):
    res = np.zeros_like(positions, float)
    res[2] = perrycioc(*perryargs,time) - perrycioc(*perryargs,0)
    return res


size = 1.0
mid = size / 4
radius = size / 8
grid = Grid("testGrid", (21, 21, 21), 5e-2)
# region conductor loop in the x plane
xslice = slice(size / 2 - radius, size / 2 + radius)
grid[xslice, mid - radius:mid + radius, mid - radius:size - mid + radius] \
    = Conductor("testConductor1", 8.5e28 * -1.6e-19, -1.6e-19 / 9e-31, 8e7)
grid[xslice, size - mid - radius:size - mid + radius, mid - radius:size - mid + radius] \
    = Conductor("testConductor2", 8.5e28 * -1.6e-19, -1.6e-19 / 9e-31, 8e7)
grid[xslice, mid + radius:size - mid - radius, size - mid - radius:size - mid + radius] \
    = Conductor("testConductor3", 8.5e28 * -1.6e-19, -1.6e-19 / 9e-31, 8e7)
grid[xslice, mid + radius:size - mid - radius, mid - radius:mid + radius] \
    = Conductor("testConductor4", 8.5e28 * -1.6e-19, -1.6e-19 / 9e-31, 8e7)
# endregion
grid[xslice,mid-radius:mid+radius,size/2-radius:size/2+radius] = Source("testSource", my_emf)
grid.run(200, save = True)
