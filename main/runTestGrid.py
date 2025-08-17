import numpy as np, fdtd_fun as fdtd
from numpy import ndarray, dtype, bool

from fdtd_fun import Conductor
from fdtd_fun import Source
from fdtd_fun.grid import State


def empty_starting_rho(r) -> np.ndarray:
    return np.zeros_like(r[0])


save_path = "main/"  # TODO: figure out if this is okay. since it's not relative. depending on the way this is run


def my_emf(positions: ndarray, time: float):
    res = np.zeros_like(positions, float)
    res[2] = 1
    return res


size = 1.0
mid = size / 4
radius = size / 8
grid = fdtd.Grid("testGrid", (1.0, 1.0, 1.0), 4e-2)
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


def trigger():
    pass  # do smth before every tick


grid.run(empty_starting_rho, 200, save_path,
         trigger)
