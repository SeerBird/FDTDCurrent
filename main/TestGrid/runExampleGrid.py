import numpy as np

from fdtdcurrent import Grid, Conductor, Source, Detector
from fdtdcurrent.detector import Detectable


# time-independent emf of 1 V/m in the +x direction
def my_emf(positions: np.ndarray, time: float):
    res = np.zeros_like(positions, float)
    res[0] = 1
    return res


# the free charge density, charge carrier specific charge, and conductivity of copper in SI
copper_rho_s_sigma = (8.5e28 * -1.6e-19, -1.6e-19 / 9e-31, 8e7)
size = 1.0  # size of the grid in meters
thickness = size / 4
grid = Grid("ExampleGrid", (size, size, size), 5e-2)
# region conductor slab in the middle x plane
xslice = slice(size / 2 - thickness / 2, size / 2 + thickness / 2)
grid[xslice, :, :] = Conductor("testConductor1", *copper_rho_s_sigma)
# endregion
# apply my_emf in the 1st quadrant (y>size/2, z>size/2) of the conductor slab
grid[xslice, size / 2:, size / 2:] = Source("testSource", my_emf)
# add a detector to the middle x plane, detecting the x component of the electric field
det = Detector("det1", [Detectable.Ex])
grid[size / 2, :, :] = det


def trigger():
    data = det.values
    pass  # do something with the values read by the detector while the grid is running if you want


grid.run(200, save=True)  # run the grid for 200 steps, saving the state at each step to a file
