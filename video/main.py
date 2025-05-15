import numpy as np, fdtd_fun as fdtd
from numpy import ndarray, dtype, bool


# here we're pretending to have an example of how the library will be used
def my_sphere(r) -> ndarray[tuple[int, ...], dtype[bool]]:
    return r[0] ^ 2 + r[1] ^ 2 + r[2] ^ 2 < 3


def my_plotter(s):
    pass  # what a plotter


def my_starting_rho(r) -> np.ndarray:
    return np.zeros_like(r[0]) + 1


def sphere(theta, phi, r):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


u = np.linspace(0, np.pi / 2, 200)
v = np.linspace(0, np.pi / 2, 200)
params = np.meshgrid(u, v)
octant_sphere = sphere(params[0], params[1], 3.0)

grid = fdtd.Grid((10.0, 20.0, 1.0), 0.01)
# assign boundaries
grid[0.0:4.0, 2.1:16.0, 0.3:0.4] = fdtd.Conductor("bababoi",0)  # slice indexing
det1 = fdtd.Detector("bababooie")
grid[octant_sphere] = det1  # ndarray indexing
# sources??
scene = fdtd.Scene(grid, None, None)
scene.add_detector(det1)
grid.run(my_starting_rho, scene.frame)
