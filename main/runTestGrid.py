import numpy as np, fdtd_fun as fdtd
from numpy import ndarray, dtype, bool

from fdtd_fun.grid import State

# region func defs whatevs
# here we're pretending to have an example of how the library will be used
def my_sphere(r) -> ndarray[tuple[int, ...], dtype[bool]]:
    return r[0] ^ 2 + r[1] ^ 2 + r[2] ^ 2 < 3


def my_plotter(s):
    pass  # what a plotter


def my_starting_rho(r) -> np.ndarray:
    return np.zeros_like(r[0])


def sphere(theta, phi, r):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


def boogy_woogy(r: np.ndarray, t: float) -> ndarray:
    phase = r[0] ** 2 + r[1] ** 2 + r[2] ** 2
    return np.asarray((np.cos(phase + t), np.sin(phase + t), np.cos(phase + t) + np.sin(phase + t)))


def uniform(r: np.ndarray, t: float) -> ndarray:
    return np.asarray((np.zeros_like(r[0]) - 1000, np.zeros_like(r[0]), np.zeros_like(r[0])))


# endregion
save_path = "main/"  # gotta correct the pathing at some point cuz this is ugly
u = np.linspace(0, np.pi / 2, 10)
v = np.linspace(0, np.pi / 2, 10)
params = np.meshgrid(u, v)
octant_sphere = sphere(params[0], params[1], 2.0)

grid = fdtd.Grid("testGrid", (40, 40, 40), 0.1)
# assign boundaries
grid[0:30,0:10,15:25] = fdtd.Conductor("1", 10**10, 10**-19, 10**8)
grid[30:,0:30,15:25] = fdtd.Conductor("2", 10**10, 10**-19, 10**8)
grid[10:,30:,15:25] = fdtd.Conductor("3", 10**10, 10**-19, 10**8)
grid[0:10,10:,15:25] = fdtd.Conductor("4", 10**10, 10**-19, 10**8)
grid[5:25, 0:10, 15:25] = fdtd.Source("boogy", uniform)


# add sources
def trigger():
    pass  # do smth before every tick


grid.run(my_starting_rho, 200, save_path,
         trigger)  # grid runs, the only interaction with any outside code is through the
# trigger() function.
