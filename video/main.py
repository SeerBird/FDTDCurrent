import manim
import numpy as np, fdtd_fun as fdtd
from numpy import ndarray, dtype, bool

from fdtd_fun.constants import Field
from fdtd_fun.grid import State


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


def boogy_woogy(r: np.ndarray, t: float) -> State:
    phase = r[0] ** 2 + r[1] ** 2 + r[2] ** 2
    return State(np.asarray((np.cos(phase + t), np.sin(phase + t), np.cos(phase + t) + np.sin(phase + t))), None, None,
                 None)


def uniform(r: np.ndarray, t: float) -> State:
    return State(np.asarray((np.zeros_like(r[0]) + 1, np.zeros_like(r[0]), np.zeros_like(r[0]))), None, None, None)


u = np.linspace(0, np.pi / 2, 10)
v = np.linspace(0, np.pi / 2, 10)
params = np.meshgrid(u, v)
octant_sphere = sphere(params[0], params[1], 2.0)
save_path = "video/"  # gotta correct the pathing at some point cuz this is ugly
grid = fdtd.Grid("testGrid", (5.0, 4.0, 3.0), 0.1)
# assign boundaries
grid[0.0:4.0, 2.1:3.0, 0.3:0.4] = fdtd.Conductor("bababoi", 1, 1, 1)
grid[:, :, :] = fdtd.Source("boogy", boogy_woogy)


# add sources
def trigger():
    pass  # do smth before every tick


grid.run(my_starting_rho, 100, save_path,
         trigger)  # grid runs, the only interaction with any outside code is through the
# trigger() function.


newGrid = fdtd.Grid.load_from_file(
    save_path + f"{grid.name}.dat")  # we can load the grid from a file. this restores (or should restore)
# all the GridObjects on the grid and sets the state to the initial state
# after this, calling newGrid.load_next_frame() sets the grid to the next state
det1 = fdtd.Detector("bababooie", {Field.E: True,
                                   Field.rho: True})  # new detectors can be added that weren't needed when the sim was running but will
# be needed for the visualisation
newGrid[octant_sphere] = det1
# manim.config.quality = "low_quality"
scene = fdtd.GridScene(newGrid, None, None)
scene.render()  # scene.render(), among other things, calls scene.construct(), which is the method in which we need
# to repeatedly use the grid state and call grid.load_next_frame()
