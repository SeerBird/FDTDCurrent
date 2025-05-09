import numpy as np, fdtd_fun as fdtd
from numpy import ndarray, dtype, bool


def my_sphere(r)-> ndarray[tuple[int, ...], dtype[bool]]:
    return r[0]^2+r[1]^2+r[2]^2<3
def my_plotter(s:fdtd.State):
    pass # what a plotter
def my_starting_rho(r)->np.ndarray:
    return np.zeros_like(r[0])+1
param = np.linspace(0,2,400)
line = np.asarray([1 + param / 2, 2 - param, 3 + param])
grid = fdtd.Grid(30,40,50,1)
grid.add_material(my_sphere,0,0)
grid.add_detector(line, my_plotter)
grid.run(my_starting_rho)
