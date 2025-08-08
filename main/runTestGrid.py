import numpy as np, fdtd_fun as fdtd
from numpy import ndarray, dtype, bool

from fdtd_fun.grid import State

def empty_starting_rho(r) -> np.ndarray:
    return np.zeros_like(r[0])

save_path = "main/"  # TODO: figure out if this is okay. since it's not relative. depending on the way this is run

grid = fdtd.Grid("testGrid", (9, 10, 11), 0.1)
def trigger():
    pass  # do smth before every tick


grid.run(empty_starting_rho, 200, save_path,
         trigger)
