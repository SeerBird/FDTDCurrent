import numpy as np, fdtd_fun as fdtd
from numpy import ndarray, dtype, bool
def uniform(r: np.ndarray, t: float) -> ndarray:
    return np.asarray((np.zeros_like(r[0]) + 1 , np.zeros_like(r[0]), np.zeros_like(r[0])))
def my_starting_rho(r) -> np.ndarray:
    return np.zeros_like(r[0])

# add sources
def trigger():
    pass  # do smth before every tick
grid = fdtd.Grid("slabMiddle", (40, 40, 40), 0.1*10**-20)
# assign boundaries
grid[:,10:30,10:30] = fdtd.Conductor("1", -10**10, -10**12, 10**8)
grid[15:25, 10:30,10:30] = fdtd.Source("boogy", uniform)



grid.run(my_starting_rho, 2000, "main/",
         trigger)  # grid runs, the only interaction with any outside code is through the
# trigger() function.