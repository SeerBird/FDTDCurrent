import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt

from fdtd_fun import Grid, Conductor, Source, constants
from main.util import perrycioc, copper_rho_s_sigma

slope = 1e18
loc = -1e-18
k = 1
rho, s, sigma = copper_rho_s_sigma
w = (-(s * rho / 2 / sigma) ** 2 + s * rho / constants.eps_0) ** 0.5
Q = s * rho * constants.eps_0 / 4 / sigma ** 2  # just calling this Q, not checking if it is the quality factor
T = 2 * np.pi / w

def my_emf(positions: ndarray, time: float):
    res = np.zeros_like(positions, float)
    #res[2] = perrycioc(0,k,loc,slope,time)
    res[2] = k
    return res

def runSlab(dt:float, steps:int|float, size:int, pos: tuple[int,int,int]):
    grid = Grid("testSlab", (size,size,size), dt = dt)
    grid[:,:,:] = Conductor("testConductor1", *copper_rho_s_sigma)
    grid[:,:,:] = Source("testSource", my_emf)
    Jz = []
    t = []
    def trigger():
        sub = grid[pos]
        j = sub[2].reshape(3)
        Jz.append(j[2])
        t.append(grid.time())
    grid.run(steps, save = True, trigger = trigger)
    return np.asarray(Jz), np.asarray(t)





