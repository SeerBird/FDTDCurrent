import numpy as np
from matplotlib import pyplot as plt

from fdtdcurrent import *
from main.Aerial.aerialConf import size, nsteps, signal_resolution
from main.util import gaussian, copper_rho_s_sigma, cos_peak

signal_width = 1e-15 #s
cospeakargs = (signal_width/1.7, signal_width, 1.0,)

def my_emf(positions: np.ndarray, time: float):
    res = np.zeros_like(positions, float)
    res[2] = cos_peak(*cospeakargs, time)
    return res



mid = size // 2
grid = Grid("aerial", (size, size, size), dt= signal_width / signal_resolution)
t = np.arange(nsteps) * grid.dt
plt.plot(t, cos_peak(*cospeakargs, t))
plt.show()
grid[mid - 2:mid + 2,  mid - 2:mid + 2,0:5] = Conductor("testConductor1", *copper_rho_s_sigma)
grid[ mid-1:mid + 1, mid-1:mid + 1,1:3] = Source("testSource", my_emf)
grid.run(nsteps, save=True)
