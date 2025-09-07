import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt

from fdtd_fun import Grid, Conductor, Source, constants
from main.util import perrycioc, copper_rho_s_sigma

slope = 1e18
loc = -1e-18
k = 1
def my_emf(positions: ndarray, time: float):
    res = np.zeros_like(positions, float)
    #res[2] = perrycioc(0,k,loc,slope,time)
    res[2] = k
    return res
rho,s,sigma = copper_rho_s_sigma
w = (-(s*rho/2/sigma)**2+s*rho/constants.eps_0)**0.5
Q = s*rho*constants.eps_0/4/sigma**2 # just calling this Q, not checking if it is the quality factor
T = 2*np.pi/w
grid = Grid("testSlab", (5, 5, 5), dt = T/20)
t = np.arange(200)*grid.dt
#plt.plot(t,perrycioc(0,1,loc,slope,t))
#plt.show()
grid[:,:,:] = Conductor("testConductor1", *copper_rho_s_sigma)
grid[:,:,:] = Source("testSource", my_emf)
grid.run(200, save = True)

def prediction(t):
    return np.exp(-s*rho/2/sigma*t)*s*rho*k/w*np.sin(w*t)
grid = Grid.load_from_file("testSlab.dat")
J = []
Jx = []
Jy = []
Jz = []
Ez = []
t = []
while True:
    sub = grid[2,2,2]
    j = sub[2].reshape(3)
    J.append((j[0]**2+j[1]**2+j[2]**2)**0.5)
    Jx.append(j[0])
    Jy.append(j[1])
    Jz.append(j[2])
    Ez.append(sub[0,2].reshape(1))
    t.append(grid.time())
    if not grid.load_next_frame():
        break
t = np.asarray(t)
plt.subplot(1,2,1)
plt.plot(t,Jz,"b", label = "Data")
plt.plot(t,prediction(t),"r", label = "Predicted")
plt.title("Jz")
plt.subplot(1,2,2)
plt.plot(t,Ez)
plt.title("Ez")
plt.tight_layout()
plt.show()

