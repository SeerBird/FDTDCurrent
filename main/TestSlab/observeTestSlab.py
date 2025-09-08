from matplotlib import pyplot as plt

from fdtd_fun import Grid, Detector
from fdtd_fun.detector import Detectable
from fdtd_fun.visualisation import animate
from main.TestSlab.runTestSlab import runSlab, T

Jz,t = runSlab(T/200,100,20,(0,0,0))
grid = Grid.load_from_file("testSlab.dat")
grid[:,:,10] = Detector("Slice0",[Detectable.Jz, Detectable.Bz, Detectable.Bx, Detectable.By, Detectable.Ez])
animate(grid,time = 20,fps = 1, show =True)
plt.plot(t,Jz)
plt.show()