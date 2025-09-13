from matplotlib import pyplot as plt

from fdtd_fun import Grid, Detector
from fdtd_fun.detector import Detectable
from fdtd_fun.visualisation import animate
from main.TestSlab.runTestSlab import runSlab, T

runSlab(T/20,100,20,None)
grid = Grid.load_from_file("testSlab.dat")
#grid[:,:,10] = Detector("Slice0",[Detectable.Jz, Detectable.Ez, Detectable.Ex, Detectable.Ey, Detectable.Bz])
#grid[:,10,10] = Detector("Along x", [Detectable.Ex,Detectable.Bx,Detectable.By, Detectable.Bz])
#grid[10,:,10] = Detector("Along y", [Detectable.Ey,Detectable.Bx,Detectable.By, Detectable.Bz])
#grid[10,10,:] = Detector("Along z", [Detectable.Ez,Detectable.Bx,Detectable.By, Detectable.Bz])
grid[0,:,10] = Detector("Line 1", [Detectable.Ex])
grid[:,-1,10]= Detector("Line 2", [Detectable.Ey])
grid[-1,::-1,10] = Detector("Line 3", [Detectable.Ex])
grid[::-1,0,10]= Detector("Line 4", [Detectable.Ey])
animate(grid,time = 10,fps = 10, show =True, save = False)