from matplotlib import pyplot as plt

from fdtd_fun import Grid, Detector
from fdtd_fun.detector import Detectable
from fdtd_fun.visualisation import animate
from main.TestSlab.runTestSlab import runSlab, T

size = 6
runSlab(T/20,100,size,None)
grid = Grid.load_from_file("testSlab.dat")
grid[0,:,size//2] = Detector("Line 1", [Detectable.Ex])
grid[:,-1,size//2]= Detector("Line 2", [Detectable.Ey])
grid[-1,::-1,size//2] = Detector("Line 3", [Detectable.Ex])
grid[::-1,0,size//2]= Detector("Line 4", [Detectable.Ey])
animate(grid,time = 10,fps = 10, show =True, save = False)