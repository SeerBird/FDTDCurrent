from matplotlib import pyplot as plt

from fdtd_fun import Grid, Detector
from fdtd_fun.detector import Detectable
from fdtd_fun.visualization import animate
from main.TestSlab.runTestSlab import runSlab, T

runSlab(T/20,100,20,None)
grid = Grid.load_from_file("testSlab.dat")
grid[:,:,-1] = Detector("High z plane", [Detectable.Ex,Detectable.Ey,Detectable.Ez])
grid[:,-1,:]= Detector("High y plane", [Detectable.Ex,Detectable.Ey,Detectable.Ez])
grid[-1,:,:] = Detector("High x plane", [Detectable.Ex,Detectable.Ey,Detectable.Ez])
animate(grid,time = 10,fps = 10, show =True, save = False)