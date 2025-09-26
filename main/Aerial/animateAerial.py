from fdtd_fun import Grid, Detector
from fdtd_fun.detector import Detectable
from fdtd_fun.visualisation import animate
from main.Aerial.aerialConf import size

grid = Grid.load_from_file("aerial.dat")
grid[:,:,0] = Detector("zplane", [Detectable.E,Detectable.B])
animate(grid,10,10)