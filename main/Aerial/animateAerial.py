from fdtdcurrent import Grid, Detector
from fdtdcurrent.detector import Detectable
from fdtdcurrent.visualization import animate
from main.Aerial.aerialConf import size

grid = Grid.load_from_file("aerial.dat")
grid[:,:,0] = Detector("zplane", [Detectable.E,Detectable.B])
grid[size//2,1:-1,1:-1] = Detector("divx", [Detectable.divB])
grid[1:-1,size//2,1:-1] = Detector("divy", [Detectable.divB])
grid[1:-1,1:-1,size//2] = Detector("divz", [Detectable.divB])
animate(grid,100,1)