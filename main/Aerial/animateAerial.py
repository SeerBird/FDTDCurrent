from fdtdcurrent import Grid, Detector
from fdtdcurrent.detector import Detectable
from fdtdcurrent.visualization import animate

grid = Grid.load_from_file("aerial.dat")
grid[:,:,0] = Detector("zplane", [Detectable.E,Detectable.B])
animate(grid,10,10)