from fdtd_fun import Grid,Detector, Field
from fdtd_fun.detector import Detectable
from fdtd_fun.visualization import animate

grid = Grid.load_from_file("testGrid.dat")
grid[0.5, :, :] = Detector("Section", [Detectable.E, Detectable.B, Detectable.J, Detectable.Ex])
grid[:,0.5,0.5] = Detector("Core", [Detectable.V])
animate(grid, time = 4, fps = 30)
