import fdtd_fun as fdtd
from fdtd_fun import Grid
from fdtd_fun.grid import Field
from fdtd_fun.visualisation.matplotlib_helpers import animate

grid = Grid.load_from_file("testGrid.dat")
grid[0.5, :, :] = fdtd.Detector("Section", {Field.E: True, Field.J:True, Field.B:True})
grid[:,0.5,0.5] = fdtd.Detector("Core", {Field.B:True})
animate(grid)
