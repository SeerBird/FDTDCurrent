import fdtd_fun as fdtd
from fdtd_fun import Grid
from fdtd_fun.constants import Field
from fdtd_fun.visualisation.matplotlib_helpers import animate

grid = Grid.load_from_file("main/testGrid.dat")
det1 = fdtd.Detector("Line", {Field.E: False,
                              Field.rho: True})
grid[:, 20, 20] = det1
grid[11:30, 11:30, 20] = fdtd.Detector("Section", {Field.E: True,
                                                   Field.rho: True, Field.J: True})
animate(grid)
