import fdtd_fun as fdtd
from fdtd_fun import Grid
from fdtd_fun.constants import Field
from fdtd_fun.visualisation.matplotlib_helpers import animate

grid = Grid.load_from_file("main/slabMiddle.dat")
grid[:, :, 20] = fdtd.Detector("Section", {Field.E: True,
                                                   Field.rho: True, Field.J: True, Field.H:True})
animate(grid)
