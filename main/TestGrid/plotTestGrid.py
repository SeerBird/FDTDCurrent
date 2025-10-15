from fdtdcurrent import Grid, Detector, Field
from fdtdcurrent.detector import Detectable
from fdtdcurrent.visualization import animate

grid = Grid.load_from_file("testGrid.dat")
grid[0.5, :, :] = Detector("Section", [Detectable.E, Detectable.B, Detectable.J, Detectable.Ex])
grid[:, 0.5, 0.5] = Detector("Core", [Detectable.V])
grid[0.5, 1:-1, 1:-1] = Detector("Divx", [Detectable.divB])
grid[1:-1, 0.5, 1:-1] = Detector("Divy", [Detectable.divB])
grid[1:-1, 1:-1, 0.5] = Detector("Divz", [Detectable.divB])

animate(grid, time=120, fps=1)
