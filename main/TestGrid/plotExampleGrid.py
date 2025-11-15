from fdtdcurrent import Grid, Detector
from fdtdcurrent.detector import Detectable
from fdtdcurrent.visualization import animate

grid = Grid.load_from_file("ExampleGrid.dat")
# the detector we added before running the grid gets restored when loading it from a file

grid[1,1:-1,1:-1] = Detector("lowx", [Detectable.Bx]) # pick a subset of the grid to detect
animate(grid, time=120, fps=1) # produce an animation of the detector values on the grid that
# lasts 120 seconds with 1 frame per second
