from fdtdcurrent import Grid, Detector, Field
from fdtdcurrent.detector import Detectable
from fdtdcurrent.visualization import animate

grid = Grid.load_from_file("testGrid.dat")
detecting = Detectable.divB
grid[1,1:-1,1:-1] = Detector("lowx", [detecting])
grid[1:-1,1,1:-1] = Detector("lowy", [detecting])
grid[1:-1,1:-1,1] = Detector("lowz", [detecting])
grid[0.5,1:-1,1:-1] = Detector("midx", [detecting])
grid[1:-1,0.5,1:-1] = Detector("midy", [detecting])
grid[1:-1,1:-1,0.5] = Detector("midz", [detecting])
grid[-2,1:-1,1:-1] = Detector("highx", [detecting])
grid[1:-1,-2,1:-1] = Detector("highy", [detecting])
grid[1:-1,1:-1,-2] = Detector("highz", [detecting])
animate(grid, time=120, fps=1)
