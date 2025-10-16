from fdtdcurrent import Grid, Detector
from fdtdcurrent.detector import Detectable
from fdtdcurrent.visualization import animate
from main.Aerial.aerialConf import size, nsteps

detecting = Detectable.divB
grid = Grid.load_from_file("aerial.dat")
grid[1,1:-1,1:-1] = Detector("lowx", [detecting])
grid[1:-1,1,1:-1] = Detector("lowy", [detecting])
grid[1:-1,1:-1,1] = Detector("lowz", [detecting])
grid[size//2,1:-1,1:-1] = Detector("midx", [detecting])
grid[1:-1,size//2,1:-1] = Detector("midy", [detecting])
grid[1:-1,1:-1,size//2] = Detector("midz", [detecting])
grid[-2,1:-1,1:-1] = Detector("highx", [detecting])
grid[1:-1,-2,1:-1] = Detector("highy", [detecting])
grid[1:-1,1:-1,-2] = Detector("highz", [detecting])
fps = 10
animate(grid,nsteps/fps,fps, preferredRatio=0.8)