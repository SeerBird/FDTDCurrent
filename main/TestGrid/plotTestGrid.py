from fdtd_fun import Grid,Detector, Field
from fdtd_fun.detector import Observable
from fdtd_fun.visualisation import animate

grid = Grid.load_from_file("testGrid.dat")
grid[0.5, :, :] = Detector("Section", [Observable.E,Observable.B,Observable.J,Observable.Ex])
grid[:,0.5,0.5] = Detector("Core", [Observable.V])
animate(grid, time = 4, fps = 30)
