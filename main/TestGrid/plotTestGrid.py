from fdtd_fun import Grid,Detector, Field
from fdtd_fun.visualisation import animate

grid = Grid.load_from_file("testGrid.dat")
grid[0.5, :, :] = Detector("Section", [Field.E,Field.J,Field.B])
grid[:,0.5,0.5] = Detector("Core", [Field.V])
animate(grid)
