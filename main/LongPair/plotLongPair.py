from fdtd_fun import Grid,Detector, Field
from fdtd_fun.visualisation import animate
d:int = 3
grid = Grid.load_from_file("longPair.dat")
grid[:, (d*3)//2, :] = Detector("Section", [Field.J])
grid[d:grid.Nx-d,d*3//2,d*3//2] = Detector("Wire1", [Field.V, Field.E])
grid[d:grid.Nx-d,d*3//2,d*7//2] = Detector("Wire2", [Field.V, Field.E])
animate(grid)
