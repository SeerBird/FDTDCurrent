from fdtd_fun import Grid,Detector, Field
from fdtd_fun.visualization import animate
d:int = 3
grid = Grid.load_from_file("longPair.dat")
#grid[:, (d*3)//2, :] = Detector("Section", [Field.J])
grid[(d*5)//2:grid.Nx-(d*5)//2,d*3//2,d*3//2] = Detector("Wire1", [Field.V, Field.E])
grid[(d*5)//2:grid.Nx-(d*5)//2,d*3//2,d*7//2] = Detector("Wire2", [Field.V, Field.E])
animate(grid, time = 12.0, fps = 10)
