import fdtd_fun as fdtd

from fdtd_fun.constants import Field

newGrid = fdtd.Grid.load_from_file(
    f"main/testGrid.dat")  # we can load the grid from a file. this restores (or should restore)
# all the GridObjects on the grid and sets the state to the initial state
# after this, calling newGrid.load_next_frame() sets the grid to the next state
det1 = fdtd.Detector("bababooie", {Field.E: False,
                                   Field.rho: True})  # new detectors can be added that weren't needed when the sim was running but will
# be needed for the visualisation
newGrid[:, 20, 20] = det1
# manim.config.quality = "low_quality"
scene = fdtd.GridScene(newGrid, None, None)
scene.render()  # scene.render(), among other things, calls scene.construct(), which is the method in which we need
# to repeatedly use the grid state and call grid.load_next_frame()
