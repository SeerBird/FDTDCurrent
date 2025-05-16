from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fdtd_fun import Grid, Detector, Conductor, Source

from manim import *



class GridScene(Scene):
    def __init__(self, grid: Grid, camera_pos, camera_dir):
        super().__init__()
        self._grid = grid
        self.detectors: dict[str, Detector] = {}
        self.materials: dict[str, Conductor] = {}
        self.sources: dict[str, Source] = {}


    def construct(self):
        while True:
            if self._grid.load_next_frame():
                break
            # use detectors/values in the grid




