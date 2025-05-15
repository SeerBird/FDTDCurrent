from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fdtd_fun import Grid, Detector, Material, Source



class Scene:
    detectors: dict[str, Detector]
    materials: dict[str, Material]
    sources: dict[str, Source]

    def __init__(self, grid: Grid, camera_pos, camera_dir):
        pass

    def add_detector(self, det: Detector):
        if self.detectors.__contains__(det.name):
            raise ValueError("This detector is already in this scene")
        self.detectors[det.name] = det
        # maybe more handling is needed

    def add_material(self, mat: Material):
        pass

    def add_source(self, src: Source):
        pass

    def frame(self):  # this is most subject to change as idk how manim works
        # check the Detector/Material/Source/Grid class code to see what you have access to
        # I suggest you keep working in index units instead of meters because most of the values are already indexes
        # but idk
        pass
