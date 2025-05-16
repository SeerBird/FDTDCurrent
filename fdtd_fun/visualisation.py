from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fdtd_fun import Grid, Detector, Conductor, Source



class Scene:
    def __init__(self, grid: Grid, camera_pos, camera_dir):
        self.detectors: dict[str, Detector] = {}
        self.materials: dict[str, Conductor] = {}
        self.sources: dict[str, Source] = {}

    def add_detector(self, det: Detector):
        if self.detectors.__contains__(det.name):
            raise ValueError("This detector is already in this scene")
        self.detectors[det.name] = det
        # maybe more handling is needed

    def add_material(self, mat: Conductor):
        if self.materials.__contains__(mat.name):
            raise ValueError("This material is already in this scene")
        self.materials[mat.name] = mat
        # maybe more handling is needed

    def add_source(self, src: Source):
        if self.sources.__contains__(src.name):
            raise ValueError("This source is already in this scene")
        self.sources[src.name] = src
        # maybe more handling is neededw

    def frame(self):  # this is most subject to change as idk how manim works
        #TODO: produce a frame from the current grid state
        # unless we can't do frames and need to use manim to somehow interpolate idk
        # check the Detector/Material/Source/Grid class code to see what you have access to
        # I suggest you keep working in index units instead of meters because most of the values are already indexes
        # but idk
        # honestly we just need to talk ab how manim works
        # if you can make helper methods to get mobjects like a cuboid/mesh/whatever way we cen represent the
        # materials/sources/detectors that would be good
        pass
