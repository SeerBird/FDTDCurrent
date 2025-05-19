from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Callable, Sequence

from manim.mobject.vector_field import DEFAULT_SCALAR_FIELD_COLORS

from fdtd_fun.constants import Field

if TYPE_CHECKING:
    from fdtd_fun import Grid, Detector, Conductor, Source

from manim import *


class GridScene(ThreeDScene):
    def __init__(self, grid: Grid, camera_pos, camera_dir):
        super().__init__()
        self._grid = grid

    def construct(self):
        self.set_camera_orientation(phi=85 * DEGREES, theta=60 * DEGREES, zoom=0.8)
        grid = self._grid
        grid.load_next_frame()
        for _, cond in grid.conductors.items():
            pass  # set up stationary bounding box or smth
        for _, src in grid.sources.items():
            pass  # maybe a bounding box as well?
        detectors = {}
        for name, det in grid.detectors.items():
            pos = grid.positions(det.x, det.y, det.z) # shape = (3,...)
            pos = np.moveaxis(pos, 0, -1) # shape = (...,3)
            itershape = pos.shape[:-1]
            fields = {}
            detectors[name] = (itershape, pos, fields)
            if det.E is not None:
                field = AnyVectorField()
                # TODO: OH MY GOD THIS IS SO HORRIBLE DO SOMETHINGGGG. the indexinggg.. moveaxis...
                for index in np.ndindex(itershape):
                    field.add(field.get_vector(pos[index], np.moveaxis(det.E, 0, -1)[index]))
                fields[Field.E] = field
                self.add(field)
            if det.rho is not None:
                field = ScalarField()
                index: tuple
                for index in np.ndindex(itershape):
                    field.add(field.get_point(pos[index],det.rho[index]))
                fields[Field.rho] = field
                self.add(field)

                # fade in or smth
            # repeat for H, J, rho(gotta make the scalar field class)
        self.wait(1 / 15)
        while True:
            if self._grid.load_next_frame():
                break
            for name, det in grid.detectors.items():
                itershape = detectors[name][0]
                pos = detectors[name][1]
                fields = detectors[name][2]
                if det.E is not None:
                    nextField = AnyVectorField()
                    index: tuple
                    for index in np.ndindex(itershape):
                        nextField.add(nextField.get_vector(pos[index], np.moveaxis(det.E, 0, -1)[index]))
                    self.remove(fields[Field.E])
                    self.add(nextField)
                    fields[Field.E] = nextField
                if det.rho is not None:
                    nextField = ScalarField()
                    index: tuple
                    for index in np.ndindex(itershape):
                        nextField.add(nextField.get_point(pos[index],det.rho[index]))
                    self.remove(fields[Field.rho])
                    self.add(nextField)
                    fields[Field.rho] = nextField
            self.wait(1 / 15)


class AnyVectorField(VGroup):
    # TODO: this is a mess. manim is ugly and I copied it and this is ugly
    def __init__(self,
                 length_func: Callable[[float], float] = lambda norm: 1.0 / (1 + np.exp(-norm)),
                 min_vec_norm: float = 0,
                 max_vec_norm: float = 1, #TODO: figure out how to input/calculate the range from the scene constructor
                 colors=None,  # has a default
                 color_scheme: Callable[[np.ndarray], float] | None = None,
                 vector_config: dict | None = None,
                 **kwargs: Any):
        super().__init__([], **kwargs)
        if colors is None:
            colors = DEFAULT_SCALAR_FIELD_COLORS
        if color_scheme is None:
            def color_scheme(p):
                return np.linalg.norm(p)
        self.color_scheme = color_scheme
        self.rgbs = np.array(list(map(color_to_rgb, colors)))
        self.length_func = length_func

        def value_to_rgb(value: np.ndarray):
            color_value = np.clip(
                self.color_scheme(value),
                min_vec_norm,
                max_vec_norm,
            )
            alpha = inverse_interpolate(
                min_vec_norm,
                max_vec_norm,
                color_value,
            )
            alpha *= len(self.rgbs) - 1
            c1 = self.rgbs[int(alpha)]
            c2 = self.rgbs[min(int(alpha + 1), len(self.rgbs) - 1)]
            alpha %= 1
            return (1 - alpha) * c1 + alpha * c2

        self.value_to_rgb = value_to_rgb
        if vector_config is None:
            vector_config = {}
        self.vector_config = vector_config

    def get_vector(self, pos: np.ndarray, value: np.ndarray):
        """"""
        norm: float = float(np.sqrt(value[0] ** 2 + value[1] ** 2 + value[2] ** 2))
        if norm != 0:
            value *= self.length_func(norm) / norm
        vect = Vector(value, **self.vector_config)
        vect.shift(pos)
        # vect = Arrow3D(pos,pos+value, **self.vector_config) #TODO: make this Arrow3D without the computer killing itself
        vect.set_color(ManimColor.from_rgb(self.value_to_rgb(value)))
        return vect


class ScalarField(VGroup):
    def __init__(self, size_func: Callable[[float], float] = lambda norm: 0.2 / (1 + np.exp(-norm)),
                 min_value: float = 0,
                 max_value: float = 1,
                 colors=None,  # has a default
                 color_scheme: Callable[[float], float] | None = None, **kwargs: Any):
        super().__init__([], **kwargs)
        if colors is None:
            colors = DEFAULT_SCALAR_FIELD_COLORS
        if color_scheme is None:
            def color_scheme(p:float):
                return p
        self.color_scheme = color_scheme
        self.rgbs = np.array(list(map(color_to_rgb, colors)))
        self.size_func = size_func

        def value_to_rgb(value: float):
            color_value = np.clip(
                self.color_scheme(value),
                min_value,
                max_value,
            )
            alpha = inverse_interpolate(
                min_value,
                max_value,
                color_value,
            )
            alpha *= len(self.rgbs) - 1
            c1 = self.rgbs[int(alpha)]
            c2 = self.rgbs[min(int(alpha + 1), len(self.rgbs) - 1)]
            alpha %= 1
            return (1 - alpha) * c1 + alpha * c2

        self.value_to_rgb = value_to_rgb

    def get_point(self, pos: np.ndarray, value: float):
        value = self.size_func(value)
        point = Cube(value)
        point.shift(pos)
        point.set_color(ManimColor.from_rgb(self.value_to_rgb(value)))
        return point
