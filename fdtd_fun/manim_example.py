from manim import *


class VF3D(ThreeDScene):
    def construct(self):
        t = 0
        self.set_camera_orientation(phi=85 * DEGREES, theta=15 * DEGREES, zoom=0.8)  # set camera position

        # This bit sets up the x-y plane at z = 0.
        plane = NumberPlane(x_range=(-7.1, 7.1, 1),
                            y_range=(-4.1, 4.1, 1))  # defines the range of the plane + gridline intervals
        plane.add(plane.get_axis_labels())  # adds axes labels
        self.add(plane)  # adds the plane and axes to the scene

        # This bit defines the vector field.
        # First define a function defining the vector field mathematically - NB RIGHT = +i, UP = +j, OUT = +k. THis bit will need to be replaced by code
        # which extracts the raw data from the simulation and assigns it to positions.
        # NB if we want only to show certain arrows in the vector field, we can set the vectors at positions outside the vector field to 0, and then they shouldn't show up in the simulation.

        func = lambda pos: (np.sin(pos[1] * t / 2)) * RIGHT + (np.sin(pos[0])) ** 2 * UP + (
            np.sin(pos[2] + pos[0] * t ** 2)) * OUT

        # defines vector field function as an arrow vector field, using the func defined above. the range functions can be set to any interval and step, though if the arrows are too close together it is useful to
        # edit the length of the arrows using the length_func argument so that they don't overlap

        vector_field_1 = ArrowVectorField(
            func, x_range=[-6.5, 6.5, 0.5], y_range=[-3.5, 3.5, 0.5], z_range=[-2, 2, 2], length_func=lambda x: x / 2
        )

        # classes the vector field as a group of vectors
        field3D1 = VGroup(*vector_field_1)

        # adds the vector field to the scene
        self.play(Create(field3D1))

        # Optional!!!
        self.begin_ambient_camera_rotation(rate=0.12, about='theta')
        self.begin_ambient_camera_rotation(rate=0.04, about='phi')
        # starts rotating the camera around the plane

        # For loop to iterate over time steps (this is why func has a time component) I assume this will need minor modification, but the rough structure works.
        for t in range(1, 10, 1):
            func = lambda pos: (np.sin(pos[1] * t / 2)) * RIGHT + (np.sin(pos[0])) ** 2 * UP + (
                np.sin(pos[2] + pos[0] * t ** 2)) * OUT
            vector_field_t1 = ArrowVectorField(
                func, x_range=[-6.5, 6.5, 0.5], y_range=[-3.5, 3.5, 0.5], z_range=[-2, 2, 2],
                length_func=lambda x: x / 2
            )
            # defines the vector field at the next time step
            field3D2 = VGroup(*vector_field_t1)
            # defines the vector field as a group of vectors
            self.play(Transform(field3D1, field3D2))
            # transforms the initial field tot he next
            if t < 10:
                self.remove(field3D1)
            field3D1 = field3D2
            # sets and hopefully stores the field as the starting field for the next iteration
        self.wait(1)