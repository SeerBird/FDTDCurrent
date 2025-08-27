from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Callable, Sequence

import numpy as np
from matplotlib import pyplot as plt, animation

from fdtd_fun.typing_ import Field, Comp

if TYPE_CHECKING:
    from fdtd_fun import Grid

fieldsToCheck = [Field.E, Field.B, Field.J, Field.V]
fieldNames = {Field.E: "E", Field.B: "B", Field.J: "J", Field.V:"V"}
components = [Comp.x, Comp.y, Comp.z]
compNames = {Comp.x: "x", Comp.y: "y", Comp.z: "z"}
fieldColors = {Field.E: "b", Field.B: "r", Field.J: "o", Field.V:"y"}


def animate(grid: Grid, time: float = 4.0, preferredRatio: float = 0.7):
    fig = plt.figure()
    ims = []
    detectors = {}
    total: int = 0
    frames = 0
    names: list[str] = []
    # region set up data template
    for name, det in grid.detectors.items():
        det.read()
        pos: np.ndarray = grid[det.x, det.y, det.z] * grid.ds  # shape = (3,...)
        shapeList = list(pos.shape[1:])
        while True:
            try:
                shapeList.remove(1)
            except ValueError:
                break
        fields = {}
        detectors[det.name] = (pos, fields, shapeList)
        if det.E is not None:
            fields[Field.E] = True
            names.append(f"{name}:E")
            names.append(f"{name}:Ex")
            names.append(f"{name}:Ey")
            names.append(f"{name}:Ez")
            total += 4
        if det.B is not None:
            fields[Field.B] = True
            names.append(f"{name}:B")
            names.append(f"{name}:Bx")
            names.append(f"{name}:By")
            names.append(f"{name}:Bz")
            total += 4
        if det.J is not None:
            fields[Field.J] = True
            names.append(f"{name}:J")
            names.append(f"{name}:Jx")
            names.append(f"{name}:Jy")
            names.append(f"{name}:Jz")
            total += 4
        if det.V is not None:
            fields[Field.V] = True
            names.append(f"{name}:V")
            total+=1

    # endregion
    # region get layout
    xn: int = 1
    yn: int = 1
    while xn * yn < total:
        if np.abs(np.log((xn + 1) / yn / preferredRatio)) < np.abs(np.log(xn / (yn + 1) / preferredRatio)):
            xn += 1
        else:
            yn += 1
    for j in range(total):
        fig.add_subplot(xn, yn, j + 1)
    ax = fig.get_axes()
    # endregion
    # region set titles and axes
    for i in range(len(names)):
        ax[i].set_title(names[i])
    # endregion
    while True:
        frames += 1
        frameArtists = []
        # region draw
        subplotCounter = 0
        for name, myTuple in detectors.items():
            fields = myTuple[1]
            shapeList = myTuple[2]
            det = grid.detectors[name]
            indexes = np.arange(shapeList[0])
            for f in fieldsToCheck:
                if fields.get(f):
                    field = det.__getattribute__(fieldNames[f])
                    if len(shapeList) == 1:
                        if f==Field.V: #kinda messy, clean up if we get more scalar fields
                            frameArtists += ax[subplotCounter].plot(indexes[:-1],
                                                                    field.reshape(shapeList[0]-1), fieldColors[f])
                            subplotCounter += 1
                            continue
                        frameArtists += ax[subplotCounter].plot(indexes,
                                                                ((field[0] ** 2 + field[1] ** 2 + field[2] ** 2) ** 0.5)
                                                                .reshape(shapeList),fieldColors[f])
                        subplotCounter += 1
                        for comp in components:
                            frameArtists += ax[subplotCounter].plot(indexes, field[comp.value].reshape(shapeList),fieldColors[f])
                            subplotCounter += 1
                    elif len(shapeList) == 2:
                        frameArtists.append(
                            ax[subplotCounter].imshow(((field[0] ** 2 + field[1] ** 2 + field[2] ** 2) ** 0.5)
                                                      .reshape(shapeList),
                                                      cmap="plasma", interpolation="none", origin='lower'))
                        subplotCounter += 1
                        for comp in components:
                            frameArtists.append(ax[subplotCounter].imshow(field[comp.value].reshape(shapeList),
                                                                          cmap="plasma", interpolation="none",
                                                                          origin='lower'))
                            subplotCounter += 1
        # endregion
        fig.tight_layout()
        ims.append(frameArtists)
        if not grid.load_next_frame():
            break

    ani = animation.ArtistAnimation(fig, ims, interval=1, blit=True, repeat_delay=1000)
    fig.tight_layout()
    ani.save(f"ani{grid.name}.mp4", dpi=300, fps=frames / time)
