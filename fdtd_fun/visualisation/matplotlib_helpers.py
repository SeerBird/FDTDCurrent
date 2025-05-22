from __future__ import annotations
import matplotlib

from typing import TYPE_CHECKING, Any, Iterable, Callable, Sequence

import numpy as np
from matplotlib import pyplot as plt, animation
from matplotlib.axes import Axes

from fdtd_fun.constants import Field

if TYPE_CHECKING:
    from fdtd_fun import Grid, Detector, Conductor, Source


def animate(grid: Grid, preferredRatio: float = 1.0):
    ims = []
    grid.load_next_frame()
    ds = grid.ds
    detectors = {}
    total: int = 0
    names: list[str] = []
    for name, det in grid.detectors.items():
        pos: np.ndarray = grid.positions(det.x, det.y, det.z)  # shape = (3,...)
        shapeList = list(pos.shape[1:])
        indexes = []
        for length in shapeList:
            indexes.append(np.arange(length))
        indexes.append("ij")
        while True:
            try:
                shapeList.remove(1)
            except ValueError:
                break
        fields = {}
        detectors[det.name] = (pos, fields, shapeList)
        if det.rho is not None:
            fields[Field.rho] = True
            names.append(f"{name}:rho")
            total += 1
        if det.E is not None:
            fields[Field.E] = True
            names.append(f"{name}:E")
            total += 1
        if det.J is not None:
            fields[Field.J] = True
            names.append(f"{name}:J")
            total += 1
        if det.H is not None:
            fields[Field.H] = True
            names.append(f"{name}:H")
            total += 1
    # region get layout
    xn: int = 1
    yn: int = 1
    while xn * yn < total:
        if np.abs(np.log((xn + 1) / yn / preferredRatio)) < np.abs(np.log(xn / (yn + 1) / preferredRatio)):
            xn += 1
        else:
            yn += 1
    fig, ax = plt.subplots(xn, yn, squeeze=False)
    ax1d: list[Axes] = list(ax.reshape((xn * yn)))
    # endregion
    # region set titles and axes
    for i in range(len(names)):
        ax1d[i].set_title(names[i])
    # endregion
    while True:
        frameArtists = []
        counter: int = 0
        for name, myTuple in detectors.items():
            pos = myTuple[0]
            fields = myTuple[1]
            shapeList = myTuple[2]
            det = grid.detectors[name]
            if len(shapeList) == 1:
                indexes = np.arange(shapeList[0])
                if fields.get(Field.rho):
                    frameArtists.extend(ax1d[counter].plot(indexes, det.rho.reshape(shapeList)))
                    counter += 1
                if fields.get(Field.E):
                    E = np.sqrt(det.E[0] ** 2 + det.E[1] ** 2 + det.E[2] ** 2)
                    frameArtists.extend(ax1d[counter].plot(indexes, E.reshape(shapeList)))
                    counter += 1
            elif len(shapeList) == 2:
                if fields.get(Field.rho):
                    frameArtists.append(ax1d[counter].imshow(det.rho.reshape(shapeList),
                                                             cmap="plasma", interpolation="nearest"))
                    counter += 1
                if fields.get(Field.E):
                    field = det.E
                    fieldImage = np.sqrt(field[0] ** 2 + field[1] ** 2 + field[2] ** 2).reshape(shapeList)
                    axesImage = ax1d[counter].imshow(fieldImage, cmap="plasma", interpolation="nearest")
                    frameArtists.append(axesImage)
                    #frameArtists.append(fig.colorbar(axesImage, cax=ax1d[counter]))
                    #TODO: figure out why colorbar is so slow/switch to FuncAnimation
                    counter += 1
                if fields.get(Field.J):
                    field = det.J
                    fieldImage = np.sqrt(field[0] ** 2 + field[1] ** 2 + field[2] ** 2).reshape(shapeList)
                    axesImage = ax1d[counter].imshow(fieldImage, cmap="plasma", interpolation="nearest")
                    frameArtists.append(axesImage)
                    #frameArtists.append(fig.colorbar(axesImage, cax=ax1d[counter]))
                    #TODO: figure out why colorbar is so slow/switch to FuncAnimation
                    counter += 1
                if fields.get(Field.H):
                    field = det.H
                    fieldImage = np.sqrt(field[0] ** 2 + field[1] ** 2 + field[2] ** 2).reshape(shapeList)
                    axesImage = ax1d[counter].imshow(fieldImage, cmap="plasma", interpolation="nearest")
                    frameArtists.append(axesImage)
                    #frameArtists.append(fig.colorbar(axesImage, cax=ax1d[counter]))
                    #TODO: figure out why colorbar is so slow/switch to FuncAnimation
                    counter += 1
        ims.append(frameArtists)
        if grid.load_next_frame():
            break

    ani = animation.ArtistAnimation(fig, ims, interval=20, blit=True, repeat_delay=1000)
    fig.tight_layout()
    plt.show()
