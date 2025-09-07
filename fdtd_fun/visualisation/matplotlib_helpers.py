from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Iterable, Callable, Sequence

import numpy as np
from matplotlib import pyplot as plt, animation

from fdtd_fun.detector import Detectable
from fdtd_fun.typing_ import Field, Comp, Key

if TYPE_CHECKING:
    from fdtd_fun import Grid
logger = logging.getLogger(__name__)
fieldsToCheck = [Field.E, Field.B, Field.J, Field.V]
obsNames = {
    Detectable.E: "E",
    Detectable.Ex: "Ex",
    Detectable.Ey: "Ey",
    Detectable.Ez: "Ez",
    Detectable.B: "B",
    Detectable.Bx: "Bx",
    Detectable.By: "By",
    Detectable.Bz: "Bz",
    Detectable.J: "J",
    Detectable.Jx: "Jx",
    Detectable.Jy: "Jy",
    Detectable.Jz: "Jz",
    Detectable.V: "V"
}
obsColors = {
    Detectable.E: "b",
    Detectable.Ex: "b",
    Detectable.Ey: "b",
    Detectable.Ez: "b",
    Detectable.B: "r",
    Detectable.Bx: "r",
    Detectable.By: "r",
    Detectable.Bz: "r",
    Detectable.J: "m",
    Detectable.Jx: "m",
    Detectable.Jy: "m",
    Detectable.Jz: "m",
    Detectable.V: "y"
}


def animate(grid: Grid, time: float = 4.0, fps: int = 30, preferredRatio: float = 0.7):
    if grid.file is None or grid.tot_frames is None:
        raise ValueError("This Grid doesn't seem to have been loaded from a file - please use Grid.load_from_file()")
    fig = plt.figure()
    ims = []
    detectors = {}
    totSubplots: int = 0
    tot_frames = int(time * fps)
    frame_step = float(grid.tot_frames) / tot_frames
    frame = 0
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
        detectors[det.name] = (pos, det, shapeList)
        for obs in det._toRead:
            totSubplots += 1

    # endregion
    # region get layout
    xn: int = 1
    yn: int = 1
    while xn * yn < totSubplots:
        if np.abs(np.log((xn + 1) / yn / preferredRatio)) < np.abs(np.log(xn / (yn + 1) / preferredRatio)):
            xn += 1
        else:
            yn += 1
    for j in range(totSubplots):
        fig.add_subplot(xn, yn, j + 1)
    ax = fig.get_axes()
    # endregion
    # region set titles and axes
    i = 0
    for _, det in grid.detectors.items():
        for obs in det._toRead:
            ax[i].set_title(f"{det.name}:{obs.name}")
            i += 1
    # endregion
    while True:
        logger.debug(f"Frame {frame}")
        frameArtists = []
        # region draw
        subplotCounter = 0
        for _, myTuple in detectors.items():
            det = myTuple[1]
            shapeList = myTuple[2]
            indexes = np.arange(shapeList[0])
            for i in range(len(det._toRead)):
                obs = det._toRead[i]
                value = det.values[i]
                if len(shapeList) == 1:
                    if obs == Detectable.V:
                        frameArtists += ax[subplotCounter].plot(indexes[:-1],
                                                                value.reshape(shapeList[0] - 1), obsColors[obs])
                        subplotCounter += 1
                    else:
                        frameArtists += ax[subplotCounter].plot(indexes,
                                                                value.reshape(shapeList), obsColors[obs])
                        subplotCounter += 1
                elif len(shapeList) == 2:
                    frameArtists.append(
                        ax[subplotCounter].imshow(value.reshape(shapeList),
                                                  cmap="plasma", interpolation="none", origin='lower'))
                    subplotCounter += 1
        # endregion
        titletext = f"Time: {grid.time() * 1e9:.1F} ns"
        ttl = plt.text(0.5, 1, titletext, horizontalalignment='center', verticalalignment='top',
                       transform=plt.gcf().transFigure)
        frameArtists.append(ttl)
        ims.append(frameArtists)
        frame += 1
        while frame * frame_step > grid.t:
            if not grid.load_next_frame():
                break
        else:
            continue
        break  # only reachable if the inner loop is broken out of

    logger.info("Creating animation")
    ani = animation.ArtistAnimation(fig, ims, blit=True, repeat_delay=1000)
    fig.tight_layout()
    logger.info("Saving animation to a file")
    ani.save(f"ani{grid.name}.mp4", dpi=300, fps=fps)
    grid._reload()
    logger.info("Reloaded grid to initial state")


class ReadingOverTime:
    def __init__(self, ylabel: str, x: Key, y: Key, z: Key, toRead: list[Detectable],
                 func: Callable[[np.ndarray, ...], float]):
        # consider not using Detectable here as that is for detectors only and readings over time do not use detectors
        """
        :param x:
        :param y:
        :param z:
        :param toRead:
        :param func:
        """
        self.x = x
        self.y = y
        self.z = z
        self.toRead = toRead
        self.func = func
        self.ylabel = ylabel


def plotOverTime(grid: Grid, nframes: int, ncolumns: int = 4, *args: ReadingOverTime):
    if grid.file is None or grid.tot_frames is None:
        raise ValueError("This Grid doesn't seem to have been loaded from a file - please use Grid.load_from_file()")
    if len(args) == 0:
        raise ValueError("No ReadingOverTime objects supplied")
    # region prep
    fig = plt.figure()
    nrows=1
    while ncolumns*nrows<len(args):
        nrows+=1
    readings = [[]]*len(args)
    for i in range(len(args)):
        fig.add_subplot(nrows,ncolumns,i+1)
    ax = fig.get_axes()
    for i in range(len(args)):
        reading:ReadingOverTime = args[i]
        ax[i].set_ylabel(reading.ylabel)
        ax[i].set_xlabel("Time, s")
    timestep = float(grid.tot_frames) / nframes

    # endregion
    while True:
        break # actually maybe I won't use this

