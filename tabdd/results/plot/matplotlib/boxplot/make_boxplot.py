from typing import Literal
import numpy as np
import matplotlib.pyplot as plt

from ..colormaps import get_cmap

DEFAULT_COLORMAP = [get_cmap("tab10")(i) for i in range(10)]


def make_boxplot(
    x: np.ndarray,
    y: np.ndarray,
    ax: plt.Axes,
    colormap: list[tuple[float]] = DEFAULT_COLORMAP,
    rotate_xlabels: int = None,
    vert: bool = True,
    show_text: bool = True,
    # figsize: tuple = (5,5),
    style: Literal["color", "hatch", "color-hatch"] = "hatch",
    text_offsets: list[float] = None,
):
    bp = ax.boxplot(
        y,
        notch=True,
        labels=x,
        patch_artist=True,
        medianprops=dict(color="black"),
        showmeans=True,
        meanline=True,
        meanprops=dict(color="black"),
        vert=vert,
    )

    if rotate_xlabels is not None:
        ax.set_xticklabels(x, rotation=rotate_xlabels, ha="right")

    # Define a list of hatch patterns
    hatch_patterns = ["//", "\\\\", "xx", "-", "++", "..", "-\\", "\\|"]

    # Define a list of colors for the hatches
    # hatch_colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'black', 'purple']

    if style == "hatch":
        # Apply hatch patterns and colors to each box
        for box_patch, hatch, color in zip(bp["boxes"], hatch_patterns, colormap):
            box_patch.set_hatch(hatch)  # Set the hatch pattern
            box_patch.set_edgecolor(color)  # Set the color of the hatch
            box_patch.set_facecolor("none")  # Remove the style
    elif style == "color":
        for box_patch, hatch, color in zip(bp["boxes"], hatch_patterns, colormap):
            box_patch.set_facecolor(color)
    elif style == "color-hatch":
        for box_patch, hatch, color in zip(bp["boxes"], hatch_patterns, colormap):
            box_patch.set_hatch(hatch)  # Set the hatch pattern
            box_patch.set_facecolor(color)  # Remove the style

    # Calculate the mean for each dataset
    means = [np.mean(yy) for yy in y]

    # max_y = max([max(data) for data in y])  # Maximum of all data
    # pad = (max_y - min([min(data) for data in y])) * 0.05  # A small padding above the highest box

    if show_text:
        # Add mean value to each box
        for i, mean in enumerate(means):
            offset = 0.01
            if text_offsets is not None:
                offset = text_offsets[i]
                if offset is None:
                    offset = 0.01

            text = f"{mean:.2f}"
            if plt.rcParams["text.usetex"]:
                text = f"\\textbf{{{mean:.2f}}}"

            if vert:
                ax.text(
                    i + 1,
                    mean + offset,
                    text,  # Add a small padding above the mean
                    horizontalalignment="center",
                    color="black",  # Change text color for visibility
                    weight="semibold",
                    backgroundcolor="none",
                )  # Add a background for visibility
            else:
                ax.text(
                    mean + offset,
                    i + 1,
                    text,  # Add a small padding above the mean
                    horizontalalignment="center",
                    verticalalignment="center",
                    color="black",  # Change text color for visibility
                    weight="semibold",
                    backgroundcolor="none",
                )  # Add a background for visibility
