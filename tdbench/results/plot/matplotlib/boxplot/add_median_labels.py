import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

def add_median_labels(ax: plt.Axes, fmt: str = ".4f") -> None:
    """Add text labels to the median lines of a seaborn boxplot.

    Args:
        ax: plt.Axes, e.g. the return value of sns.boxplot()
        fmt: format string for the median value
    """
    lines = ax.get_lines()
    boxes = [c for c in ax.get_children() if "Patch" in str(c)]
    start = 4
    if not boxes:  # seaborn v0.13 => fill=False => no patches => +1 line
        boxes = [c for c in ax.get_lines() if len(c.get_xdata()) == 5]
        start += 1
    lines_per_box = len(lines) // len(boxes)
    for mean in lines[start::lines_per_box]:
        x, y = (data.mean() for data in mean.get_data())
        # choose value depending on horizontal or vertical plot orientation
        value = x if len(set(mean.get_xdata())) == 1 else y
        text = ax.text(x, y, f'hihi{value:{fmt}}',
                        ha='center', va='center',
                        color='white')
        # create median-colored border around white text for contrast
        text.set_path_effects([
            path_effects.Stroke(linewidth=3, foreground=mean.get_color()),
            path_effects.Normal(),
        ])