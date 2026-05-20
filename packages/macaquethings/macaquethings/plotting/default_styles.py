import matplotlib as mpl
import seaborn as sns

# global constants
# --- png save options
DPI = 500

# --- plot size utilities
FULL_WIDTH = 7.0
HALF_WIDTH = FULL_WIDTH / 2
THIRD_WIDTH = FULL_WIDTH / 3
QUARTER_WIDTH = FULL_WIDTH / 4

HALF_PANEL_SIZE = (HALF_WIDTH, QUARTER_WIDTH)  # half width
QUARTER_PANEL_SIZE = (QUARTER_WIDTH, QUARTER_WIDTH)  # quarter width
FULL_PANEL_SIZE = (FULL_WIDTH, HALF_WIDTH)  # full width
THIRD_PANEL_SIZE = (THIRD_WIDTH, QUARTER_WIDTH)  # third width


# set matplotlib defaults
def figure_style(font_size=5):
    """
    Set style for plotting figures
    """
    sns.set(
        style="ticks",
        context="paper",
        font="sans-serif",
        rc={
            "font.size": font_size,
            "figure.titlesize": font_size,
            "figure.labelweight": font_size,
            "axes.titlesize": font_size,
            "axes.labelsize": font_size,
            "axes.linewidth": 0.5,
            "lines.linewidth": 1,
            "lines.markersize": 3,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "savefig.transparent": False,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.minor.size": 2,
            "ytick.minor.size": 2,
            "xtick.minor.width": 0.5,
            "ytick.minor.width": 0.5,
            "legend.fontsize": font_size,
            "legend.title_fontsize": font_size,
            "legend.frameon": False,
        },
    )
    mpl.rcParams["svg.fonttype"] = "none"  # save svg with editable text
    mpl.rcParams["figure.dpi"] = DPI
