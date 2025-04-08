import matplotlib as mpl
from cycler import cycler

__all__ = ["color_schemes", "set_theme"]

dark_charcoal = "#2A2A2A"
copper_orange = "#D35400"
rich_brown = "#5D4037"
ember_red = "#C0392B"
pale_gold = "#F1C40F"
light_gray = "#F5F5F5"
slate_blue = "#34495E"
steel_blue = "#5C9DC0"

color_schemes = {
    "light": {
        "background": light_gray,
        "text": dark_charcoal,
        "grid": "#DDDDDD",
        "primary": copper_orange,
        "secondary": slate_blue,
        "tertiary": pale_gold,
        "highlight": ember_red,
    },
    "dark": {
        "background": dark_charcoal,
        "text": light_gray,
        "grid": "#444444",
        "primary": copper_orange,
        "secondary": steel_blue,
        "tertiary": pale_gold,
        "highlight": ember_red,
    },
}


def set_theme(theme: str):
    """
    Configure matplotlib plotting parameters with a predefined color theme.

    This function sets up matplotlib's global parameters (rcParams) to apply
    a consistent color scheme across all subsequently created plots. It configures
    colors for plot elements including backgrounds, text, grid lines, and the color
    cycle used for plotting multiple data series.

    All documentation and example notebooks should check for an environment variable
    `ARCHIMEDES_THEME` to set the theme; this is automatically set to build the
    light/dark themed documentation.

    Parameters
    ----------
    theme : str
        The name of the color theme to apply. Must be one of ('light', 'dark').

    Returns
    -------
    None
        This function modifies matplotlib's global state but does not return a value.

    Notes
    -----
    This function modifies matplotlib's global rcParams, which affects all subsequent
    plots created in the same session. Call this function before creating any plots
    that should use the specified theme.

    The function sets the following matplotlib parameters:
    - figure.figsize: Sets a default figure size of (7, 3)
    - Background colors (axes.facecolor, figure.facecolor)
    - Text colors (text.color, axes.labelcolor, xtick.color, ytick.color)
    - Grid colors (grid.color)
    - Color cycle for multiple series (axes.prop_cycle)

    Examples
    --------
    >>> import archimedes as arc
    >>> import matplotlib.pyplot as plt
    >>> arc.set_theme('dark')
    >>> plt.plot([1, 2, 3, 4])
    >>> plt.title('A plot with dark theme')
    >>> plt.show()

    >>> # Change to another theme
    >>> arc.set_theme('light')
    >>> plt.figure()
    >>> plt.plot([4, 3, 2, 1])
    >>> plt.title('A plot with light theme')
    >>> plt.show()

    See Also
    --------
    matplotlib.pyplot.rcParams : The parameter dictionary that this function modifies
    """

    colors = color_schemes[theme]

    # Set up the figure with theme colors
    mpl.rcParams["figure.figsize"] = (7, 3)
    mpl.rcParams["axes.facecolor"] = colors["background"]
    mpl.rcParams["figure.facecolor"] = colors["background"]
    mpl.rcParams["text.color"] = colors["text"]
    mpl.rcParams["axes.labelcolor"] = colors["text"]
    mpl.rcParams["xtick.color"] = colors["text"]
    mpl.rcParams["ytick.color"] = colors["text"]
    mpl.rcParams["grid.color"] = colors["grid"]

    mpl.rcParams["axes.prop_cycle"] = cycler(
        color=[
            colors["primary"],
            colors["secondary"],
            colors["tertiary"],
        ]
    )
