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
    'light': {
        'background': light_gray,
        'text': dark_charcoal,
        'grid': '#DDDDDD',
        'primary': copper_orange,
        'secondary': slate_blue,
        'tertiary': pale_gold,
        'highlight': ember_red,
    },
    'dark': {
        'background': dark_charcoal,
        'text': light_gray,
        'grid': '#444444',
        'primary': copper_orange,
        'secondary': steel_blue, 
        'tertiary': pale_gold,
        'highlight': ember_red,
    }
}

def set_theme(theme):
    colors = color_schemes[theme]
    
    # Set up the figure with theme colors
    mpl.rcParams['figure.figsize'] = (7, 3)
    mpl.rcParams['axes.facecolor'] = colors['background']
    mpl.rcParams['figure.facecolor'] = colors['background']
    mpl.rcParams['text.color'] = colors['text']
    mpl.rcParams['axes.labelcolor'] = colors['text']
    mpl.rcParams['xtick.color'] = colors['text']
    mpl.rcParams['ytick.color'] = colors['text']
    mpl.rcParams['grid.color'] = colors['grid']

    mpl.rcParams['axes.prop_cycle'] = cycler(color=[
        colors['primary'],
        colors['secondary'],
        colors['tertiary'],
    ])
