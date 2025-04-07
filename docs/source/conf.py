# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Archimedes"
copyright = "2025, Jared Callaham"
author = "Jared Callaham"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
]

intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

myst_enable_extensions = [
    "amsmath",
    "attrs_inline",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

autosummary_generate = True
autosummary_imported_members = True

templates_path = ["_templates"]
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = "alabaster"
html_theme = "furo"
html_theme_options = {
    # Light mode variables
    "light_css_variables": {
        "color-brand-primary": "#D35400",  # copper orange
        "color-brand-content": "#C0392B",  # ember red
        "color-admonition-background": "rgba(211, 84, 0, 0.1)",  # copper orange (transparent)
        "color-background-primary": "#F5F5F5",  # light gray
        "color-background-secondary": "#EEEEEE",  # slightly darker gray
        "color-foreground-primary": "#2A2A2A",  # dark charcoal
        "color-foreground-secondary": "#5D4037",  # rich brown
        "color-link": "#C0392B",  # ember red
        "color-link-hover": "#D35400",  # copper orange
    },
    # Dark mode variables
    "dark_css_variables": {
        "color-brand-primary": "#F1C40F",  # pale gold
        "color-brand-content": "#D35400",  # copper orange
        "color-admonition-background": "rgba(241, 196, 15, 0.1)",  # pale gold (transparent)
        "color-background-primary": "#2A2A2A",  # dark charcoal
        "color-background-secondary": "#1A1A1A",  # darker charcoal
        "color-foreground-primary": "#F5F5F5",  # light gray
        "color-foreground-secondary": "#DDDDDD",  # slightly darker light gray
        "color-link": "#D35400",  # copper orange
        "color-link-hover": "#F1C40F",  # pale gold
    },
}

html_static_path = ["_static"]

# These paths are either relative to html_static_path
# or fully qualified paths (eg. https://...)
html_css_files = [
    'custom.css',
]

html_js_files = []

# Add favicon configuration
html_favicon = '_static/favicon.ico'