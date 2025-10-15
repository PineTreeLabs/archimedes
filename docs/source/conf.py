# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Archimedes"
copyright = "2025, Pine Tree Labs, LLC"
author = "Jared Callaham"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # "myst_parser",
    "myst_nb",
    "sphinx_design",
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinxcontrib.googleanalytics",
]

intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

myst_enable_extensions = [
    "amsmath",
    "attrs_inline",
    "attrs_block",
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

nb_kernel_rgx_aliases = {
    ".*": "archimedes"  # Use the registered Jupyter kernel name
}
nb_execution_mode = "cache"
nb_execution_timeout = 60
nb_execution_cache_path = ".jupyter_cache"
nb_execution_allow_errors = False  # Fail build on exceptions
nb_execution_raise_on_error = True  # Raise immediately on error
nb_execution_excludepatterns = [
    "experimental/*",  # Skip WIP content
    "benchmarks/*",  # Skip long-running benchmarks
]

autosummary_generate = True
autosummary_imported_members = True

# Maximum signature line length before breaking into multiple lines
maximum_signature_line_length = 88
autodoc_preserve_defaults = True
# autodoc_typehints = "description"  # or "signature", "both", "none"

templates_path = ["_templates"]
exclude_patterns = []

googleanalytics_id = "G-DMLVH3TEDW"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = "alabaster"
html_theme = "furo"
html_theme_options = {
    # Light mode variables
    "light_css_variables": {
        "color-brand-primary": "#D35400",  # copper orange
        "color-brand-content": "#C0392B",  # ember red
        "color-admonition-background": "rgba(211, 84, 0, 0.1)",  # transparent orange
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
        "color-admonition-background": "rgba(241, 196, 15, 0.1)",  # transparent gold
        "color-background-primary": "#2A2A2A",  # dark charcoal
        "color-background-secondary": "#1A1A1A",  # darker charcoal
        "color-foreground-primary": "#F5F5F5",  # light gray
        "color-foreground-secondary": "#DDDDDD",  # slightly darker light gray
        "color-link": "#D35400",  # copper orange
        "color-link-hover": "#F1C40F",  # pale gold
    },
    "top_of_page_buttons": [],
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/pinetreelabs/archimedes",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
            "class": "",
        },
    ],
}

html_static_path = ["_static"]

# These paths are either relative to html_static_path
# or fully qualified paths (eg. https://...)
html_css_files = [
    "custom.css",
]

html_js_files = []

# Add favicon configuration
html_favicon = "_static/favicon.ico"
