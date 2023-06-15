# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Infinity Core API"
copyright = "2022, Infinity AI, Inc."
release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.napoleon", "myst_parser"]
autodoc_typehints = "description"
# autoclass_content = 'both'

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]

html_title = ""
html_theme_options = {
    "home_page_in_toc": True,
    "github_url": "https://github.com/toinfinityai/infinity-api",
    "repository_url": "https://github.com/toinfinityai/infinity-api",
    "repository_branch": "master",
    "path_to_docs": "docs",
    "use_repository_button": True,
}

html_logo = "_static/infinity_ai_logo.png"
