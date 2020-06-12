# Sphinx guidelines for IMRSV production packages

## Initial setup

Example documentation files (eg: `docs/source/conf.py`) have been added to this repo for your convenience. Please **delete everything** inside the docs folder before running the next commands.

Run `sphinx-quickstart`, please do so inside the [docs](../docs) folder.

Following settings are the recommended options for the `sphinx-quickstart` command. For all other options, please use the default values.
- Separate source and build directories? [y]
- autodoc: automatically insert docstrings from modules? [y]
- doctest: automatically test code snippets in doctest blocks [y]
- todo: write "todo" entries that can be shown or hidden on build [y]
- coverage: checks for documentation coverage [y]
- mathjax: include math, rendered in the browser by MathJax [y]
- viewcode: include links to the source code of documented Python objects [y]
- githubpages: create .nojekyll file to publish the document on GitHub pages [y]

[ReadtheDocs](https://sphinx-rtd-theme.readthedocs.io/en/latest/) theme is recommended to be used within IMRSV. Please install this theme by `$ pip install sphinx_rtd_theme`.

## Edits to [docs/source/conf.py](../docs/source/conf.py)

Please do the following edits inside `conf.py`. (`conf.py` is a `Python` file, edit this file as you would do with any other `Python` file):

- [ ] Add `autoclass_content = 'both'` to generate docs from `__init__` functions within Classes.
- [ ] If `Markdown` support is required for documentation, please install `recommonmark` as shown [here](https://www.sphinx-doc.org/en/master/usage/markdown.html) and add `recommonmark` into the extensions section in [docs/source/conf.py](../docs/source/conf.py). Also change the `source_suffix` variable to , `source_suffix = ['.rst', '.md']`
- [ ] Change `html_theme` variable to `html_theme = "sphinx_rtd_theme"`
- [ ] Add
```
import os
import sys
sys.path.insert(0, os.path.abspath('../../src/imrsv/'))
```

## [Sphinx-apidoc](https://www.sphinx-doc.org/en/master/man/sphinx-apidoc.html) for automatic documentation

- [Sphinx-apidoc](https://www.sphinx-doc.org/en/master/man/sphinx-apidoc.html) can be used to automatically generate documentation from comments made within your `Python` files.
- Run `$ sphinx-apidoc -o <OUTPUT_PATH> <MODULE_PATH> [EXCLUDE_PATTERN, ..]` to create documentation from your `Python` files. eg: `sphinx-apidoc -o docs/source src/imrsv/production/`

## Building the documentation

- Run `$ make html` from within the `docs/` folder to make the html documentation. Now your documentation will be in, `docs/build/html/`. You can view the documentation by opening `docs/build/html/index.html` file in your favourite web browser. (i.e. chrome or firefox)
