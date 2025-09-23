# stable-cart

An experimental decision tree regressor that focuses on stability while
maintaining the familiar scikit-learn API.

## Documentation

The project documentation is authored with [Sphinx](https://www.sphinx-doc.org)
and is automatically published to GitHub Pages whenever changes are pushed
to the `main` branch. To build the docs locally run:

```bash
python -m pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
```

The rendered site will be available from the generated
`docs/_build/html/index.html` file.
