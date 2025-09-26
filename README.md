# stable-cart


An experimental decision tree regressor that focuses on stability while
maintaining the familiar scikit-learn API.

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/stable-cart.git
cd stable-cart
pip install -e .
```

Or install directly with pip (once published to PyPI):

```bash
pip install stable-cart
```

## Usage

### Training a stable CART

```python
from stable_cart.less_greedy_tree import LessGreedyTreeClassifier

# Example dataset
X = [[0, 0], [1, 1], [1, 0], [0, 1]]
y = [0, 1, 1, 0]

# Train a more stable decision tree
tree = LessGreedyTreeClassifier(max_depth=3)
tree.fit(X, y)

# Predict
print(tree.predict([[1, 0], [0, 0]]))
```

### Evaluation

The package includes evaluation utilities for comparing model stability across different runs.

```python
from stable_cart import evaluation
from sklearn.tree import DecisionTreeClassifier

# Example: evaluate stability of sklearn CART vs. LessGreedyTree
evaluation.compare_models(
    models=[DecisionTreeClassifier(), LessGreedyTreeClassifier()],
    X=X,
    y=y,
    n_trials=10
)
```

This produces accuracy, stability, and agreement metrics across multiple random seeds.

## Modules

- **`less_greedy_tree.py`**
  Implements the `LessGreedyTreeClassifier`, a variant of CART with modified split selection to reduce variance and improve stability.

- **`evaluation.py`**
  Functions to test and compare models across seeds, reporting accuracy, stability, and cross-run consistency.

- **`__init__.py`**
  Exposes package-level imports for easy access.

## Motivation

Standard decision trees (CART) are **unstable**: small changes in the training data can result in very different tree structures. While ensembles like random forests address this with bagging, **stable CART** focuses on modifying the tree induction process itself to produce **more consistent models** without sacrificing interpretability.

## Roadmap

- [ ] Add regression tree support
- [ ] Publish package to PyPI
- [ ] Add visualization utilities for comparing stable vs. standard trees


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
