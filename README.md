# Prediction instability audits for fitted models

This package measures how much a model's predictions change when its training
data change. It refits the complete model-building procedure on bootstrap
samples, predicts the same evaluation cases, and returns the individual
prediction distributions and their aggregate uncertainty.

The package does not claim to stabilize a model. It tells you whether a fitted
procedure is stable enough for its intended use, where it is unstable, and how
candidate procedures trade validation performance for stability.

The supported API works with arbitrary estimators and pipelines and includes a
representative-model selector plus fixed-design linear calibration. Their names
describe their mechanisms rather than claiming that they stabilize a model.

## What the package measures

`bootstrap_predictions` repeats the full call to `fit` on bootstrap samples of
the training data. This matters because preprocessing, feature selection,
hyperparameter tuning, and estimation can all contribute to instability. Put
those steps in the supplied estimator or pipeline and the audit repeats them.
The audit seed controls resampling; estimator seeds remain the responsibility
of the supplied factory, so algorithmic randomness can be included or held
fixed deliberately.

The function reports two comparisons:

* MAPE compares each bootstrap refit with the model fitted on the full training
  data. For class labels, this is disagreement with the original classifier.
* Pairwise instability compares two independently refitted models. For numeric
  predictions it is mean squared difference. For class labels it is the
  probability of disagreement. For class probabilities it is squared Euclidean
  distance between the full probability vectors.

Classification uses a pairs bootstrap conditional on a draw containing at
least two observed classes, so class prevalence can otherwise vary across
refits and standard classifiers remain usable. Rejected draws are reported.
Probability columns are aligned by class, and
the result is invariant to renaming or renumbering classes.

## Installation

```bash
pip install stable-cart
```

The plotting functions need matplotlib:

```bash
pip install "stable-cart[plots]"
```

## Audit one model-building procedure

```python
from sklearn.datasets import make_regression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV

from stable_cart import bootstrap_instability

X, y = make_regression(
    n_samples=500,
    n_features=12,
    noise=10,
    random_state=0,
)
X_train, X_eval = X[:350], X[350:]
y_train = y[:350]

result = bootstrap_instability(
    lambda: make_pipeline(StandardScaler(), RidgeCV()),
    X_train,
    y_train,
    X_eval,
    task="continuous",
    n_bootstrap=500,
    random_state=0,
)

print(result)
```

The result includes the mean, 90th percentile, and maximum per-case prediction
variance; MAPE against the original fit; pairwise instability; and Monte Carlo
standard errors for both aggregate comparisons. Use the standard errors to
decide whether more resamples are needed. Twenty resamples are useful for a
quick diagnostic, not a final estimate.

To inspect the raw prediction distribution:

```python
from stable_cart import bootstrap_predictions, plot_prediction_instability

raw = bootstrap_predictions(
    lambda: make_pipeline(StandardScaler(), RidgeCV()),
    X_train,
    y_train,
    X_eval,
    n_bootstrap=500,
    random_state=0,
)

plot_prediction_instability(raw)
```

## Audit a multiclass probability model

Class labels have no numerical spacing. The package therefore measures either
label disagreement or movement in aligned probability vectors. It never takes
the variance of integer class codes or the variance of maximum confidence.

```python
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from stable_cart import bootstrap_instability

X, y = load_wine(return_X_y=True)

result = bootstrap_instability(
    lambda: make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000),
    ),
    X[:130],
    y[:130],
    X[130:],
    task="categorical",
    prediction_method="predict_proba",
    n_bootstrap=500,
    random_state=0,
)
```

Probability columns are aligned through each estimator's `classes_` attribute.
The raw result keeps the complete `(resample, case, class)` array.

## Compare procedures on a validation frontier

`stability_frontier` sweeps a parameter grid and returns the configurations no
other configuration beats on both validation score and instability.

```python
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from stable_cart import stability_frontier

X_train, X_validation, y_train, y_validation = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=0,
)

frontier = stability_frontier(
    lambda **params: DecisionTreeRegressor(random_state=0, **params),
    {
        "max_depth": [2, 4, 8],
        "ccp_alpha": [0.0, 0.001, 0.01],
    },
    X_train,
    y_train,
    X_eval=X_validation,
    y_eval=y_validation,
    task="continuous",
    n_bootstrap=200,
    random_state=0,
)

for point in frontier["frontier"]:
    print(point["score"], point["instability"], point["params"])
```

The reported score is a validation score used to construct the frontier. It is
not an unbiased final performance estimate. Evaluate the selected procedure on
an untouched test set or use an outer resampling loop.

The frontier does not choose a model for you. A procedure that ignores its data
has zero instability and poor predictive performance. Reading stability beside
validation performance makes that failure visible.

## Select a representative fitted model

The top-level API contains a generic representative-model selector and
fixed-design linear calibration:

```python
from sklearn.linear_model import Ridge

from stable_cart import RepresentativeEstimator

representative = RepresentativeEstimator(
    estimator=Ridge(),
    task="regression",
    n_candidates=20,
    random_state=0,
)
representative.fit(X_train, y_train)
predictions = representative.predict(X_eval)
```

`RepresentativeEstimator` passes scikit-learn's maintained estimator checks in
regression and classification modes, supports multiclass classification, and is
not tree-specific. In a frozen 288-dataset evaluation, selecting the prediction
medoid reduced median prediction instability relative to selecting the
best-validation candidate from the same pool in every design cell. The score
tradeoff was acceptable for linear classification and tree regression, not for
tree classification; linear regression showed negligible benefit. This is
task-specific evidence, not a universal guarantee.

The supported surface and its limits are listed in
[`PACKAGE_SCOPE.md`](PACKAGE_SCOPE.md).

## Complete user workflow

[`examples/user_workflow.py`](examples/user_workflow.py) runs the public API as
an installed user would: regression and multiclass audits, a validation
score-instability frontier followed by untouched test evaluation,
representative-model selection, tree-structure diagnostics, fixed-design linear
calibration, plots, and a JSON summary.

```bash
python examples/user_workflow.py --output workflow-output
```

## Development

```bash
uv sync --all-groups
make lint
make test
uv run pyright
make docs
```

`make ci-docker` runs the same lint and test checks in a standard Python 3.11
container.

## Methodological basis

The bootstrap protocol follows Riley and Collins, "Stability of clinical
prediction models developed using statistical or machine learning methods,"
*Biometrical Journal* 65(8), 2023. Their `pminternal` R package implements the
clinical prediction workflow that motivated this package.

The identity behind squared pairwise instability is elementary:

```text
E[(f_D(x) - f_D'(x))^2] = 2 Var(f_D(x))
```

It requires independent, identically distributed refits. The package's
contribution is operational: repeat the actual model-building procedure,
preserve the individual prediction distributions, apply valid classification
metrics, quantify Monte Carlo error, and connect the result to model selection
without reporting the validation frontier as final test performance.
