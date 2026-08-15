"""Tests for the prediction-stability plots.

A plot is hard to assert about, so these check the two things that can silently
go wrong: that the data reaching the axes is the data that was passed in, and
that a more stable model really does draw a tighter cloud than a less stable
one. Anything beyond that is a matter for the eye.
"""

import numpy as np
import pytest

# matplotlib ships in the optional "plots" extra, so an environment with only
# the runtime dependencies must skip these rather than fail to collect them.
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from sklearn.datasets import make_classification, make_regression  # noqa: E402
from sklearn.ensemble import BaggingRegressor  # noqa: E402
from sklearn.tree import (  # noqa: E402
    DecisionTreeClassifier,
    DecisionTreeRegressor,
)

from stable_cart import (  # noqa: E402
    bootstrap_predictions,
    plot_mape_by_prediction,
    plot_prediction_instability,
    plot_stability_frontier,
    stability_frontier,
)


@pytest.fixture
def regression_data():
    """A regression problem with signal and noise."""
    return make_regression(
        n_samples=300, n_features=5, n_informative=3, noise=8.0, random_state=0
    )


@pytest.fixture
def raw(regression_data):
    """Bootstrap predictions from an unpruned tree — deliberately unstable."""
    X, y = regression_data
    return bootstrap_predictions(
        lambda: DecisionTreeRegressor(max_depth=8, random_state=0),
        X[:200],
        y[:200],
        X[200:],
        n_bootstrap=10,
        random_state=0,
    )


class TestInstabilityPlot:
    """The canonical Riley and Collins picture."""

    def test_plots_one_point_per_resample_per_individual(self, raw):
        _fig, ax = plt.subplots()

        plot_prediction_instability(raw, ax=ax)

        n_boot, n_eval = raw["bootstrap"].shape
        assert ax.collections[0].get_offsets().shape[0] == n_boot * n_eval
        plt.close("all")

    def test_x_axis_is_the_original_model(self, raw):
        """Plotting resample-against-resample would look similar and mean nothing."""
        _fig, ax = plt.subplots()

        plot_prediction_instability(raw, ax=ax)

        drawn = np.unique(ax.collections[0].get_offsets()[:, 0])
        assert np.allclose(drawn, np.unique(raw["original"]))
        plt.close("all")

    def test_large_evaluation_sets_are_subsampled(self, regression_data):
        """Above a few thousand points the cloud becomes a rectangle."""
        X, y = regression_data
        big = bootstrap_predictions(
            lambda: DecisionTreeRegressor(max_depth=4, random_state=0),
            X,
            y,
            X,
            n_bootstrap=20,
            random_state=0,
        )
        _fig, ax = plt.subplots()

        plot_prediction_instability(big, ax=ax, max_points=50)

        assert ax.collections[0].get_offsets().shape[0] == 50 * 20
        plt.close("all")

    def test_a_stabler_model_draws_a_tighter_cloud(self, regression_data):
        """The plot must rank models the way the metric does, or it misleads."""
        X, y = regression_data
        common = {"n_bootstrap": 10, "random_state": 0}
        loose = bootstrap_predictions(
            lambda: DecisionTreeRegressor(max_depth=10, random_state=0),
            X[:200],
            y[:200],
            X[200:],
            **common,
        )
        tight = bootstrap_predictions(
            lambda: BaggingRegressor(
                DecisionTreeRegressor(max_depth=10), n_estimators=15, random_state=0
            ),
            X[:200],
            y[:200],
            X[200:],
            **common,
        )

        assert np.mean(tight["mape_per_point"]) < np.mean(loose["mape_per_point"])


class TestMapePlot:
    """Instability as a function of predicted value."""

    def test_bins_cover_every_evaluation_point(self, raw):
        _fig, ax = plt.subplots()

        plot_mape_by_prediction(raw, ax=ax, n_bins=5)

        line = ax.lines[0]
        assert len(line.get_xdata()) == 5
        plt.close("all")

    def test_the_reference_line_is_the_overall_mean(self, raw):
        _fig, ax = plt.subplots()

        plot_mape_by_prediction(raw, ax=ax)

        # The dotted horizontal is the last line drawn.
        assert ax.lines[-1].get_ydata()[0] == pytest.approx(
            float(np.mean(raw["mape_per_point"]))
        )
        plt.close("all")

    def test_rejects_a_single_bin(self, raw):
        with pytest.raises(ValueError, match="at least 2"):
            plot_mape_by_prediction(raw, n_bins=1)

    def test_works_for_classification(self):
        """Labels have no variance, so the statistic is disagreement instead."""
        X, y = make_classification(
            n_samples=250, n_features=5, n_informative=3, random_state=0
        )
        result = bootstrap_predictions(
            lambda: DecisionTreeClassifier(max_depth=6, random_state=0),
            X[:180],
            y[:180],
            X[180:],
            task="categorical",
            n_bootstrap=8,
            random_state=0,
        )
        _fig, ax = plt.subplots()

        plot_mape_by_prediction(result, ax=ax, n_bins=3)

        assert "disagreement" in ax.get_ylabel()
        plt.close("all")


class TestFrontierPlot:
    """Two families on one set of axes — the comparison the package exists for."""

    @pytest.fixture
    def frontiers(self, regression_data):
        X, y = regression_data
        common = {"X": X, "y": y, "n_bootstrap": 8, "random_state": 0}
        return {
            "CART": stability_frontier(
                lambda **kw: DecisionTreeRegressor(max_depth=5, random_state=0, **kw),
                {"ccp_alpha": [0.0, 1.0, 100.0]},
                **common,
            )
        }

    def test_draws_the_frontier_as_a_line_and_the_rest_as_hollow_marks(self, frontiers):
        _fig, ax = plt.subplots()

        plot_stability_frontier(frontiers, ax=ax, annotate=False)

        drawn = len(ax.lines[0].get_xdata())
        assert drawn == len(frontiers["CART"]["frontier"])
        plt.close("all")

    def test_accuracy_is_the_vertical_axis(self, frontiers):
        """Flipping the axes would invert the reading of the plot."""
        _fig, ax = plt.subplots()

        plot_stability_frontier(frontiers, ax=ax, annotate=False)

        assert set(ax.lines[0].get_ydata()) == {
            p["accuracy"] for p in frontiers["CART"]["frontier"]
        }
        plt.close("all")

    def test_can_plot_mape_instead_of_variance(self, frontiers):
        _fig, ax = plt.subplots()

        plot_stability_frontier(frontiers, ax=ax, metric="mape", annotate=False)

        assert "absolute" in ax.get_xlabel()
        plt.close("all")

    def test_rejects_an_unknown_metric(self, frontiers):
        with pytest.raises(ValueError, match="instability"):
            plot_stability_frontier(frontiers, metric="nonsense")
