"""Smoke the documented user workflow through public package imports."""

import importlib.util
import json
import pathlib

import pytest

# examples/ is not part of the distribution, and the wheel job runs this suite
# with the repository root off sys.path on purpose -- so that `stable_cart`
# resolves to the installed wheel rather than the working tree. Load the
# example from disk instead of importing it as a package.
_EXAMPLE = pathlib.Path(__file__).resolve().parents[1] / "examples" / "user_workflow.py"
_spec = importlib.util.spec_from_file_location("stable_cart_example_workflow", _EXAMPLE)
_example = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_example)
run_workflow = _example.run_workflow


@pytest.mark.e2e
def test_user_workflow_creates_actionable_results_and_plots(tmp_path):
    result = run_workflow(tmp_path, n_bootstrap=6, n_candidates=4)

    assert result["regression"]["frontier"]["n_configurations"] == 6
    assert result["regression"]["frontier"]["n_frontier"] >= 1
    assert result["regression"]["frontier"]["test_score"] > 0.0
    assert result["regression"]["representative"]["candidate_count"] == 4
    assert result["multiclass"]["bootstrap_shape"] == [6, 54, 3]
    assert result["multiclass"]["probabilities_sum_to_one"] is True
    assert result["multiclass"]["representative_accuracy"] > 0.8
    assert len(list(tmp_path.glob("*.png"))) == 3
    assert json.loads((tmp_path / "workflow_summary.json").read_text()) == result
