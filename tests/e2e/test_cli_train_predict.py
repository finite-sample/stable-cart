import json
import os
import shutil
import subprocess
from pathlib import Path
import pytest


@pytest.mark.e2e
@pytest.mark.slow
def test_cli_train_then_predict_smoke(tmp_path):
    """
    If your package exposes a CLI (e.g., `yourpkg train ...`), smoke test it.
    This test will SKIP if the CLI isn't on PATH.
    """
    cli = shutil.which("yourpkg") or shutil.which("yourcli")
    if not cli:
        pytest.skip("CLI not found on PATH; skipping CLI E2E.")

    data = tmp_path / "data.csv"
    # Tiny CSV; your CLI should accept this schema (adjust if needed)
    data.write_text("x,y\n0.0,0.0\n1.0,1.0\n2.0,4.0\n3.0,9.0\n")

    model_dir = tmp_path / "model"
    pred_out = tmp_path / "pred.csv"

    # Train
    r1 = subprocess.run(
        [cli, "train", "--data", str(data), "--out", str(model_dir)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert r1.returncode == 0, f"Train failed: {r1.stderr}"
    assert model_dir.exists()

    # Predict
    r2 = subprocess.run(
        [cli, "predict", "--data", str(data), "--model", str(model_dir), "--out", str(pred_out)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert r2.returncode == 0, f"Predict failed: {r2.stderr}"
    assert pred_out.exists() and pred_out.stat().st_size > 0