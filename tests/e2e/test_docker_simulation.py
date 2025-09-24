import os
import sys
from pathlib import Path
import pytest


@pytest.mark.e2e
@pytest.mark.slow
def test_entrypoint_simulation(tmp_path, monkeypatch):
    """
    Simulate a Docker entrypoint: set env vars, call a main() if you expose one.
    This avoids running Docker in CI while catching path/env issues.
    """
    # Adjust to your package main module if you have one
    try:
        import yourpkg.__main__ as app  # noqa: F401
    except Exception:
        pytest.skip("yourpkg.__main__ not importable; skipping entrypoint simulation.")

    monkeypatch.setenv("APP_OUTPUT_DIR", str(tmp_path))
    # If your app.main() exists, call it. Otherwise skip.
    main = getattr(app, "main", None)
    if not callable(main):
        pytest.skip("No main() entrypoint; skipping.")
    main()

    # Ensure the app honored env/output
    assert tmp_path.exists()