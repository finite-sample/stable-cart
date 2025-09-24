"""Test configuration ensuring local modules are importable."""
from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:  # pragma: no cover - compatibility shim
    from sklearn import base as _sk_base
except Exception:  # pragma: no cover
    _sk_base = None
else:
    if not hasattr(_sk_base, "BaseClassifierMixin"):
        setattr(_sk_base, "BaseClassifierMixin", _sk_base.ClassifierMixin)
