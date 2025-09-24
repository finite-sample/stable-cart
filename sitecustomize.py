"""Test-time compatibility shims for optional runtime dependencies."""
from __future__ import annotations

import sys
from pathlib import Path

_src = Path(__file__).resolve().parent / 'src'
if _src.exists():
    sys.path.insert(0, str(_src))

try:  # pragma: no cover - defensive aliasing
    from sklearn import base as _sk_base
except Exception:  # pragma: no cover
    _sk_base = None

if _sk_base is not None and not hasattr(_sk_base, "BaseClassifierMixin"):
    # scikit-learn 1.7 removed BaseClassifierMixin; older code (and our tests)
    # still import it, so mirror the previous alias to ClassifierMixin.
    setattr(_sk_base, "BaseClassifierMixin", _sk_base.ClassifierMixin)
