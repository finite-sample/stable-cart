"""Which documented parameters can actually change a prediction?

A parameter that no input can move is worse than a missing one: it is a promise
in the docstring that the code does not keep. This script finds them by
experiment rather than by reading, because reading is what let them accumulate.

The trap in a naive version of this check is gating. Many parameters only do
anything when some ``enable_*`` flag is on, so perturbing one parameter at a
time from the defaults reports them inert when they are merely switched off.
Each parameter is therefore perturbed under three contexts:

1. the estimator's own defaults,
2. every boolean in the same ``# === SECTION ===`` of the signature set True,
3. every boolean in the whole signature set True.

A parameter is **live** if it changes at least one prediction under at least one
context, on at least one dataset. Anything else is inert, and inert is the
evidence for deleting it.

Run: ``uv run python scripts/param_effect.py``
"""

import argparse
import inspect
import json
import re
import sys
import typing
import warnings
from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification, make_friedman1, make_regression

from stable_cart import (
    BootstrapVariancePenalizedTree,
    CentroidTree,
    LessGreedyHybridTree,
    RobustPrefixHonestTree,
    StableTree,
)

warnings.filterwarnings("ignore")

# Parameters whose effect is not in question: the task itself, the seed, and the
# size limits every tree has. Perturbing them proves nothing about the design.
STRUCTURAL = {
    "task",
    "random_state",
    "max_depth",
    "min_samples_split",
    "min_samples_leaf",
    "base_tree_class",
    "base_params",
}

SECTION_RE = re.compile(r"#\s*===\s*(.+?)\s*===")
INIT_ASSIGN_RE = re.compile(r"^\s*self\.\w+(\s*:[^=]+)?\s*=\s*[\w.]+\s*$", re.M)


def _reachable_source() -> str:
    """Source of every package module importing ``stable_cart`` pulls in.

    Anything not reachable from the public package cannot be part of a
    parameter's story, and including it would credit a parameter for a read in
    code no user ever executes.
    """
    chunks = []
    for name, module in list(sys.modules.items()):
        if not name.startswith("stable_cart"):
            continue
        path = getattr(module, "__file__", None)
        if path:
            chunks.append(Path(path).read_text())
    return "\n".join(chunks)


def _read_counts(source: str, names: set[str], cls: type) -> dict[str, int]:
    """
    How often each parameter is read outside a plain ``self.x = x`` line.

    A parameter the class renames on the way to its base gets credited with the
    reads of the name it is renamed to. Counting only the exposed name would
    report a live parameter as unread and put it on the deletion list — which is
    exactly what happened to ``top_levels``, the signature feature of
    ``RobustPrefixHonestTree``.
    """
    body = INIT_ASSIGN_RE.sub("", source)
    aliases = getattr(cls, "_PARAM_ALIASES", {})

    def count(name: str) -> int:
        targets = aliases.get(name, ())
        targets = (targets,) if isinstance(targets, str) else targets
        return sum(len(re.findall(rf"self\.{n}\b", body)) for n in (name, *targets))

    return {n: count(n) for n in names}


def _sections(cls: type) -> dict[str, str]:
    """Map each parameter to the signature section it is declared under."""
    try:
        source = inspect.getsource(cls.__init__)
    except (OSError, TypeError):
        return {}
    out, current = {}, "UNSECTIONED"
    for line in source.splitlines():
        marker = SECTION_RE.search(line)
        if marker:
            current = marker.group(1)
            continue
        name = re.match(r"\s*([a-z_][a-z0-9_]*)\s*[:=]", line)
        if name and name.group(1) != "self":
            out[name.group(1)] = current
    return out


# Parameters constrained to sum to one cannot be perturbed alone — the
# constructor rejects the result — so they are perturbed as a set. Without this
# every one of them reports inert for the wrong reason.
JOINT = {
    ("split_frac", "val_frac", "est_frac"): [
        {"split_frac": 0.4, "val_frac": 0.3, "est_frac": 0.3},
        {"split_frac": 0.8, "val_frac": 0.1, "est_frac": 0.1},
    ]
}


def _joint_for(name: str) -> list[dict]:
    """Perturbations to apply when a parameter cannot legally move on its own."""
    for group, settings in JOINT.items():
        if name in group:
            return [s for s in settings if name in s]
    return []


def _alternatives(param: inspect.Parameter) -> list:
    """Values to try for one parameter, given its default and annotation."""
    default = param.default
    if isinstance(default, bool):
        return [not default]
    if isinstance(default, int):
        return sorted({max(1, default * 2 + 1), max(1, default // 2)} - {default})
    if isinstance(default, float):
        return sorted({0.0, default + 0.5, default * 3 + 1.0} - {default})
    if isinstance(default, str):
        # Literal["a", "b", "c"] — sweep the alternatives the type allows.
        annotation = param.annotation
        options = typing.get_args(annotation)
        return [o for o in options if isinstance(o, str) and o != default]
    if isinstance(default, tuple) and all(isinstance(v, float) for v in default):
        return [(0.1, 0.9)] if default != (0.1, 0.9) else [(0.05, 0.95)]
    return []


def _boolean_params(sig: inspect.Signature) -> list[str]:
    return [
        name
        for name, p in sig.parameters.items()
        if isinstance(p.default, bool) and name not in STRUCTURAL
    ]


def _contexts(sig: inspect.Signature, sections: dict[str, str]) -> dict[str, dict]:
    """Named parameter contexts under which every parameter gets retested."""
    bools = _boolean_params(sig)
    all_on = dict.fromkeys(bools, True)
    by_section: dict[str, dict] = {}
    for name in bools:
        by_section.setdefault(sections.get(name, "UNSECTIONED"), {})[name] = True
    out = {"defaults": {}, "all_flags_on": all_on}
    for section, flags in by_section.items():
        out[f"section:{section}"] = flags
    return out


def _predict(cls, task, X, y, kwargs):
    return cls(task=task, random_state=42, **kwargs).fit(X, y).predict(X)


def _scan(cls, task, datasets, verbose):
    sig = inspect.signature(cls.__init__)
    sections = _sections(cls)
    contexts = _contexts(sig, sections)

    baselines = {}
    for label, ctx in contexts.items():
        try:
            baselines[label] = [_predict(cls, task, X, y, ctx) for X, y in datasets]
        except Exception:
            baselines[label] = None

    live, inert, untestable, fits = {}, {}, {}, 0
    for name, param in sig.parameters.items():
        if name == "self" or name in STRUCTURAL:
            continue
        if param.default is inspect.Parameter.empty:
            continue
        joint = _joint_for(name)
        overrides = joint or [{name: alt} for alt in _alternatives(param)]
        if not overrides:
            continue

        where, attempted, raised = None, 0, 0
        for label, ctx in contexts.items():
            base = baselines[label]
            if base is None:
                continue
            # Section context for the parameter's own section is where a gated
            # parameter has its best chance; the others are cheap insurance.
            for override in overrides:
                kwargs = {**ctx, **override}
                attempted += 1
                try:
                    for (X, y), b in zip(datasets, base, strict=True):
                        got = _predict(cls, task, X, y, kwargs)
                        fits += 1
                        if not np.array_equal(b, got):
                            where = (label, override[name])
                            break
                except Exception:
                    raised += 1
                    continue
                if where:
                    break
            if where:
                break

        record = {
            "default": repr(param.default),
            "section": sections.get(name, "UNSECTIONED"),
            "moved_under": where[0] if where else None,
            "moved_to": repr(where[1]) if where else None,
        }
        if where:
            live[name] = record
        elif raised == attempted:
            # Every legal-looking perturbation was rejected by the constructor.
            # That is a statement about validation, not about the parameter.
            untestable[name] = record
        else:
            inert[name] = record

    if verbose:
        print(
            f"{cls.__name__} [{task}] — {len(live)} live, {len(inert)} inert, "
            f"{len(untestable)} untestable"
        )
        for name, info in inert.items():
            print(f"    [INERT]      {name:38s} default={info['default']}")
        for name, info in untestable.items():
            print(f"    [UNTESTABLE] {name:38s} default={info['default']}")
    return {"live": live, "inert": inert, "untestable": untestable, "n_fits": fits}


def main():
    """Scan every estimator on both tasks and write the verdict per parameter."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/param_effect")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--verdicts-only",
        action="store_true",
        help="re-derive the verdicts from a previous run's per_task.json",
    )
    args = parser.parse_args()

    Xc1, yc1 = make_classification(
        n_samples=500, n_features=10, n_informative=6, random_state=0
    )
    Xc2, yc2 = make_classification(
        n_samples=400,
        n_features=6,
        n_informative=4,
        n_clusters_per_class=2,
        random_state=11,
    )
    Xr1, yr1 = make_regression(
        n_samples=500, n_features=10, n_informative=6, noise=2.0, random_state=0
    )
    Xr2, yr2 = make_friedman1(n_samples=400, noise=1.0, random_state=5)
    clf, reg = [(Xc1, yc1), (Xc2, yc2)], [(Xr1, yr1), (Xr2, yr2)]

    cases = [
        (LessGreedyHybridTree, "classification", clf),
        (LessGreedyHybridTree, "regression", reg),
        (BootstrapVariancePenalizedTree, "classification", clf),
        (BootstrapVariancePenalizedTree, "regression", reg),
        (RobustPrefixHonestTree, "classification", clf),
        (RobustPrefixHonestTree, "regression", reg),
        (CentroidTree, "classification", clf),
        (CentroidTree, "regression", reg),
        (StableTree, "classification", clf),
        (StableTree, "regression", reg),
    ]

    classes = {cls.__name__: cls for cls, _task, _data in cases}
    report: dict[str, dict] = {}
    cached = Path(args.output) / "per_task.json"
    if args.verdicts_only:
        report = json.loads(cached.read_text())
    else:
        for cls, task, datasets in cases:
            report.setdefault(cls.__name__, {})[task] = _scan(
                cls, task, datasets, not args.quiet
            )

    # A parameter is only inert for an estimator if it is inert on *both* tasks.
    # Deletion needs a second, independent reason: that no reachable code reads
    # it. An experiment alone can be an accident of these two datasets; a grep
    # alone misses everything read through a strategy object.
    source = _reachable_source()
    print()
    print(f"{'estimator':32s} {'live':>6s} {'inert':>7s} {'unread':>7s} {'delete':>7s}")
    print("-" * 66)
    summary = {}
    for name, tasks in report.items():
        inert = set.intersection(*(set(t["inert"]) for t in tasks.values()))
        live = set.union(*(set(t["live"]) for t in tasks.values()))
        inert -= live
        reads = _read_counts(source, inert, classes[name])
        unread = {n for n, c in reads.items() if c == 0}
        summary[name] = {
            "delete": sorted(unread),
            "inert_but_read": {n: reads[n] for n in sorted(inert - unread)},
        }
        print(
            f"{name:32s} {len(live):6d} {len(inert):7d} {len(unread):7d} {len(unread):7d}"
        )
        if not args.quiet:
            if unread:
                print("    delete:        " + ", ".join(sorted(unread)))
            if inert - unread:
                print(
                    "    inert but read: "
                    + ", ".join(f"{n}({reads[n]})" for n in sorted(inert - unread))
                )

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if not args.verdicts_only:
        (out / "per_task.json").write_text(json.dumps(report, indent=2))
    (out / "verdicts.json").write_text(json.dumps(summary, indent=2))
    print(f"\nCreated: {out / 'verdicts.json'}")


if __name__ == "__main__":
    main()
