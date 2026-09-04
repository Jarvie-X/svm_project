"""Readiness check for the Wine Recognition demonstration."""

from __future__ import annotations

from dataclasses import dataclass


EXPECTED_FEATURES = 13
EXPECTED_CATEGORIES = 3


@dataclass(frozen=True)
class ReadinessResult:
    """The technical readiness evidence produced by the check."""

    runtime_available: bool
    dataset_loaded: bool
    feature_count: int | None = None
    category_count: int | None = None
    sample_count: int | None = None
    error: str | None = None

    @property
    def ready(self) -> bool:
        return self.runtime_available and self.dataset_loaded


def check_readiness() -> ReadinessResult:
    """Load and validate the approved dataset and its ML runtime.

    Imports are intentionally performed inside this function so the CLI can
    report a clear failure rather than crashing with an import traceback when
    the product environment is missing scikit-learn.
    """

    try:
        import sklearn
        from sklearn.datasets import load_wine
    except ImportError as exc:
        return ReadinessResult(
            runtime_available=False,
            dataset_loaded=False,
            error=f"scikit-learn is unavailable: {exc}",
        )

    try:
        wine = load_wine()
        feature_count = int(wine.data.shape[1])
        sample_count = int(wine.data.shape[0])
        category_count = int(len(wine.target_names))
        target_count = int(len(set(wine.target)))

        if feature_count != EXPECTED_FEATURES:
            raise ValueError(
                f"expected {EXPECTED_FEATURES} measurements, found {feature_count}"
            )
        if category_count != EXPECTED_CATEGORIES or target_count != EXPECTED_CATEGORIES:
            raise ValueError(
                f"expected {EXPECTED_CATEGORIES} categories, found "
                f"{category_count} names and {target_count} target values"
            )
        if sample_count != len(wine.target):
            raise ValueError("measurement and target sample counts do not match")
    except (AttributeError, IndexError, TypeError, ValueError) as exc:
        return ReadinessResult(
            runtime_available=True,
            dataset_loaded=False,
            error=f"Wine Recognition dataset validation failed: {exc}",
        )

    return ReadinessResult(
        runtime_available=True,
        dataset_loaded=True,
        feature_count=feature_count,
        category_count=category_count,
        sample_count=sample_count,
        error=None,
    )


def main() -> int:
    """Print human-readable evidence and return a process status."""

    result = check_readiness()
    print(f"Machine-learning runtime (scikit-learn): {'available' if result.runtime_available else 'unavailable'}")
    if result.dataset_loaded:
        print(
            "Wine Recognition dataset: loaded "
            f"({result.sample_count} samples, {result.feature_count} measurements, "
            f"{result.category_count} recognized categories)"
        )
    else:
        print(f"Wine Recognition dataset: not ready ({result.error})")
    print(f"Readiness: {'PASS' if result.ready else 'FAIL'}")
    return 0 if result.ready else 1


if __name__ == "__main__":
    raise SystemExit(main())

