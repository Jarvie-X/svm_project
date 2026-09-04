"""Release entry point for the complete Wine Recognition demonstration."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from wine_classifier import MINIMUM_ACCURACY, run_classification
from wine_prediction import (
    MEASUREMENTS,
    MeasurementValidationError,
    measurement_guidance,
    predict_measurements,
)
from wine_readiness import check_readiness


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the complete Wine Recognition demonstration without prompts"
    )
    parser.add_argument(
        "--values",
        nargs="+",
        required=True,
        metavar="VALUE",
        help="13 positive numeric values in the displayed measurement order",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run readiness, evaluation, and one complete prediction in sequence."""

    args = _parser().parse_args(argv)
    print("Wine Recognition demonstration release")
    print("Educational dataset classification using all 13 measurements.")
    print("Measurement guide:")
    for number, measurement in enumerate(measurement_guidance(), start=1):
        print(f"{number}. {measurement.label} — {measurement.guidance}")

    readiness = check_readiness()
    if not readiness.ready:
        print(f"Release readiness: FAIL ({readiness.error})")
        return 1
    print(
        "Release readiness: PASS — "
        f"{readiness.sample_count} samples, {readiness.category_count} recognized categories"
    )

    evaluation = run_classification()
    print(
        f"Held-out test accuracy: {evaluation.accuracy:.2%} "
        f"({evaluation.evaluation_count} labeled test samples; "
        f"fixed stratified 25% split, random_state=42)"
    )
    if evaluation.accuracy < MINIMUM_ACCURACY:
        print(f"Release evaluation: FAIL (minimum is {MINIMUM_ACCURACY:.0%})")
        return 1

    try:
        result = predict_measurements(args.values)
    except MeasurementValidationError as exc:
        print("Input correction needed:")
        for error in exc.errors:
            print(f"- {error}")
        return 2

    print("Input measurements supplied:")
    for measurement, (_, value) in zip(MEASUREMENTS, result.measurements):
        print(f"- {measurement.label}: {value:g}")
    print(f"Predicted category: {result.category}")
    print(f"Category guidance: {result.guidance}")
    print(
        "Accuracy summarizes performance across labeled test samples; it does "
        "not guarantee an individual prediction. This educational result is "
        "not wine quality, safety, health, or purchasing advice."
    )
    print("Demonstration: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
