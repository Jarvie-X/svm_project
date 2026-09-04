"""Guided, single-sample entry point for the Wine Recognition classifier."""

from __future__ import annotations

import argparse
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from wine_classifier import RANDOM_STATE, run_classification


@dataclass(frozen=True)
class Measurement:
    """A field shown to a reviewer when entering a sample."""

    name: str
    label: str
    guidance: str


# These are the feature names supplied by sklearn's approved load_wine data.
# The dataset does not publish units for these values, so the guidance says so
# rather than inventing units or suggesting that users collect lab data.
MEASUREMENTS: tuple[Measurement, ...] = tuple(
    Measurement(name, name.replace("_", " ").title(),
                "Enter a positive numeric dataset value; units are not provided by the dataset.")
    for name in (
        "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium",
        "total_phenols", "flavanoids", "nonflavanoid_phenols",
        "proanthocyanins", "color_intensity", "hue", "od280/od315_of_diluted_wines",
        "proline",
    )
)
FEATURE_NAMES = tuple(measurement.name for measurement in MEASUREMENTS)


class MeasurementValidationError(ValueError):
    """Raised when a guided prediction entry cannot be classified."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(errors)
        super().__init__(" ".join(self.errors))


@dataclass(frozen=True)
class PredictionResult:
    """The explained result for one complete measurement set."""

    category: str
    class_index: int
    measurements: tuple[tuple[str, float], ...]
    accuracy: float
    evaluation_count: int
    evaluation_method: str
    guidance: str

    @property
    def measurement_summary(self) -> tuple[tuple[str, float], ...]:
        """Return the submitted values paired with their model field names."""

        return self.measurements


def measurement_guidance() -> tuple[Measurement, ...]:
    """Return all thirteen fields in the order expected by the model."""

    return MEASUREMENTS


def _values_in_field_order(values: Sequence[object] | Mapping[str, object]) -> list[object]:
    if isinstance(values, Mapping):
        missing = [name for name in FEATURE_NAMES if name not in values]
        unexpected = [str(name) for name in values if name not in FEATURE_NAMES]
        errors = [f"Missing measurement '{name}'. Enter a value." for name in missing]
        errors.extend(f"Unknown measurement '{name}'. Remove it." for name in unexpected)
        if errors:
            raise MeasurementValidationError(errors)
        return [values[name] for name in FEATURE_NAMES]
    return list(values)


def _validated_values(values: Sequence[object] | Mapping[str, object]) -> list[float]:
    raw_values = _values_in_field_order(values)
    errors: list[str] = []
    if len(raw_values) != len(MEASUREMENTS):
        errors.append(
            f"Enter exactly {len(MEASUREMENTS)} measurements; received {len(raw_values)}."
        )

    validated: list[float] = []
    for position, measurement in enumerate(MEASUREMENTS):
        if position >= len(raw_values):
            continue
        raw = raw_values[position]
        try:
            value = float(raw)
        except (TypeError, ValueError):
            errors.append(f"{measurement.label}: enter a numeric value.")
            continue
        if not math.isfinite(value):
            errors.append(f"{measurement.label}: enter a finite numeric value (not NaN or infinity).")
        elif value <= 0:
            errors.append(f"{measurement.label}: enter a value greater than zero.")
        else:
            validated.append(value)

    if errors:
        raise MeasurementValidationError(errors)
    return validated


def predict_measurements(values: Sequence[object] | Mapping[str, object]) -> PredictionResult:
    """Validate, classify, and explain one complete set of measurements."""

    validated = _validated_values(values)

    from sklearn.datasets import load_wine
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    wine = load_wine()
    model = make_pipeline(StandardScaler(), SVC(random_state=RANDOM_STATE))
    model.fit(wine.data, wine.target)
    class_index = int(model.predict([validated])[0])
    evaluation = run_classification()
    category = f"Class {class_index}"
    return PredictionResult(
        category=category,
        class_index=class_index,
        measurements=tuple(zip(FEATURE_NAMES, validated)),
        accuracy=evaluation.accuracy,
        evaluation_count=evaluation.evaluation_count,
        evaluation_method=(
            "fixed stratified 25% split of the labeled Wine Recognition dataset "
            f"(random_state={RANDOM_STATE})"
        ),
        guidance=(
            f"{category} is one of the three recognized categories in the Wine "
            "Recognition dataset. It is a dataset label, not a wine varietal "
            "or quality grade."
        ),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Guided Wine Recognition category prediction")
    parser.add_argument(
        "--values", nargs="+", metavar="VALUE",
        help="13 values in the displayed field order (omit to enter them interactively)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the guided entry flow."""

    args = _parser().parse_args(argv)
    print("Wine sample classification demonstration")
    print("Enter the 13 positive numeric dataset measurements (units are not provided).")
    print("This educational result is not wine quality, safety, health, or purchasing advice.")
    print("Measurement guide:")
    for number, measurement in enumerate(MEASUREMENTS, start=1):
        print(f"{number}. {measurement.label} — {measurement.guidance}")

    values = args.values
    if values is None:
        values = [input(f"{number}. {measurement.label}: ")
                  for number, measurement in enumerate(MEASUREMENTS, start=1)]
    try:
        result = predict_measurements(values)
    except MeasurementValidationError as exc:
        print("Input correction needed:")
        for error in exc.errors:
            print(f"- {error}")
        return 2

    print("Input measurements supplied:")
    for name, value in result.measurements:
        print(f"- {name}: {value:g}")
    print(
        f"Held-out test accuracy: {result.accuracy:.2%} "
        f"({result.evaluation_count} labeled test samples; {result.evaluation_method})"
    )
    print(f"Predicted category: {result.category}")
    print(f"Category guidance: {result.guidance}")
    print(
        "Accuracy summarizes performance across labeled test samples; it does "
        "not guarantee an individual prediction. This educational result is "
        "not wine quality, safety, health, or purchasing advice."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
