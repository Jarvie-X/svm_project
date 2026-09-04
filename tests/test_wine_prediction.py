import subprocess
import sys

import pytest
from sklearn.datasets import load_wine

from wine_prediction import (
    FEATURE_NAMES,
    MeasurementValidationError,
    measurement_guidance,
    predict_measurements,
)


def test_guidance_presents_all_thirteen_model_fields():
    guidance = measurement_guidance()

    assert len(guidance) == 13
    assert tuple(field.name for field in guidance) == FEATURE_NAMES
    assert all(field.label and field.guidance for field in guidance)


def test_valid_wine_sample_produces_one_recognized_category():
    wine = load_wine()

    result = predict_measurements(wine.data[0].tolist())

    assert result.category in {"Class 0", "Class 1", "Class 2"}
    assert result.class_index in {0, 1, 2}


def test_prediction_result_contains_accuracy_input_summary_and_guidance():
    values = load_wine().data[0].tolist()

    result = predict_measurements(values)

    assert result.accuracy >= 0.90
    assert result.evaluation_count == 45
    assert result.measurement_summary == tuple(zip(FEATURE_NAMES, values))
    assert result.category in {"Class 0", "Class 1", "Class 2"}
    assert "recognized categories" in result.guidance
    assert "not a wine varietal or quality grade" in result.guidance


@pytest.mark.parametrize("values", [[], [1] * 12, [1] * 14])
def test_incomplete_or_wrong_length_entry_is_rejected(values):
    with pytest.raises(MeasurementValidationError, match="exactly 13"):
        predict_measurements(values)


@pytest.mark.parametrize("bad_value", ["", "not-a-number", "NaN", "inf", "-inf", "0", "-1"])
def test_invalid_measurement_has_an_understandable_correction(bad_value):
    values = [1] * 13
    values[0] = bad_value

    with pytest.raises(MeasurementValidationError) as caught:
        predict_measurements(values)

    assert "Alcohol" in str(caught.value)
    assert "enter" in str(caught.value).lower()


def test_mapping_reports_each_missing_field():
    with pytest.raises(MeasurementValidationError) as caught:
        predict_measurements({})

    assert all(name in str(caught.value) for name in FEATURE_NAMES)


def test_prediction_cli_reports_result_for_scripted_valid_sample():
    values = [str(value) for value in load_wine().data[0]]
    completed = subprocess.run(
        [sys.executable, "-m", "wine_prediction", "--values", *values],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "13" in completed.stdout
    assert "Input measurements supplied:" in completed.stdout
    assert all(f"- {name}:" in completed.stdout for name in FEATURE_NAMES)
    assert "Held-out test accuracy:" in completed.stdout
    assert "labeled test samples" in completed.stdout
    assert "Predicted category: Class " in completed.stdout
    assert "Category guidance:" in completed.stdout
    assert "does not guarantee an individual prediction" in completed.stdout


def test_prediction_cli_reports_correction_for_missing_values():
    completed = subprocess.run(
        [sys.executable, "-m", "wine_prediction", "--values", "1", "2"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "exactly 13" in completed.stdout
    assert "Input correction needed" in completed.stdout
