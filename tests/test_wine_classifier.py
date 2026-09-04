import subprocess
import sys

from wine_classifier import run_classification


def test_classifier_uses_all_measurements_and_meets_accuracy_target():
    result = run_classification()

    assert result.feature_count == 13
    assert result.category_count == 3
    assert result.training_count + result.evaluation_count == result.sample_count
    assert result.accuracy >= 0.90


def test_classifier_split_and_accuracy_are_repeatable():
    first = run_classification()
    second = run_classification()

    assert first.training_indices == second.training_indices
    assert first.evaluation_indices == second.evaluation_indices
    assert first.accuracy == second.accuracy


def test_classifier_cli_runs_without_interaction_and_reports_success():
    completed = subprocess.run(
        [sys.executable, "-m", "wine_classifier"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "13 measurements" in completed.stdout
    assert "Fixed split" in completed.stdout
    assert "Held-out evaluation accuracy:" in completed.stdout
    assert "Classification: PASS" in completed.stdout
