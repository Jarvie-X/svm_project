import subprocess
import sys

from sklearn.datasets import load_wine


def _run(*values):
    return subprocess.run(
        [sys.executable, "-m", "wine_demo", "--values", *map(str, values)],
        check=False,
        capture_output=True,
        text=True,
    )


def test_release_flow_is_complete_and_non_interactive():
    completed = _run(*load_wine().data[0])

    assert completed.returncode == 0
    assert "Release readiness: PASS" in completed.stdout
    assert "Held-out test accuracy: 97.78%" in completed.stdout
    assert "Input measurements supplied:" in completed.stdout
    assert "Predicted category: Class 0" in completed.stdout
    assert "recognized categories" in completed.stdout
    assert "not a wine varietal or quality grade" in completed.stdout
    assert "Demonstration: PASS" in completed.stdout


def test_release_flow_repeats_deterministically():
    values = load_wine().data[0]
    first = _run(*values)
    second = _run(*values)

    assert first.returncode == second.returncode == 0
    assert first.stdout == second.stdout


def test_release_flow_rejects_invalid_input_without_prediction():
    completed = _run(1, 2)

    assert completed.returncode != 0
    assert "Input correction needed:" in completed.stdout
    assert "exactly 13" in completed.stdout
    assert "Demonstration: PASS" not in completed.stdout
