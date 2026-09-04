import subprocess
import sys

from wine_readiness import check_readiness


def test_wine_dataset_readiness_contract():
    result = check_readiness()

    assert result.ready
    assert result.runtime_available
    assert result.dataset_loaded
    assert result.feature_count == 13
    assert result.category_count == 3
    assert result.sample_count > 0


def test_readiness_cli_passes_and_reports_evidence():
    completed = subprocess.run(
        [sys.executable, "-m", "wine_readiness"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "scikit-learn): available" in completed.stdout
    assert "13 measurements" in completed.stdout
    assert "3 recognized categories" in completed.stdout
    assert "Readiness: PASS" in completed.stdout

