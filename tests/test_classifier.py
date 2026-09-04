from svm_project.classifier import run_classification


def test_classification_workflow_loads_trains_and_evaluates() -> None:
    result = run_classification()

    assert result.dataset_samples == 150
    assert result.dataset_features == 4
    assert result.train_samples == 120
    assert result.test_samples == 30
    assert 0.0 <= result.accuracy <= 1.0
    assert result.accuracy >= 0.9
    assert result.classifier.classes_.tolist() == [0, 1, 2]


def test_classification_split_is_repeatable() -> None:
    first = run_classification(random_state=7)
    second = run_classification(random_state=7)

    assert first.accuracy == second.accuracy
    assert first.classifier.support_.tolist() == second.classifier.support_.tolist()
