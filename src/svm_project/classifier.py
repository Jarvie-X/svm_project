"""Train and evaluate an SVM against scikit-learn's built-in Iris dataset."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


@dataclass(frozen=True)
class ClassificationResult:
    """The observable results of one readiness workflow execution."""

    dataset_samples: int
    dataset_features: int
    train_samples: int
    test_samples: int
    random_state: int
    accuracy: float
    classifier: SVC

    def as_dict(self) -> dict[str, object]:
        """Return serialisable readiness results (excluding the estimator)."""

        values = asdict(self)
        values.pop("classifier")
        return values


def run_classification(*, random_state: int = 42, test_size: float = 0.2) -> ClassificationResult:
    """Load Iris, train an SVM, and evaluate a deterministic holdout split.

    The explicit random state and stratification make repeated readiness runs
    use the same train/test partition while preserving all three classes.
    """

    iris = load_iris()
    features, labels = iris.data, iris.target
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    classifier = SVC()
    classifier.fit(x_train, y_train)
    predictions = classifier.predict(x_test)
    accuracy = float(accuracy_score(y_test, predictions))

    return ClassificationResult(
        dataset_samples=int(features.shape[0]),
        dataset_features=int(features.shape[1]),
        train_samples=int(x_train.shape[0]),
        test_samples=int(x_test.shape[0]),
        random_state=random_state,
        accuracy=accuracy,
        classifier=classifier,
    )


def main() -> None:
    """Run the deployment readiness check and print its result as JSON."""

    result = run_classification()
    print(json.dumps({"status": "ready", **result.as_dict()}, sort_keys=True))


if __name__ == "__main__":
    main()
