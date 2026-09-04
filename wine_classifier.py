"""Repeatable SVM classification workflow for the Wine Recognition dataset."""

from __future__ import annotations

from dataclasses import dataclass


TEST_SIZE = 0.25
RANDOM_STATE = 42
MINIMUM_ACCURACY = 0.90


@dataclass(frozen=True)
class ClassificationResult:
    """Evidence returned by one deterministic training/evaluation run."""

    feature_count: int
    category_count: int
    sample_count: int
    training_count: int
    evaluation_count: int
    training_indices: tuple[int, ...]
    evaluation_indices: tuple[int, ...]
    accuracy: float

    @property
    def meets_acceptance(self) -> bool:
        """Whether the held-out accuracy satisfies the story requirement."""

        return self.accuracy >= MINIMUM_ACCURACY


def run_classification() -> ClassificationResult:
    """Train and evaluate the classifier using a fixed stratified split."""

    import numpy as np
    from sklearn.datasets import load_wine
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    wine = load_wine()
    indices = np.arange(len(wine.target))
    training_indices, evaluation_indices = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=wine.target,
    )

    model = make_pipeline(StandardScaler(), SVC())
    model.fit(wine.data[training_indices], wine.target[training_indices])
    predictions = model.predict(wine.data[evaluation_indices])

    return ClassificationResult(
        feature_count=int(wine.data.shape[1]),
        category_count=int(len(wine.target_names)),
        sample_count=int(len(wine.target)),
        training_count=len(training_indices),
        evaluation_count=len(evaluation_indices),
        training_indices=tuple(int(index) for index in training_indices),
        evaluation_indices=tuple(int(index) for index in evaluation_indices),
        accuracy=float(accuracy_score(wine.target[evaluation_indices], predictions)),
    )


def main() -> int:
    """Run the workflow and print machine- and human-readable evidence."""

    result = run_classification()
    print(
        "Wine Recognition dataset: "
        f"{result.sample_count} samples, {result.feature_count} measurements, "
        f"{result.category_count} recognized categories"
    )
    print(
        f"Fixed split: {result.training_count} training / "
        f"{result.evaluation_count} evaluation samples (random_state={RANDOM_STATE})"
    )
    print(f"Held-out evaluation accuracy: {result.accuracy:.2%}")
    print(f"Evaluation indices: {','.join(map(str, result.evaluation_indices))}")
    print(f"Classification: {'PASS' if result.meets_acceptance else 'FAIL'}")
    return 0 if result.meets_acceptance else 1


if __name__ == "__main__":
    raise SystemExit(main())
