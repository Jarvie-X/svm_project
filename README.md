# svm_project

## Classification runtime readiness

Install the runtime dependency and run the repeatable Iris SVM workflow:

```sh
python -m pip install -r requirements.txt
PYTHONPATH=src python -m svm_project.classifier
pytest
```

The workflow uses scikit-learn's built-in Iris dataset, a stratified fixed
train/test split, and an SVM classifier. A successful run prints a JSON result
with `"status": "ready"` and the evaluation accuracy.
