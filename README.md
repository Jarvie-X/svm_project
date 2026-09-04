# Wine Recognition demonstration

The project includes a guided single-sample prediction entry point. It uses
all thirteen measurements from scikit-learn's Wine Recognition dataset and
returns one of its three dataset categories.

```text
python -m wine_prediction
```

For a quick scripted prediction, provide thirteen values in the displayed
order:

```text
python -m wine_prediction --values 13.2 1.78 2.14 11.2 100 2.65 2.76 0.26 1.28 4.38 1.05 3.4 1050
```

The result is an educational dataset classification, not wine quality,
safety, health, or purchasing advice. Run `python -m pytest` to verify the
readiness, classifier, and guided prediction workflows.
