# Module 27 - Fairness Evaluation with Fairlearn

## Overview

In this exercise you will evaluate a hospital readmission classifier for fairness across demographic groups. You will train the model, then use Fairlearn's `MetricFrame` to measure accuracy, precision, and recall broken down by gender and race, compute demographic parity difference and equalized odds difference for each attribute, and implement a fairness gate that blocks promotion if either metric exceeds its threshold for either attribute.


## 1. Complete and run `train.py`

Complete the TODOs in `train.py` to prepare the dataset before training. The model training and saving logic is already provided.

```bash
python train.py
```

This produces: `model.joblib`, `X_test.npy`, `y_test.npy`, `sf_test.csv`.

---

## 2. Complete `check_fairness.py`

Complete the TODOs to load the trained model, evaluate fairness across gender subgroups, and implement the fairness gate. 

Read more: https://fairlearn.org/main/user_guide/

---

## 3. Run the fairness check

```bash
python check_fairness.py
```

This also writes `fairness_report.json` with the overall accuracy, the
per-group breakdown, and the gender and race fairness gaps. If any metric
exceeds its threshold, the script exits with code 1.

---

## 4. Wire it into a CI pipeline

This fairness gate is meant to run as part of a CI pipeline, as two jobs:

- a `train` job that runs `train.py` and uploads `model.joblib`, `X_test.npy`,
  `y_test.npy`, and `sf_test.csv` as artifacts
- a `check-fairness` job that downloads those artifacts and runs
  `check_fairness.py`, uploading `fairness_report.json` as an artifact

If `check_fairness.py` exits with code 1, the `check-fairness` job — and the
pipeline — fails, blocking the model from moving on to deployment.

