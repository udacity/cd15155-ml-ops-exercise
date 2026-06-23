# Module 15 - Model Quality Gate with Deepchecks

## Overview

In this exercise you will extend the automated training pipeline by adding a model quality gate. Before a new model is promoted to `@production`, two validation steps must pass:

1. **Champion/challenger**: the new `@dev` model must outperform the current `@production` model on validation accuracy.
2. **Quality gate**: Deepchecks must confirm the model passes a model evaluation suite covering class performance, prediction drift, simple model comparison, and weak segment performance.

---

## Setup

Start the MLflow tracking server:

```bash
mlflow server --port 5000 --host 0.0.0.0 --allowed-hosts "*"
```

Start the prefect server and configure prefect to communicate with the server
```bash
export PREFECT_UI_API_URL=$(echo $VSCODE_PROXY_URI | sed 's/{{port}}/4200/')api
prefect config set PREFECT_API_URL=http://0.0.0.0:4200/api
prefect server start --host 0.0.0.0
```

Start a prefect worker
```bashpython -m prefect worker start --pool default 
```
---

## 1. Understand the pipeline

Open `flow.py`. The pipeline runs three Prefect tasks in order:

- **Train**: fine-tunes the ViT model on the beans dataset and registers it under the `@dev` alias.
- **Validate**: runs champion/challenger comparison and Deepchecks quality checks. Only returns `True` if both pass.
- **Promote**: assigns the `@production` alias to `@dev` if validation passed.

You do not need to modify `flow.py` or `train.py`.

---

## 2. Complete `validate.py`

The champion/challenger logic is already implemented in `champion_challenger()`. You need to complete the Deepchecks quality gate in `run_quality_checks()`.

Read more about Deepchecks [VisionData](https://docs.deepchecks.com/stable/vision/usage_guides/visiondata_object.html)
Read more about the [model evaluation suite](https://docs.deepchecks.com/stable/api/generated/deepchecks.vision.suites.model_evaluation.html)

---

## 3. Run the pipeline

```bash
python flow.py
```

The pipeline will train the model, run validation (champion/challenger + Deepchecks), and promote to `@production` only if both checks pass.