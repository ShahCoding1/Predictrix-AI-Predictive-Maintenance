# Predictrix AI — Explainable Predictive Maintenance

An end-to-end research prototype for **industrial machine failure classification**, with reproducible scikit-learn experiments, a FastAPI service, a responsive dashboard, batch inference, and model-agnostic local sensitivity explanations.

> **Research prototype only.** Not for safety-critical industrial decisions. Synthetic demonstration metrics must not be represented as validation on actual machinery.

## Quick start

Python 3.11+ recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m ml.train --demo
uvicorn backend.app.main:app --reload
```

Open http://127.0.0.1:8000 for the dashboard, or http://127.0.0.1:8000/docs for interactive API docs.

### Train on UCI AI4I 2020

Download **AI4I 2020 Predictive Maintenance Dataset** from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset). Place the CSV at `data/ai4i2020.csv`, then run:

```bash
python -m ml.train --csv data/ai4i2020.csv
```

The loader maps original UCI headers to internal feature names. Only five sensor/process features plus product type are used. Identifiers, target labels, and failure-mode labels are excluded from predictors to prevent leakage. Do not commit unlicensed or private datasets.

## ML methodology

- Stratified random 60% fit / 20% validation / 20% held-out test split, fixed seed 42.
- Preprocessing fitted on training data only, using scikit-learn Pipelines.
- Logistic Regression, Random Forest, HistGradientBoosting.
- Model selected by validation average precision (PR-AUC).
- One held-out test evaluation at the default decision threshold 0.5.
- Metrics: PR-AUC, ROC-AUC, precision, recall, F1, Brier score, confusion matrix.
- Local explanations use one-feature-at-a-time **reference perturbation**, **not SHAP**, and do not establish causation.

**Caveats:** Synthetic AI4I is not a chronological real-time sensor stream. Random splits may overstate generalization to new equipment or operating regimes. Production threshold tuning, calibration, external validation, fairness across product types, drift monitoring, and prospective trials remain future work. Reported probabilities are not necessarily calibrated.

## API

- `GET /api/health` — service and model readiness
- `GET /api/model` — experimental metrics and provenance
- `POST /api/predict` — one prediction
- `POST /api/predict/batch` — 1–500 predictions
- `POST /api/explain` — local reference perturbation analysis

Example:

```bash
curl -X POST http://localhost:8000/api/predict -H "Content-Type: application/json" -d '{"air_temperature":300,"process_temperature":311,"rotational_speed":1420,"torque":48,"tool_wear":175,"product_type":"M"}'
```

## Testing

```bash
python -m pytest -q
```

## Docker

First train a model on the host, then:

```bash
docker compose up --build
```

## Repository map

- `ml/` — dataset validation, demo generation, model training and evaluation
- `backend/app/` — FastAPI endpoints and static frontend serving
- `frontend/` — responsive no-build dashboard
- `tests/` — data, training and API tests
- `docs/` — architecture, research notes and roadmap
- `artifacts/` — locally trained models and reports (not committed)

## Planned extensions

- Real SHAP explanations with verified version compatibility and appropriate background samples
- NASA C-MAPSS run-to-failure temporal modelling and remaining-useful-life evaluation
- MLflow experiment tracking and calibration studies
- PostgreSQL, accounts and long-running job queue
- Model registry, CI security checks and managed deployment

These are **not implemented** in the current release. See `docs/ROADMAP.md`.
