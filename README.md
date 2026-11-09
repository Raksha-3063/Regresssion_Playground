# Regression Playground

An interactive, full-stack regression experimentation workspace. Upload a CSV, inspect data quality, choose a target and algorithm, train leakage-safe scikit-learn pipelines, tune parameters with cross-validation, and compare saved runs.

## Quick start

```powershell
docker compose up --build
```

Open `http://localhost:5173`. The API is available at `http://localhost:8000/docs`.

For local development, create a Python virtual environment and install `backend/requirements.txt`; then run `uvicorn app.main:app --reload` from `backend`. In `frontend`, run `npm install` followed by `npm run dev`.

## Design notes

- Datasets and trained artifacts are kept locally in `backend/storage` (git-ignored).
- Training splits before fitting imputers, encoders, scalers, selectors, or model tuning. The held-out test split is not used by tuning.
- The initial build ships all eight requested estimators and a conservative, documented tuning grid for each. It is deliberately local/single-user; add authentication and persistent metadata storage before multi-user deployment.
