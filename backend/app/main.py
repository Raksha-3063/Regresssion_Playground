"""Regression Playground API: local, leakage-safe ML experimentation."""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, learning_curve, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / "storage"
DATASETS = STORE / "datasets"
MODELS = STORE / "models"
for folder in (DATASETS, MODELS):
    folder.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Regression Playground API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# In the starter, run metadata is intentionally local and ephemeral. Serialized models survive restarts.
datasets: dict[str, dict[str, Any]] = {}
experiments: dict[str, dict[str, Any]] = {}

ALGORITHMS: dict[str, dict[str, Any]] = {
    "linear": {"label": "Linear Regression", "summary": "Fits a straight-line relationship by minimizing squared error.", "when": "A strong transparent baseline when relationships are roughly linear.", "params": {"fit_intercept": {"type": "boolean", "default": True, "description": "Whether to estimate a constant baseline term."}}},
    "polynomial": {"label": "Polynomial Regression", "summary": "Expands numeric features into powers and interactions, then fits a regularized linear model.", "when": "For smooth curved relationships with a modest number of features.", "params": {"degree": {"type": "integer", "default": 2, "min": 1, "max": 6, "description": "Highest polynomial power. Higher values capture curvature but can overfit."}, "alpha": {"type": "number", "default": 1.0, "min": 0, "max": 100, "description": "Ridge regularization. Higher values shrink coefficients and reduce overfitting."}}},
    "tree": {"label": "Decision Tree", "summary": "Recursively splits features into rules that reduce prediction error.", "when": "For non-linear, interpretable decision rules.", "params": {"max_depth": {"type": "integer", "default": 6, "min": 1, "max": 40, "description": "Tree depth. Higher depth is more flexible and more likely to overfit."}, "min_samples_leaf": {"type": "integer", "default": 2, "min": 1, "max": 30, "description": "Samples required in each leaf. Higher values smooth the model."}}},
    "forest": {"label": "Random Forest", "summary": "Averages many randomized decision trees to reduce variance.", "when": "A robust default for tabular non-linear data.", "params": {"n_estimators": {"type": "integer", "default": 200, "min": 20, "max": 1000, "description": "Number of trees. More trees stabilize results but take longer."}, "max_depth": {"type": "integer", "default": 10, "min": 1, "max": 40, "description": "Maximum tree depth. Deeper trees fit more detail."}, "min_samples_leaf": {"type": "integer", "default": 1, "min": 1, "max": 30, "description": "Minimum samples in a leaf. Larger values regularize trees."}}},
    "knn": {"label": "K-Nearest Neighbors", "summary": "Predicts from the average of nearby training observations.", "when": "For small, local-pattern datasets after scaling features.", "params": {"n_neighbors": {"type": "integer", "default": 5, "min": 1, "max": 50, "description": "Neighbors to average. More neighbors give smoother, higher-bias predictions."}, "weights": {"type": "select", "default": "distance", "options": ["uniform", "distance"], "description": "Distance weighting makes closer observations count more."}}},
    "svr": {"label": "Support Vector Regressor", "summary": "Finds a margin-tolerant function; an RBF kernel captures non-linear patterns.", "when": "For scaled, medium-sized datasets with complex smooth relationships.", "params": {"C": {"type": "number", "default": 10.0, "min": 0.01, "max": 1000, "description": "Penalty for errors. Higher C fits training data more closely and may overfit."}, "epsilon": {"type": "number", "default": 0.1, "min": 0.001, "max": 2, "description": "Error-insensitive tube. Higher epsilon makes a simpler, less sensitive model."}, "gamma": {"type": "select", "default": "scale", "options": ["scale", "auto"], "description": "Kernel influence range. Larger effective gamma creates more local curves."}}},
    "bagging": {"label": "Bagging Regressor", "summary": "Trains estimators on bootstrap samples and averages their predictions.", "when": "To stabilize a high-variance tree model.", "params": {"n_estimators": {"type": "integer", "default": 100, "min": 10, "max": 500, "description": "Number of bootstrap estimators. More estimators reduce variance at more cost."}, "max_samples": {"type": "number", "default": 1.0, "min": 0.1, "max": 1, "description": "Fraction of training rows per estimator. Lower values diversify estimators."}}},
    "voting": {"label": "Voting Regressor", "summary": "Averages predictions from complementary base estimators.", "when": "When diverse individual models perform well in different regions.", "params": {}},
}

TUNING_GRIDS: dict[str, dict[str, list[Any]]] = {
    "linear": {"model__fit_intercept": [True, False]},
    "polynomial": {"model__poly__degree": [1, 2, 3], "model__ridge__alpha": [0.01, 0.1, 1.0, 10.0]},
    "tree": {"model__max_depth": [3, 6, 10, None], "model__min_samples_leaf": [1, 2, 5, 10]},
    "forest": {"model__n_estimators": [100, 250], "model__max_depth": [None, 8, 16], "model__min_samples_leaf": [1, 2, 5]},
    "knn": {"model__n_neighbors": [3, 5, 9, 15], "model__weights": ["uniform", "distance"]},
    "svr": {"model__C": [0.1, 1, 10, 100], "model__epsilon": [0.05, 0.1, 0.2], "model__gamma": ["scale", "auto"]},
    "bagging": {"model__n_estimators": [50, 150, 300], "model__max_samples": [0.5, 0.8, 1.0]},
    "voting": {},
}

class TrainRequest(BaseModel):
    dataset_id: str
    target: str
    algorithm: str
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    test_size: float = Field(default=0.2, ge=0.1, le=0.4)
    random_state: int = Field(default=42, ge=0, le=999999)
    scale_numeric: bool = True
    select_k_best: int | None = Field(default=None, ge=1)

class TuneRequest(TrainRequest):
    method: Literal["grid", "random"] = "random"
    folds: int = Field(default=5, ge=2, le=10)
    iterations: int = Field(default=12, ge=2, le=40)

class PredictRequest(BaseModel):
    rows: list[dict[str, Any]]

def fail(detail: str, code: int = 400) -> None:
    raise HTTPException(status_code=code, detail=detail)

def get_frame(dataset_id: str) -> pd.DataFrame:
    record = datasets.get(dataset_id)
    if not record:
        # enables predictions after an API restart only if metadata is re-uploaded; intentional starter limitation
        fail("Dataset session was not found. Upload the CSV again.", 404)
    return pd.read_pickle(record["path"])

def profile(frame: pd.DataFrame) -> dict[str, Any]:
    stats = frame.describe(include="all").replace({np.nan: None}).to_dict()
    return {
        "rows": int(frame.shape[0]), "columns": int(frame.shape[1]),
        "duplicates": int(frame.duplicated().sum()),
        "missing": {str(k): int(v) for k, v in frame.isna().sum().items()},
        "columns_info": [{"name": str(c), "dtype": str(frame[c].dtype), "missing": int(frame[c].isna().sum()), "unique": int(frame[c].nunique(dropna=True))} for c in frame.columns],
        "preview": json.loads(frame.head(12).replace({np.nan: None}).to_json(orient="records")),
        "statistics": stats,
    }

def make_preprocessor(X: pd.DataFrame, scale: bool, k_best: int | None) -> ColumnTransformer | Pipeline:
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    num_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale: num_steps.append(("scaler", StandardScaler()))
    cat_steps = [("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    transformer = ColumnTransformer([("numeric", Pipeline(num_steps), numeric), ("categorical", Pipeline(cat_steps), categorical)], remainder="drop")
    if k_best:
        return Pipeline([("columns", transformer), ("selector", SelectKBest(f_regression, k=k_best))])
    return transformer

def make_model(name: str, p: dict[str, Any], seed: int) -> Any:
    if name == "linear": return LinearRegression(**p)
    if name == "polynomial":
        return Pipeline([("poly", PolynomialFeatures(degree=int(p.get("degree", 2)), include_bias=False)), ("ridge", Ridge(alpha=float(p.get("alpha", 1.0))))])
    if name == "tree": return DecisionTreeRegressor(random_state=seed, **p)
    if name == "forest": return RandomForestRegressor(random_state=seed, n_jobs=-1, **p)
    if name == "knn": return KNeighborsRegressor(**p)
    if name == "svr": return SVR(**p)
    if name == "bagging": return BaggingRegressor(estimator=DecisionTreeRegressor(random_state=seed), random_state=seed, n_jobs=-1, **p)
    if name == "voting":
        return VotingRegressor([("ridge", Ridge(alpha=1.0)), ("tree", DecisionTreeRegressor(max_depth=7, random_state=seed)), ("forest", RandomForestRegressor(n_estimators=120, random_state=seed, n_jobs=-1))], n_jobs=-1)
    fail(f"Unknown algorithm '{name}'.")

def prepare(request: TrainRequest) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Pipeline]:
    if request.algorithm not in ALGORITHMS: fail("Select a supported regression algorithm.")
    frame = get_frame(request.dataset_id)
    if request.target not in frame.columns: fail("Target column does not exist in this dataset.")
    y = pd.to_numeric(frame[request.target], errors="coerce")
    usable = y.notna()
    if usable.sum() < 12: fail("The target needs at least 12 numeric, non-empty rows.")
    X = frame.drop(columns=[request.target]).loc[usable].copy()
    y = y.loc[usable]
    if X.shape[1] == 0: fail("Choose a dataset with at least one feature column.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=request.test_size, random_state=request.random_state)
    if request.select_k_best and request.select_k_best > 0:
        # K cannot exceed transformed feature count; fitting will return a descriptive error below where necessary.
        k_best = request.select_k_best
    else: k_best = None
    pipe = Pipeline([("preprocessor", make_preprocessor(X_train, request.scale_numeric, k_best)), ("model", make_model(request.algorithm, request.hyperparameters, request.random_state))])
    return X_train, X_test, y_train, y_test, pipe

def metrics(y_true: pd.Series, pred: np.ndarray, n_features: int) -> dict[str, float | None]:
    score = r2_score(y_true, pred)
    n = len(y_true)
    adjusted = 1 - (1 - score) * (n - 1) / max(n - n_features - 1, 1)
    return {"r2": round(float(score), 5), "adjusted_r2": round(float(adjusted), 5) if n > n_features + 1 else None, "mae": round(float(mean_absolute_error(y_true, pred)), 5), "mse": round(float(mean_squared_error(y_true, pred)), 5), "rmse": round(float(mean_squared_error(y_true, pred) ** 0.5), 5)}

def run_experiment(request: TrainRequest, pipeline: Pipeline, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, tuning: dict[str, Any] | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    try: pipeline.fit(X_train, y_train)
    except ValueError as exc: fail(f"Could not fit this configuration: {exc}")
    elapsed = time.perf_counter() - started
    pred = pipeline.predict(X_test)
    experiment_id = str(uuid.uuid4())
    artifact = MODELS / f"{experiment_id}.joblib"
    joblib.dump({"pipeline": pipeline, "features": list(X_train.columns)}, artifact)
    result = {"id": experiment_id, "created_at": datetime.now(timezone.utc).isoformat(), "dataset_id": request.dataset_id, "target": request.target, "algorithm": request.algorithm, "algorithm_label": ALGORITHMS[request.algorithm]["label"], "hyperparameters": request.hyperparameters, "metrics": metrics(y_test, pred, X_train.shape[1]), "training_seconds": round(elapsed, 4), "test_rows": len(y_test), "actual": [round(float(x), 6) for x in y_test], "predicted": [round(float(x), 6) for x in pred], "residuals": [round(float(a - b), 6) for a, b in zip(y_test, pred)], "tuning": tuning, "artifact": str(artifact)}
    # Learning curves use only the training partition.
    if len(y_train) >= 20:
        try:
            sizes, train_scores, valid_scores = learning_curve(pipeline, X_train, y_train, cv=min(5, max(2, len(y_train)//5)), scoring="r2", train_sizes=np.linspace(.3, 1, 4), n_jobs=-1)
            result["learning_curve"] = {"sizes": sizes.tolist(), "train": np.nanmean(train_scores, axis=1).round(4).tolist(), "validation": np.nanmean(valid_scores, axis=1).round(4).tolist()}
        except Exception: result["learning_curve"] = None
    experiments[experiment_id] = result
    return public_experiment(result)

def public_experiment(e: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in e.items() if k not in {"artifact", "dataset_id"}}

@app.get("/health")
def health() -> dict[str, str]: return {"status": "ok"}

@app.get("/algorithms")
def algorithms() -> dict[str, Any]: return ALGORITHMS

@app.post("/datasets")
async def upload_dataset(file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.filename or not file.filename.lower().endswith(".csv"): fail("Upload a CSV file.")
    content = await file.read()
    if len(content) > 15 * 1024 * 1024: fail("CSV exceeds the 15 MB upload limit.", 413)
    try:
        from io import BytesIO
        frame = pd.read_csv(BytesIO(content))
    except Exception as exc: fail(f"CSV could not be parsed: {exc}")
    if frame.empty or frame.shape[1] < 2: fail("CSV must have rows and at least two columns.")
    if frame.shape[0] > 200_000: fail("CSV has more than the 200,000-row starter limit.", 413)
    frame.columns = [str(c).strip() for c in frame.columns]
    if len(set(frame.columns)) != len(frame.columns): fail("CSV columns must have unique names.")
    dataset_id = str(uuid.uuid4())
    path = DATASETS / f"{dataset_id}.pkl"
    frame.to_pickle(path)
    datasets[dataset_id] = {"path": path, "name": file.filename}
    return {"id": dataset_id, "name": file.filename, **profile(frame)}

@app.get("/datasets/{dataset_id}")
def dataset_detail(dataset_id: str) -> dict[str, Any]:
    frame = get_frame(dataset_id)
    return {"id": dataset_id, "name": datasets[dataset_id]["name"], **profile(frame)}

@app.post("/experiments/train")
def train(request: TrainRequest) -> dict[str, Any]:
    X_train, X_test, y_train, y_test, pipeline = prepare(request)
    return run_experiment(request, pipeline, X_train, X_test, y_train, y_test)

@app.post("/experiments/tune")
def tune(request: TuneRequest) -> dict[str, Any]:
    X_train, X_test, y_train, y_test, pipeline = prepare(request)
    grid = TUNING_GRIDS[request.algorithm]
    if not grid: fail("Voting Regressor uses a curated ensemble in this version and has no exposed search grid.")
    folds = min(request.folds, max(2, len(y_train)//4))
    search: GridSearchCV | RandomizedSearchCV
    if request.method == "grid": search = GridSearchCV(pipeline, grid, scoring="r2", cv=folds, n_jobs=-1, error_score="raise")
    else:
        possible = int(np.prod([len(v) for v in grid.values()]))
        search = RandomizedSearchCV(pipeline, grid, n_iter=min(request.iterations, possible), scoring="r2", cv=folds, random_state=request.random_state, n_jobs=-1, error_score="raise")
    started = time.perf_counter()
    try: search.fit(X_train, y_train)
    except ValueError as exc: fail(f"Tuning could not run: {exc}")
    request.hyperparameters = {key.replace("model__", ""): value for key, value in search.best_params_.items()}
    return run_experiment(request, search.best_estimator_, X_train, X_test, y_train, y_test, {"method": request.method, "folds": folds, "best_cv_r2": round(float(search.best_score_), 5), "best_params": request.hyperparameters, "search_seconds": round(time.perf_counter() - started, 4), "why": "This configuration achieved the highest mean R² across cross-validation folds on the training split; the held-out test split was evaluated only after selection."})

@app.get("/experiments")
def list_experiments() -> list[dict[str, Any]]:
    return [public_experiment(e) for e in sorted(experiments.values(), key=lambda x: x["created_at"], reverse=True)]

@app.get("/experiments/{experiment_id}/download")
def download_model(experiment_id: str) -> FileResponse:
    experiment = experiments.get(experiment_id)
    if not experiment: fail("Experiment not found.", 404)
    return FileResponse(experiment["artifact"], filename=f"regression-model-{experiment_id[:8]}.joblib", media_type="application/octet-stream")

@app.post("/experiments/{experiment_id}/predict")
def predict(experiment_id: str, request: PredictRequest) -> dict[str, Any]:
    experiment = experiments.get(experiment_id)
    if not experiment: fail("Experiment not found.", 404)
    if not request.rows: fail("Add at least one prediction row.")
    saved = joblib.load(experiment["artifact"])
    expected = saved["features"]
    incoming = pd.DataFrame(request.rows)
    missing = [c for c in expected if c not in incoming.columns]
    if missing: fail(f"Missing feature columns: {', '.join(missing)}")
    try: output = saved["pipeline"].predict(incoming[expected])
    except ValueError as exc: fail(f"Could not predict: {exc}")
    return {"predictions": [round(float(x), 6) for x in output]}
