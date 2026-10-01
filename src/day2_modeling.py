"""Bounded DAY2 development comparison; no holdout or external-batch scoring.

The caller supplies an already extracted, cell-level DataFrame. Importing this module
does not read data, fit a model, or create outputs. The selected model is only a
CV recommendation; freeze the winner before the authorized final evaluation.
"""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
import sklearn
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


F0 = ("log10_delta_Q_var",)
F1 = F0 + ("median_chargetime", "mean_Tavg")
REQUIRED_COLUMNS = ("batch", "cell_id", "cycle_life", "model_eligible", "charging_policy") + F1
METRICS = ("mape_pct", "mae", "rmse", "r2")
CANDIDATES = (
    ("dummy_mean", "F0", "DummyRegressor", None),
    ("linear_F0", "F0", "LinearRegression", None),
    ("linear_F1", "F1", "LinearRegression", None),
    ("ridge_F1_alpha_0.1", "F1", "Ridge", 0.1),
    ("ridge_F1_alpha_1", "F1", "Ridge", 1.0),
    ("ridge_F1_alpha_10", "F1", "Ridge", 10.0),
    ("ridge_F1_alpha_100", "F1", "Ridge", 100.0),
)
EVALUATION_STATUS = "cv_selected_pending_holdout"


def build_candidate_specs() -> list[dict[str, Any]]:
    """The bounded candidate list; edit here to make an approved grid change."""
    specs = [dict(name=n, feature_set=f, estimator=e, alpha=a, l1_ratio=None,
                  max_iter=None, tol=None) for n, f, e, a in CANDIDATES]
    specs.extend(dict(name=f"elasticnet_F1_alpha_{alpha:g}_l1_ratio_{ratio:g}",
                      feature_set="F1", estimator="ElasticNet", alpha=alpha,
                      l1_ratio=ratio, max_iter=10000, tol=1e-4)
                 for alpha in (0.1, 1.0, 10.0, 100.0) for ratio in (0.1, 0.5))
    return specs


def _require(condition: bool, message: str) -> None:
    """Leakage checks remain active even when Python runs with optimization."""
    if not condition:
        raise ValueError(message)


def _prepare(features: pd.DataFrame, seed: int):
    _require(isinstance(features, pd.DataFrame), "features must be a pandas DataFrame.")
    _require(features.columns.is_unique, "Input column names must be unique.")
    missing = sorted(set(REQUIRED_COLUMNS) - set(features.columns))
    _require(not missing, f"Missing required columns: {missing}")
    _require(len(features) > 0, "features is empty.")
    _require(
        isinstance(seed, (int, np.integer)) and not isinstance(seed, (bool, np.bool_))
        and 0 <= seed < 2**32,
        "seed must be an integer in [0, 2**32).",
    )
    input_columns = list(REQUIRED_COLUMNS)
    if "label_eligible" in features.columns:
        input_columns.append("label_eligible")
    frame = features.loc[:, input_columns].copy(deep=True)
    _require(not frame[["batch", "cell_id"]].isna().any().any(),
             "batch and cell_id cannot be missing.")
    # Canonical strings also detect collisions between, e.g., cell_id 1 and '1'.
    frame["batch"] = frame["batch"].astype(str)
    frame["cell_id"] = frame["cell_id"].astype(str)
    _require(frame["batch"].isin(["Batch1", "Batch2", "Batch3"]).all(),
             "batch must use the exact labels Batch1, Batch2, or Batch3.")
    _require(frame["cell_id"].str.strip().ne("").all(), "cell_id cannot be empty.")
    duplicates = frame.duplicated(["batch", "cell_id"], keep=False)
    _require(not duplicates.any(),
             "Expected one row per (batch, cell_id); duplicate keys: "
             f"{frame.loc[duplicates, ['batch', 'cell_id']].head(8).to_dict('records')}")
    _require(frame["model_eligible"].map(lambda v: isinstance(v, (bool, np.bool_))).all(),
             "model_eligible must contain nonmissing bool values, not 0/1 or strings.")
    frame["model_eligible"] = frame["model_eligible"].astype(bool)
    if "label_eligible" in frame:
        _require(frame["label_eligible"].map(lambda v: isinstance(v, (bool, np.bool_))).all(),
                 "label_eligible must contain nonmissing bool values.")
        frame["label_eligible"] = frame["label_eligible"].astype(bool)
    try:
        target = pd.to_numeric(frame["cycle_life"], errors="raise").to_numpy(
            dtype=float, na_value=np.nan)
    except (TypeError, ValueError) as exc:
        raise ValueError("cycle_life must be numeric or missing.") from exc
    target_ok = np.isfinite(target) & (target > 0)
    frame["cycle_life"] = target
    frame["target_valid"] = target_ok
    if "label_eligible" not in frame:
        frame["label_eligible"] = target_ok
    frame["policy_group"] = frame["charging_policy"].astype("string").str.strip().str.replace(
        r"\s+", " ", regex=True)
    frame["development_eligible"] = (
        frame["batch"].eq("Batch1") & frame["model_eligible"] & target_ok)
    frame = frame.sort_values(["batch", "cell_id"], kind="stable").reset_index(drop=True)
    eligible_ids = frame.index[frame["development_eligible"]].to_numpy()
    _require(len(eligible_ids) >= 13,
             "Need at least 13 eligible Batch1 cells for holdout and five-fold CV.")
    dev_ids, holdout_ids = train_test_split(
        eligible_ids, test_size=0.2, shuffle=True, random_state=int(seed))
    # Whole cells are split; repeated policies are allowed on both sides.
    dev_ids, holdout_ids = np.sort(dev_ids), np.sort(holdout_ids)
    frame["role"] = "excluded"
    external_ok = frame["batch"].ne("Batch1") & frame["label_eligible"] & frame["target_valid"]
    frame.loc[external_ok, "role"] = "external_reserved"
    frame.loc[dev_ids, "role"] = "development"
    frame.loc[holdout_ids, "role"] = "holdout"
    frame["exclusion_reason"] = [
        ";".join(reason for condition, reason in (
            (batch == "Batch1" and not eligible, "model_ineligible"),
            (batch != "Batch1" and not label_ok, "label_ineligible"),
            (not valid, "target_not_finite_positive")) if condition)
        for batch, eligible, label_ok, valid in zip(frame["batch"], frame["model_eligible"],
                                                   frame["label_eligible"], frame["target_valid"])
    ]
    development = frame.loc[dev_ids].reset_index(drop=True)
    dev_keys = set(zip(development["batch"], development["cell_id"]))
    reserved = frame.loc[frame["role"].ne("development")]
    reserved_keys = set(zip(reserved["batch"], reserved["cell_id"]))
    _require(dev_keys.isdisjoint(reserved_keys), "Leakage: development overlaps reserved cells.")
    _require(development["batch"].eq("Batch1").all()
             and development["development_eligible"].all(),
             "Leakage: development contains an ineligible or non-Batch1 cell.")
    _require(set(dev_ids).isdisjoint(holdout_ids)
             and set(dev_ids) | set(holdout_ids) == set(eligible_ids),
             "Leakage: eligible Batch1 split is not a disjoint, exhaustive partition.")
    # Reserved feature values are deliberately never converted, imputed, or predicted.
    try:
        x = development.loc[:, F1].apply(pd.to_numeric, errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Development features must be numeric or missing (NaN).") from exc
    _require(not np.isinf(x.to_numpy()).any(),
             "Development X contains +/-infinity; NaN is allowed for fold-local imputation.")
    manifest = frame[["batch", "cell_id", "charging_policy", "policy_group",
                      "model_eligible", "label_eligible", "target_valid",
                      "development_eligible", "role", "exclusion_reason"]].copy()
    return development, x, manifest, reserved_keys


def build_pipeline(estimator: str, alpha: float | None = None,
                   l1_ratio: float | None = None, seed: int = 42) -> Pipeline:
    """Fresh fold-local pipeline; no preprocessing is fitted by this builder."""
    if estimator == "DummyRegressor":
        model = DummyRegressor(strategy="mean")
    elif estimator == "LinearRegression":
        model = LinearRegression()
    elif estimator == "Ridge":
        model = Ridge(alpha=alpha, solver="svd")
    elif estimator == "ElasticNet":
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000,
                           selection="cyclic", random_state=seed)
    else:
        raise ValueError(f"Unsupported estimator: {estimator}")
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model),
    ])


def _metrics(y: np.ndarray, predicted: np.ndarray) -> dict[str, float | None]:
    _require(np.isfinite(predicted).all(), "Model produced nonfinite predictions.")
    _require(np.isfinite(y).all() and (y > 0).all(),
             "Scoring requires finite, strictly positive development targets.")
    # Undefined R2 (constant targets) is represented by null, not an artificial 0/1.
    scores = {
        "mape_pct": float(np.mean(np.abs((y - predicted) / y)) * 100),
        "mae": float(mean_absolute_error(y, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(y, predicted))),
        "r2": float(r2_score(y, predicted)) if len(y) >= 2 and np.ptp(y) > 0 else None,
    }
    _require(all(v is None or np.isfinite(v) for v in scores.values()),
             "Metrics overflowed or became nonfinite; check development target/feature scales.")
    return scores


def _gap(validation: dict, train: dict, metric: str):
    if validation[metric] is None or train[metric] is None:
        return None
    # Positive always means worse validation: R2 uses the opposite subtraction.
    return (train[metric] - validation[metric] if metric == "r2"
            else validation[metric] - train[metric])


def _coefficient_rows(name: str, columns: tuple, fitted: Pipeline) -> list[dict]:
    model = fitted.named_steps["model"]
    if not hasattr(model, "coef_"):
        return []
    imputer, scaler = fitted.named_steps["imputer"], fitted.named_steps["scaler"]
    raw_coef = model.coef_ / scaler.scale_
    raw_intercept = float(model.intercept_ - np.dot(raw_coef, scaler.mean_))
    return [dict(
        candidate=name, feature=column, fit_scope="development_only",
        coefficient_standardized=float(model.coef_[i]),
        coefficient_original_units=float(raw_coef[i]),
        imputation_median=float(imputer.statistics_[i]),
        scaler_mean=float(scaler.mean_[i]), scaler_scale=float(scaler.scale_[i]),
        intercept_standardized=float(model.intercept_),
        intercept_original_units=raw_intercept,
    ) for i, column in enumerate(columns)]


def _write_json(path: Path, value: Any) -> None:
    # pandas converts missing entries to JSON null (strict JSON never contains NaN).
    def default(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(f"Cannot serialize {type(obj).__name__}")
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                               allow_nan=False, default=default) + "\n", encoding="utf-8")


def _records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records", double_precision=15))


def _plot_oof(oof: pd.DataFrame, name: str, path: Path) -> None:
    selected = oof.loc[oof["candidate"].eq(name)]
    fig = Figure(figsize=(10, 4), constrained_layout=True)
    FigureCanvasAgg(fig)
    actual_ax, residual_ax = fig.subplots(1, 2)
    actual_ax.scatter(selected["actual"], selected["estimated"], alpha=0.8, s=25)
    limits = [min(selected["actual"].min(), selected["estimated"].min()),
              max(selected["actual"].max(), selected["estimated"].max())]
    actual_ax.plot(limits, limits, "--", color="gray", linewidth=1)
    actual_ax.set(xlabel="Actual cycle life", ylabel="OOF estimated cycle life",
                  title="Development OOF: actual vs estimated")
    residual_ax.scatter(selected["estimated"], selected["residual"], alpha=0.8, s=25)
    residual_ax.axhline(0, linestyle="--", color="gray", linewidth=1)
    residual_ax.set(xlabel="OOF estimated cycle life", ylabel="Residual (actual - estimated)",
                    title="Development OOF: residuals")
    fig.suptitle(f"Provisional candidate: {name}")
    fig.savefig(path, dpi=150)
    fig.clear()


def run_development(features: pd.DataFrame, output_dir: str | Path,
                    seed: int = 42) -> dict[str, Any]:
    """Compare 15 settings across five model/feature families on development only.

    Required batch labels are Batch1/Batch2/Batch3; keys must be unique. Missing
    predictors are imputed, infinities rejected, and every training fold must
    have an observed value for every F1 feature. Targets need finite positive
    values for inclusion. train_test_split reserves 20% of Batch1 cells
    with seed 42 by default. Shuffled five-fold KFold uses development cells.
    Selection uses the arithmetic mean of five validation-fold MAPE percentages;
    pooled OOF metrics are secondary. External reservation uses label_eligible
    when supplied (otherwise finite-positive target), regardless of model_eligible.

    Returns comparison_df, oof_df (all settings), oof (selected setting only),
    selected_name, selected_pipeline, metadata,
    paths, plus fold_metrics_df, split_manifest_df and coefficients_df. The
    pipeline expects only the selected ordered feature columns. No final-test
    evaluation entrypoint is provided. Input is never modified.
    """
    output = Path(output_dir).expanduser().resolve()
    project_root = Path(__file__).resolve().parents[1]
    for protected in (project_root / "data", project_root / "sources"):
        _require(output != protected and not output.is_relative_to(protected.resolve()),
                 f"output_dir cannot be inside the protected {protected.name} tree.")
    if (output / "results.json").exists():
        raise ValueError("Completed run exists; read saved results or use a new output directory")
    if output.exists() and any(p.name != "cell_features.csv" for p in output.iterdir()):
        raise ValueError("Partial run exists; use a new output directory instead of overwriting it")
    development, x, manifest, reserved_keys = _prepare(features, seed)
    seed = int(seed)
    y = development["cycle_life"].to_numpy(dtype=float)
    keys = list(zip(development["batch"], development["cell_id"]))
    folds = list(KFold(n_splits=5, shuffle=True, random_state=seed).split(x, y))
    fold_ids = np.zeros(len(development), dtype=int)
    for fold_number, (train_ids, validation_ids) in enumerate(folds, 1):
        train_keys = {keys[i] for i in train_ids}
        validation_keys = {keys[i] for i in validation_ids}
        _require(train_keys.isdisjoint(validation_keys)
                 and (train_keys | validation_keys).isdisjoint(reserved_keys),
                 f"Leakage in CV fold {fold_number}: cells overlap or include reserved cells.")
        _require(len(train_keys | validation_keys) == len(keys),
                 f"CV fold {fold_number} does not cover development cells.")
        all_missing = x.iloc[train_ids].isna().all()
        _require(not all_missing.any(),
                 f"CV fold {fold_number} has all-missing training features: "
                 f"{all_missing.index[all_missing].tolist()}")
        _require((fold_ids[validation_ids] == 0).all(), "CV validation cells repeated.")
        fold_ids[validation_ids] = fold_number
    _require((fold_ids > 0).all(), "CV failed to produce one OOF prediction per cell.")
    manifest["cv_fold"] = pd.Series(pd.NA, index=manifest.index, dtype="Int64")
    manifest.loc[manifest["role"].eq("development"), "cv_fold"] = fold_ids

    candidates = build_candidate_specs()
    comparison_rows, fold_rows, oof_rows, coefficient_rows = [], [], [], []
    selected_name, selected_pipeline, selected_score = None, None, float("inf")
    for spec in candidates:
        name, feature_set, estimator = spec["name"], spec["feature_set"], spec["estimator"]
        alpha, l1_ratio = spec["alpha"], spec["l1_ratio"]
        columns = F0 if feature_set == "F0" else F1
        candidate_x = x.loc[:, columns]
        predicted = np.full(len(y), np.nan)
        validation_fold_scores = []
        for fold_number, (train_ids, validation_ids) in enumerate(folds, 1):
            fitted = build_pipeline(estimator, alpha, l1_ratio, seed=seed)
            fitted.fit(candidate_x.iloc[train_ids], y[train_ids])
            # Both prediction calls are explicitly restricted to development indices.
            train_pred = fitted.predict(candidate_x.iloc[train_ids])
            validation_pred = fitted.predict(candidate_x.iloc[validation_ids])
            predicted[validation_ids] = validation_pred
            train_scores = _metrics(y[train_ids], train_pred)
            validation_scores = _metrics(y[validation_ids], validation_pred)
            validation_fold_scores.append(validation_scores)
            row = dict(candidate=name, feature_set=feature_set, estimator=estimator,
                       alpha=alpha, l1_ratio=l1_ratio, max_iter=spec["max_iter"],
                       tol=spec["tol"], fold=fold_number, n_train=len(train_ids),
                       n_validation=len(validation_ids),
                       train_negative_prediction_count=int((train_pred < 0).sum()),
                       validation_negative_prediction_count=int((validation_pred < 0).sum()))
            for metric in METRICS:
                row[f"train_{metric}"] = train_scores[metric]
                row[f"validation_{metric}"] = validation_scores[metric]
                row[f"gap_{metric}"] = _gap(validation_scores, train_scores, metric)
            fold_rows.append(row)
        oof_scores = _metrics(y, predicted)
        cv_mean_scores = {
            metric: (float(np.mean(values)) if values else None)
            for metric in METRICS
            for values in [[s[metric] for s in validation_fold_scores if s[metric] is not None]]
        }
        fitted = build_pipeline(estimator, alpha, l1_ratio, seed=seed).fit(candidate_x, y)
        train_pred = fitted.predict(candidate_x)
        train_scores = _metrics(y, train_pred)
        row = dict(candidate=name, feature_set=feature_set, feature_names=",".join(columns),
                   estimator=estimator, alpha=alpha, l1_ratio=l1_ratio,
                   max_iter=spec["max_iter"], tol=spec["tol"], n_development=len(y),
                   oof_negative_prediction_count=int((predicted < 0).sum()),
                   train_negative_prediction_count=int((train_pred < 0).sum()))
        for metric in METRICS:
            row[f"cv_mean_{metric}"] = cv_mean_scores[metric]
            row[f"oof_{metric}"] = oof_scores[metric]
            row[f"train_{metric}"] = train_scores[metric]
            row[f"gap_{metric}"] = _gap(oof_scores, train_scores, metric)
        comparison_rows.append(row)
        coefficient_rows.extend(_coefficient_rows(name, columns, fitted))
        for i, (batch, cell_id) in enumerate(keys):
            oof_rows.append(dict(candidate=name, batch=batch, cell_id=cell_id,
                                 charging_policy=development.iloc[i]["charging_policy"],
                             policy_group=development.iloc[i]["policy_group"],
                                 role="development", fold=int(fold_ids[i]), actual=float(y[i]),
                                 estimated=float(predicted[i]), residual=float(y[i] - predicted[i]),
                                 absolute_percentage_error_pct=float(abs((y[i] - predicted[i]) / y[i]) * 100),
                                 negative_prediction=bool(predicted[i] < 0)))
        # Exact ties use the declared candidate order; no secondary test-based rule.
        if cv_mean_scores["mape_pct"] < selected_score:
            selected_name, selected_pipeline = name, fitted
            selected_score = cv_mean_scores["mape_pct"]

    comparison = pd.DataFrame(comparison_rows).sort_values("cv_mean_mape_pct", kind="stable").reset_index(drop=True)
    comparison["rank"] = np.arange(1, len(comparison) + 1)
    comparison["provisional_selected"] = comparison["candidate"].eq(selected_name)
    comparison["evaluation_status"] = EVALUATION_STATUS
    oof = pd.DataFrame(oof_rows)
    fold_metrics = pd.DataFrame(fold_rows)
    coefficients = pd.DataFrame(coefficient_rows)
    selected_row = comparison.iloc[0]
    selected_columns = list(F0 if selected_row["feature_set"] == "F0" else F1)
    _require(oof[["batch", "cell_id"]].apply(tuple, axis=1).isin(keys).all()
             and oof["batch"].eq("Batch1").all(), "Leakage in exported OOF predictions.")
    _require(oof.groupby("candidate").size().eq(len(development)).all()
             and not oof.duplicated(["candidate", "batch", "cell_id"]).any(),
             "OOF predictions must contain exactly one row per development cell per candidate.")

    metadata = {
        "evaluation_status": EVALUATION_STATUS,
        "selection_is_provisional": True,
        "holdout_predictions_run": False, "holdout_metrics_run": False,
        "batch2_predictions_run": False, "batch3_predictions_run": False,
        "final_test_requires": "Batch2 was already evaluated in the earlier run; no automatic retest in this revision.",
        "final_evaluation_authorized": False,
        "external_batch_context": "Batch2 was previously evaluated; this revision uses only Batch1 scores.",
        "created_utc": datetime.now(timezone.utc).isoformat(), "seed": seed,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
                     "matplotlib": matplotlib.__version__, "joblib": joblib.__version__},
        "split": {"cell_key": ["batch", "cell_id"], "unit": "independent cell",
                  "method": "train_test_split", "n_splits": 1,
                  "batch": "Batch1", "holdout_fraction": 0.2,
                  "holdout_rounding": "ceil(0.2 * eligible_batch1_cells)",
                  "random_state": seed,
                  "policy_overlap_allowed": True,
                  "policy_normalization": "strip edges and collapse whitespace; no policy-number rounding",
                  "n_development_groups": int(development["policy_group"].nunique()),
                  "input_order": "sort by batch and string cell_id before splitting",
                  "role_counts": {k: int(v) for k, v in manifest["role"].value_counts().items()}},
        "cv": {"method": "KFold", "n_splits": 5, "shuffle": True,
               "random_state": seed,
               "shared_folds": True, "fold_assignment_column": "split_manifest.csv:cv_fold"},
        "primary_metric": "cv_mean_mape_pct", "secondary_metric": "pooled_oof_mape_pct",
        "cv_aggregation": "Unweighted arithmetic mean of five validation-fold scores; R2 averages defined folds only.",
        "metric_units": {
            "mape_pct": "percent", "mae": "cycles", "rmse": "cycles", "r2": "unitless"},
        "gap_definition": "OOF minus full-development train (errors); train minus OOF (R2). "
                          "Per-fold gaps compare validation with that fold's train.",
        "train_metrics_scope": "Resubstitution on full development, optimistic diagnostics only.",
        "r2_policy": "Null if targets are constant; never force an undefined R2 to 0 or 1.",
        "negative_prediction_policy": "Preserve raw predictions; no clipping; counts and flags exported.",
        "target_transform": "none; raw cycle_life",
        "elasticnet_context": "Severson 2019 used a log-life target; this raw-target F1 adaptation is not a replication.",
        "elasticnet_settings": {"alpha": [0.1, 1.0, 10.0, 100.0], "l1_ratio": [0.1, 0.5],
                                "max_iter": 10000, "tol": 1e-4, "tol_policy": "scikit-learn default",
                                "selection": "cyclic", "fit_intercept": True},
        "selection_bias_note": "OOF is reused to select features/alpha, so selected OOF is not a final unbiased test.",
        "feature_sets": {"F0": list(F0), "F1": list(F1)},
        "candidates": candidates,
        "input_provenance": "The caller supplies extraction/source version; this module reads no source files.",
        "upstream_assumption": "model_eligible and initial-cycle feature timing are established before modeling.",
        "development_sha256": hashlib.sha256(pd.util.hash_pandas_object(
            pd.concat([development[["batch", "cell_id", "cycle_life"]], x], axis=1),
            index=False).to_numpy().tobytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(manifest.to_csv(index=False).encode("utf-8")).hexdigest(),
    }
    model_spec = {
        "selected_name": selected_name, "selection_is_provisional": True,
        "evaluation_status": EVALUATION_STATUS, "fit_scope": "Batch1 development only",
        "feature_set": selected_row["feature_set"], "feature_names": selected_columns,
        "estimator": selected_row["estimator"],
        "alpha": None if pd.isna(selected_row["alpha"]) else float(selected_row["alpha"]),
        "l1_ratio": None if pd.isna(selected_row["l1_ratio"]) else float(selected_row["l1_ratio"]),
        "max_iter": None if pd.isna(selected_row["max_iter"]) else int(selected_row["max_iter"]),
        "tol": None if pd.isna(selected_row["tol"]) else float(selected_row["tol"]),
        "target_transform": "none; raw cycle_life",
        "pipeline_steps": ["SimpleImputer(strategy=median)", "StandardScaler", selected_row["estimator"]],
        "ridge_solver": "svd" if selected_row["estimator"] == "Ridge" else None,
        "dummy_strategy": "mean" if selected_row["estimator"] == "DummyRegressor" else None,
        "prediction_input": "DataFrame containing exactly feature_names in that order",
        "prediction_clipping": "none", "seed": seed, "versions": metadata["versions"],
        "selection_metric": "cv_mean_mape_pct", "tie_break": "declared candidate order",
        "n_fit_cells": len(development), "development_sha256": metadata["development_sha256"],
    }
    output = Path(output_dir).expanduser().resolve()
    # Guard the project's read-only source and raw-data trees even if the caller misroutes outputs.
    project_root = Path(__file__).resolve().parents[1]
    for protected in (project_root / "data", project_root / "sources"):
        _require(output != protected and not output.is_relative_to(protected.resolve()),
                 f"output_dir cannot be inside the protected {protected.name} tree.")
    output.mkdir(parents=True, exist_ok=True)
    paths = {name: str(output / filename) for name, filename in {
        "split_manifest": "split_manifest.csv", "comparison": "model_comparison.csv",
        "fold_metrics": "fold_metrics.csv", "oof_predictions": "oof_predictions.csv",
        "coefficients": "feature_coefficients.csv", "pipeline": "selected_pipeline.joblib",
        "model_spec": "model_spec.json", "metadata": "metadata.json",
        "results": "results.json", "oof_plot": "selected_oof_diagnostics.png",
    }.items()}
    for name, frame in (("split_manifest", manifest), ("comparison", comparison),
                        ("fold_metrics", fold_metrics), ("oof_predictions", oof),
                        ("coefficients", coefficients)):
        frame.to_csv(paths[name], index=False)
    joblib.dump(selected_pipeline, paths["pipeline"])
    _write_json(Path(paths["model_spec"]), model_spec)
    _write_json(Path(paths["metadata"]), metadata)
    _write_json(Path(paths["results"]), {
        "evaluation_status": EVALUATION_STATUS, "selected_name": selected_name,
        "metadata": metadata, "model_spec": model_spec,
        "paths_relative_to": "results.json directory",
        "paths": {name: Path(path).name for name, path in paths.items()},
        "comparison": _records(comparison), "fold_metrics": _records(fold_metrics),
        "oof_predictions": _records(oof), "split_manifest": _records(manifest),
        "coefficients": _records(coefficients),
    })
    _plot_oof(oof, selected_name, Path(paths["oof_plot"]))
    return {
        "comparison_df": comparison, "comparison": comparison,
        "oof_df": oof, "oof": oof.loc[oof["candidate"].eq(selected_name)].reset_index(drop=True),
        "selected_name": selected_name,
        "selected_pipeline": selected_pipeline, "metadata": metadata, "paths": paths,
        "fold_metrics_df": fold_metrics, "split_manifest_df": manifest,
        "coefficients_df": coefficients,
    }
