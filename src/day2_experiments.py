"""Fixed-split DAY2 feature experiments. Select on Batch1 CV, then score winner.

Run with: python -m src.day2_experiments
Existing models, splits, labels and raw data remain unchanged.
"""

from pathlib import Path
from functools import partial
import hashlib
import json
import platform
import warnings

import h5py
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import TransformedTargetRegressor
from sklearn.exceptions import ConvergenceWarning

from .day2_features import BATCH_FILES, _vector
from .day2_modeling import build_pipeline, _metrics
from .day2_evaluation import _score

SEED = 42
BASE_FEATURE = ["log10_delta_Q_var"]
STAGES = [
    ("S0", "기존: 원 수명·분산", "raw", BASE_FEATURE),
    ("S1", "로그 수명", "log10", BASE_FEATURE),
    ("S2", "+ ΔQ 최솟값", "log10", BASE_FEATURE + ["delta_Q_min"]),
    ("S3", "+ 초기 용량 특징", "log10", BASE_FEATURE + ["delta_Q_min", "qd_initial_median", "qd_fraction_change", "early_drop_per100"]),
    ("S4", "+ IR 변화율", "log10", BASE_FEATURE + ["delta_Q_min", "qd_initial_median", "qd_fraction_change", "early_drop_per100", "ir_fraction_change"]),
    ("S5", "+ 온도 변화", "log10", BASE_FEATURE + ["delta_Q_min", "qd_initial_median", "qd_fraction_change", "early_drop_per100", "ir_fraction_change", "tavg_change", "temperature_spread"]),
    ("S6", "+ 충전시간 중앙값", "log10", BASE_FEATURE + ["delta_Q_min", "qd_initial_median", "qd_fraction_change", "early_drop_per100", "ir_fraction_change", "tavg_change", "temperature_spread", "median_chargetime"]),
    ("S7", "+ C1·전환 SOC·C2", "log10", BASE_FEATURE + ["delta_Q_min", "qd_initial_median", "qd_fraction_change", "early_drop_per100", "ir_fraction_change", "tavg_change", "temperature_spread", "median_chargetime", "C1", "switch_SOC", "C2"]),
]


def extract_changes(data_dir):
    """Per-cell deterministic features, using only summary cycles 1 through 100."""
    rows = []
    for batch, filename in BATCH_FILES.items():
        with h5py.File(Path(data_dir) / filename, "r") as file:
            bg = file["batch"]
            for cell_id in range(bg["summary"].size):
                summary = file[bg["summary"][cell_id, 0]]
                d = pd.DataFrame({key: _vector(summary[key]) for key in
                                  ["cycle", "QDischarge", "IR", "Tavg", "Tmax"]})
                d = d.loc[d.cycle.between(1, 100)].copy()
                d = d.replace([np.inf, -np.inf], np.nan)
                d.loc[d.IR.le(0), "IR"] = np.nan
                d.loc[d.QDischarge.le(0), "QDischarge"] = np.nan
                # Remove the empty initial summary row, consistent with extraction.
                d.loc[d[["Tavg", "Tmax"]].eq(0).all(axis=1), ["Tavg", "Tmax"]] = np.nan
                first = d.loc[d.cycle.between(10, 20)]
                last = d.loc[d.cycle.between(90, 100)]

                def median(window, column):
                    a = window[column].dropna()
                    return float(a.median()) if len(a) >= 3 else np.nan

                q0, q1 = median(first, "QDischarge"), median(last, "QDischarge")
                ir0, ir1 = median(first, "IR"), median(last, "IR")
                t0, t1 = median(first, "Tavg"), median(last, "Tavg")
                rows.append({
                    "batch": batch, "cell_id": str(cell_id),
                    "qd_initial_median": q0,
                    "qd_fraction_change": (q1-q0)/q0 if q0 > 0 else np.nan,
                    "ir_change": ir1-ir0,
                    "ir_fraction_change": (ir1-ir0)/ir0 if ir0 > 0 else np.nan,
                    "tavg_change": t1-t0,
                    "temperature_spread": (d.Tmax-d.Tavg).median(),
                    "first_window_ir_count": int(first.IR.count()),
                    "last_window_ir_count": int(last.IR.count()),
                    "change_feature_max_cycle": float(d.cycle.max()),
                })
    return pd.DataFrame(rows)


def settings():
    yield {"name": "linear", "estimator": "LinearRegression", "alpha": None, "l1_ratio": None}
    for alpha in (0.1, 1.0, 10.0):
        yield {"name": f"ridge_{alpha:g}", "estimator": "Ridge", "alpha": alpha, "l1_ratio": None}
    for alpha in (0.001, 0.01, 0.1):
        for ratio in (0.1, 0.5):
            yield {"name": f"elasticnet_{alpha:g}_{ratio:g}", "estimator": "ElasticNet", "alpha": alpha, "l1_ratio": ratio}


def make_model(setting, target):
    pipe = build_pipeline(setting["estimator"], setting["alpha"], setting["l1_ratio"], seed=SEED)
    if target == "raw":
        return pipe
    return TransformedTargetRegressor(regressor=pipe, func=np.log10,
                                      inverse_func=partial(np.power, 10.0))


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def feature_diagnostics(model, columns, dev, test):
    pipe = model.regressor_ if hasattr(model, "regressor_") else model
    coefficients = pipe.named_steps["model"].coef_ / pipe.named_steps["scaler"].scale_
    rows = []
    for col, coef in zip(columns, coefficients):
        low, high = dev[col].min(), dev[col].max()
        rows.append({"feature": col, "train_median": dev[col].median(),
                     "test_median": test[col].median(), "train_min": low, "train_max": high,
                     "test_outside_train_n": int((test[col].lt(low) | test[col].gt(high)).sum()),
                     "coefficient_original_units": float(coef),
                     "median_difference_contribution": float((test[col].median()-dev[col].median())*coef)})
    return pd.DataFrame(rows)


def main():
    root = Path(__file__).resolve().parents[1]
    previous = root / "outputs/day2/modeling_cell_split"
    output = root / "outputs/day2/feature_experiments"
    if (output / "results.json").exists():
        print("Saved experiment already exists; inspect results.json instead of repeating evaluation.")
        return
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("Partial experiment exists; inspect saved files before restarting")
    output.mkdir(parents=True, exist_ok=True)
    original_hash = hashlib.sha256((previous / "final_evaluation.json").read_bytes()).hexdigest()
    features = pd.read_csv(previous / "cell_features.csv", dtype={"cell_id": str})
    added = extract_changes(root / "data")
    features = features.merge(added, on=["batch", "cell_id"], validate="one_to_one")
    assert features.change_feature_max_cycle.le(100).all()
    features.to_csv(output / "cell_features.csv", index=False)
    manifest = pd.read_csv(previous / "split_manifest.csv", dtype={"cell_id": str})
    joined = manifest[["batch", "cell_id", "role", "cv_fold"]].merge(
        features, on=["batch", "cell_id"], validate="one_to_one", sort=False)
    dev = joined.loc[joined.role.eq("development")].reset_index(drop=True)
    assert len(dev) == 28 and dev.batch.eq("Batch1").all()
    assert not joined.duplicated(["batch", "cell_id"]).any()
    fold_id = dev.cv_fold.astype(int).to_numpy()
    assert set(fold_id) == set(range(1, 6))
    y = dev.cycle_life.to_numpy(float)
    all_settings = list(settings())
    rows, fold_rows, oof_rows, specs = [], [], [], {}
    for stage, title, target, columns in STAGES:
        for setting in (all_settings[:1] if stage == "S0" else all_settings):
            name = f"{stage}_{setting['name']}"
            specs[name] = {"stage": stage, "title": title, "target": target,
                           "features": columns, **setting}
            x = dev[columns]
            assert not np.isinf(x.to_numpy(float)).any()
            pred = np.full(len(y), np.nan)
            fold_mapes = []
            warning_count = 0
            for fold in range(1, 6):
                tr, va = np.flatnonzero(fold_id != fold), np.flatnonzero(fold_id == fold)
                assert not x.iloc[tr].isna().all().any()
                model = make_model(setting, target)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    model.fit(x.iloc[tr], y[tr])
                warning_count += sum(issubclass(w.category, ConvergenceWarning) for w in caught)
                pred[va] = model.predict(x.iloc[va])
                score = _metrics(y[va], pred[va])
                train_score = _metrics(y[tr], model.predict(x.iloc[tr]))
                fold_mapes.append(score["mape_pct"])
                fold_rows.append({"candidate": name, "fold": fold, "n_train": len(tr),
                                  "n_valid": len(va), "train_mape_pct": train_score["mape_pct"], **score})
            pooled = _metrics(y, pred)
            rows.append({"candidate": name, "stage": stage, "title": title,
                         "target": target, "model": setting["name"], "n_features": len(columns),
                         "cv_mape_pct": float(np.mean(fold_mapes)),
                         "cv_mape_std_pp": float(np.std(fold_mapes, ddof=1)),
                         "oof_mape_pct": pooled["mape_pct"], "oof_mae": pooled["mae"],
                         "oof_rmse": pooled["rmse"], "oof_r2": pooled["r2"],
                         "convergence_warning_count": warning_count,
                         "feature_names": ",".join(columns)})
            for i in range(len(dev)):
                oof_rows.append({"candidate": name, "batch": dev.iloc[i].batch,
                                 "cell_id": dev.iloc[i].cell_id, "fold": int(fold_id[i]),
                                 "actual": y[i], "prediction": pred[i],
                                 "ape_pct": 100*abs(y[i]-pred[i])/y[i]})
    comparison = pd.DataFrame(rows).sort_values("cv_mape_pct", kind="stable").reset_index(drop=True)
    comparison.insert(0, "rank", np.arange(1, len(comparison)+1))
    assert len(comparison) == 71
    reference = pd.read_csv(previous / "model_comparison.csv").query("candidate == 'linear_F0'").iloc[0]
    baseline = comparison.loc[comparison.candidate.eq("S0_linear"), "cv_mape_pct"].iloc[0]
    assert np.isclose(baseline, reference.cv_mean_mape_pct, atol=1e-9)
    # Select and freeze on CV alone, before retrieving holdout/test targets.
    selected = comparison.iloc[0]
    spec = specs[selected.candidate]
    write_json(output / "model_spec.json", {"selected_candidate": selected.candidate,
        "selected_using": "Batch1 mean five-fold validation MAPE only", "seed": SEED,
        "n_fit_cells": 28, "selection_frozen_before_external_evaluation": True, **spec})
    model = make_model(spec, spec["target"])
    model.fit(dev[spec["features"]], y)
    joblib.dump(model, output / "selected_pipeline.joblib")
    stage_best = comparison.sort_values("rank").groupby("stage", sort=True).first().reset_index()
    linear = comparison.loc[comparison.model.eq("linear"), ["stage", "cv_mape_pct"]].rename(
        columns={"cv_mape_pct": "linear_cv_mape_pct"})
    stage_best = stage_best.merge(linear, on="stage", validate="one_to_one")
    stage_best["change_vs_previous_pp"] = stage_best.cv_mape_pct.diff()
    stage_best["linear_change_vs_previous_pp"] = stage_best.linear_cv_mape_pct.diff()
    comparison.to_csv(output / "model_comparison.csv", index=False)
    stage_best.to_csv(output / "stage_comparison.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output / "fold_metrics.csv", index=False)
    pd.DataFrame(oof_rows).to_csv(output / "oof_predictions.csv", index=False)
    manifest.to_csv(output / "split_manifest.csv", index=False)
    valid = joined.loc[joined.batch.eq("Batch1") & joined.role.eq("holdout")].copy()
    test = joined.loc[joined.batch.eq("Batch2") & joined.label_eligible
                      & joined.delta_status.eq("ok") & joined.cycle_life.gt(100)].copy()
    assert len(valid) == 8 and len(test) == 39
    predictions, scores = [], {}
    for name, subset in [("Valid (Batch 1 Hold-out)", valid), ("Test (Batch 2)", test)]:
        record, score = _score(subset, model, spec["features"])
        record["evaluation"] = name
        predictions.append(record)
        scores[name] = score
    predictions = pd.concat(predictions, ignore_index=True)
    predictions.to_csv(output / "evaluation_predictions.csv", index=False)
    feature_diagnostics(model, spec["features"], dev, test).to_csv(
        output / "selected_feature_diagnostics.csv", index=False)
    vm, tm = scores["Valid (Batch 1 Hold-out)"]["mape_pct"], scores["Test (Batch 2)"]["mape_pct"]
    report = pd.DataFrame([
        ["Train (Batch 1 CV)", selected.cv_mape_pct, "동일 28개 셀·동일 5-fold 검증 MAPE 평균"],
        ["Valid (Batch 1 Hold-out)", vm, "8개 셀; 후보 선택에 사용하지 않음"],
        ["Test (Batch 2)", tm, "39개 셀; CV 선택 후 고정 모델 재평가"],
        ["Gap (Train-Valid)", vm-selected.cv_mape_pct, "Valid − CV; %p"],
        ["Gap (Valid-Test)", tm-vm, "Test − Valid; %p"],
        ["Gap (Target-Test)", tm-9.1, "Test − 9.1; %p; 참고 비교"],
    ], columns=["구분", "MAPE (%)", "비고"])
    report.to_csv(output / "performance_reporting.csv", index=False)
    bands = pd.cut(predictions.cycle_life, [-np.inf, 500, 1000, np.inf], labels=["<=500", "500-1000", ">1000"])
    predictions.assign(life_band=bands).groupby(["evaluation", "life_band"], observed=True).agg(
        n=("cell_id", "size"), mape_pct=("ape_pct", "mean"),
        mean_residual=("residual", "mean")).to_csv(output / "evaluation_segments.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    axes[0].plot(stage_best.stage, stage_best.cv_mape_pct, "o-", label="Best candidate")
    axes[0].plot(stage_best.stage, stage_best.linear_cv_mape_pct, "o--", label="Linear only")
    axes[0].set(xlabel="Feature stage", ylabel="CV MAPE (%)", title="Same development cells and folds")
    axes[0].legend()
    axes[1].scatter(test.cycle_life, predictions.loc[predictions.evaluation.eq("Test (Batch 2)"), "prediction"], s=25)
    axes[1].plot([300, 1300], [300, 1300], "--", color="gray")
    axes[1].set(xlabel="Actual cycle life", ylabel="Predicted cycle life", title="Frozen winner: Batch2 reevaluation")
    fig.savefig(output / "experiment_summary.png", dpi=150)
    plt.close(fig)
    versions = {"python": platform.python_version(), "numpy": np.__version__,
                "pandas": pd.__version__, "sklearn": sklearn.__version__, "h5py": h5py.__version__,
                "matplotlib": matplotlib.__version__, "joblib": joblib.__version__}
    result = {"selected": selected.candidate, "spec": spec, "cv_mape_pct": selected.cv_mape_pct,
              "scores": scores, "n_candidates": len(comparison), "seed": SEED,
              "evaluation_context": "Previously evaluated Batch2; user-authorized frozen-winner reevaluation, not a new untouched test",
              "test_used_for_selection": False, "holdout_used_for_selection": False,
              "refit_including_holdout": False, "batch3_evaluated": False,
              "continuation_recovery": "Not possible with available files: 2017-06-30 continuation file absent",
              "versions": versions,
              "feature_definitions": {
                  "qd_initial_median": "median QD cycles 10–20; at least 3 positive observations",
                  "qd_fraction_change": "(median QD cycles 90–100 − median QD cycles 10–20) / initial median",
                  "ir_fraction_change": "same window ratio for IR; nonpositive IR missing; at least 3 observations per window",
                  "tavg_change": "median Tavg cycles 90–100 − median Tavg cycles 10–20",
                  "temperature_spread": "median(Tmax−Tavg), cycles 1–100",
                  "early_drop_per100": "existing 10–100 slope after 7-point rolling median, multiplied by −100",
              }, "stages": [{"id": s, "title": t, "target": y, "features": c} for s,t,y,c in STAGES],
              "settings": all_settings,
              "selection_bias": "71 candidate settings use the same CV; selected CV is optimistic. Cumulative stage changes do not isolate every feature's marginal effect.",
              "data_files": [{"batch": b, "name": n, "size_bytes": (root/"data"/n).stat().st_size,
                              "mtime_ns": (root/"data"/n).stat().st_mtime_ns} for b,n in BATCH_FILES.items()],
              "sha256": {"features": hashlib.sha256((output/"cell_features.csv").read_bytes()).hexdigest(),
                          "split_manifest": hashlib.sha256((previous/"split_manifest.csv").read_bytes()).hexdigest(),
                          "experiment_code": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                          "model": hashlib.sha256((output/"selected_pipeline.joblib").read_bytes()).hexdigest(),
                          "parent_features": hashlib.sha256((previous/"cell_features.csv").read_bytes()).hexdigest(),
                          "modeling_code": hashlib.sha256((root/"src/day2_modeling.py").read_bytes()).hexdigest(),
                          "label_audit": hashlib.sha256((root/"outputs/day2/label_audit.csv").read_bytes()).hexdigest()}}
    write_json(output / "results.json", result)
    assert hashlib.sha256((previous/"final_evaluation.json").read_bytes()).hexdigest() == original_hash
    print(stage_best[["stage", "title", "model", "cv_mape_pct", "linear_cv_mape_pct", "change_vs_previous_pp"]].round(4).to_string(index=False))
    print("Selected:", selected.candidate)
    print(report.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
