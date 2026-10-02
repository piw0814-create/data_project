"""Verify saved results without fitting models, reading MATs or writing files.

Run from the project root: python -m src.verify_results
"""
from pathlib import Path
import hashlib
import json
import re

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/day2"
EXPERIMENTS = ("modeling_cell_split", "feature_experiments", "ablation_loss",
               "capacity_penalty", "curve_models", "protocol_validation", "batch3_test",
               "batch3_original_test")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_experiment(name):
    folder = BASE / name
    predictions = pd.read_csv(folder / "evaluation_predictions.csv")
    keys = [col for col in ("candidate", "condition") if col in predictions] + ["evaluation"]
    require(not predictions.duplicated(keys + ["batch", "cell_id"]).any(),
            f"{name}: duplicate prediction cells")
    require(np.isfinite(predictions[["cycle_life", "prediction"]]).all().all(),
            f"{name}: nonfinite values")
    require(predictions.cycle_life.gt(100).all(), f"{name}: invalid target horizon")
    np.testing.assert_allclose(predictions.residual,
                               predictions.cycle_life - predictions.prediction, atol=1e-9)
    ape = 100 * abs(predictions.prediction - predictions.cycle_life) / predictions.cycle_life
    np.testing.assert_allclose(predictions.ape_pct, ape, atol=1e-9)
    rows = []
    for key, group in predictions.groupby(keys):
        key = key if isinstance(key, tuple) else (key,)
        y, pred = group.cycle_life, group.prediction
        row = dict(zip(keys, key))
        row.update(n=len(group), mape_pct=float(group.ape_pct.mean()),
                   mae=mean_absolute_error(y, pred), rmse=np.sqrt(mean_squared_error(y, pred)),
                   r2=r2_score(y, pred))
        rows.append(row)
    recalculated = pd.DataFrame(rows)
    metrics_path = folder / "evaluation_metrics.csv"
    if metrics_path.exists():
        saved = pd.read_csv(metrics_path)
        if name in ("batch3_test", "batch3_original_test"):
            saved = saved.loc[saved.evaluation.eq("Test (Batch 3)")]
        if "split" in saved:
            saved["evaluation"] = saved["split"].map(
                {"Valid": "Valid (Batch 1 Hold-out)", "Test Batch2": "Test (Batch 2)"})
        joined = recalculated.merge(saved, on=keys, suffixes=("_check", "_saved"),
                                     validate="one_to_one")
        require(len(joined) == len(saved) == len(recalculated), f"{name}: metric rows differ")
        for metric in ("n", "mape_pct", "mae", "rmse", "r2"):
            np.testing.assert_allclose(joined[metric + "_check"], joined[metric + "_saved"],
                                       rtol=1e-9, atol=1e-9)
    if name == "batch3_original_test":
        result = json.loads((folder / "results.json").read_text(encoding="utf-8"))
        require(result["new_model_fits"] == 0 and result["new_predictions"] == 44,
                "Original-model follow-up changed training")
        for path, expected in result["plan"]["input_hashes"].items():
            require(hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected,
                    f"Original-model frozen input changed: {path}")
        require(len(predictions) == 44 and predictions.batch.eq("Batch3").all(),
                "Unexpected original-model test cells")
        previous = pd.read_csv(BASE / "batch3_test/evaluation_predictions.csv")
        require(set(previous.cell_id) == set(predictions.cell_id), "Batch3 test cells differ")
        np.testing.assert_allclose(result["batch2_mape_pct"], 26.18567870657303, atol=1e-10)
        require(result["model_sha256_after"] == result["plan"]["input_hashes"][
                "outputs/day2/modeling_cell_split/selected_pipeline.joblib"],
                "Original model changed during evaluation")
        return len(predictions)
    if name == "batch3_test":
        result = json.loads((folder / "results.json").read_text(encoding="utf-8"))
        plan = result["plan"]
        require(result["new_model_fits"] == 0 and not plan["model_selection_changed"],
                "Batch3 test changed training/selection")
        for path, expected in plan["input_hashes"].items():
            require(hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected,
                    f"Batch3 frozen input changed: {path}")
        for model, expected in plan["model_hashes"].items():
            require(hashlib.sha256((BASE / "protocol_validation" / f"{model}.joblib").read_bytes()).hexdigest() == expected,
                    "Batch3 frozen model changed")
        require(predictions.batch.eq("Batch3").all() and len(predictions) == 88,
                "Unexpected Batch3 predictions")
        audit = pd.read_csv(folder / "input_audit.csv")
        require(len(audit) == 46 and audit.included.sum() == 44
                and audit.raw_features_match_cache.all()
                and audit.loc[audit.included, ["finite_model_inputs", "input_horizon_ok"]].all().all(),
                "Batch3 audit failed")
        values = pd.read_csv(folder / "performance_reporting.csv")["MAPE (%)"]
        train, valid, b2, b3 = values.iloc[0], values.iloc[1], values.iloc[2], values.iloc[6]
        np.testing.assert_allclose(values.iloc[[3, 4, 5, 7, 8]],
                                   [valid-train, b2-valid, b2-9.1, b3-b2, b3-9.1], atol=1e-9)
        return len(predictions)
    folds = pd.read_csv(folder / "fold_metrics.csv")
    filename = "cv_comparison.csv" if name in ("capacity_penalty", "curve_models") else "model_comparison.csv"
    comparison = pd.read_csv(folder / filename)
    keys = (["condition"] if name == "capacity_penalty" else ["candidate"])
    if "method" in folds:
        keys.append("method")
    fold_column = "validation_mape_pct" if name == "modeling_cell_split" else "mape_pct"
    mean_column = "cv_mean_mape_pct" if name == "modeling_cell_split" else "cv_mape_pct"
    means = folds.groupby(keys)[fold_column].mean().rename("recalculated").reset_index()
    compared = comparison.merge(means, on=keys, validate="one_to_one")
    require(len(compared) == len(comparison), f"{name}: CV candidates differ")
    np.testing.assert_allclose(compared[mean_column], compared.recalculated, atol=1e-9)
    return len(predictions)


def verify_protocol_split():
    folder = BASE / "protocol_validation"
    manifest = pd.read_csv(folder / "split_manifest.csv", dtype={"cell_id": str})
    require(not manifest.duplicated(["batch", "cell_id"]).any(), "Duplicate split cells")
    dev, hold = [manifest.loc[manifest.role.eq(role)] for role in ("development", "holdout")]
    require(len(dev) == 29 and len(hold) == 7, "Unexpected split sizes")
    require(set(dev.protocol).isdisjoint(hold.protocol), "Hold-out protocol overlap")
    require(set(dev.protocol_cv_fold) == set(range(1, 6)), "Missing protocol CV fold")
    for fold in range(1, 6):
        train = dev.loc[dev.protocol_cv_fold.ne(fold)]
        valid = dev.loc[dev.protocol_cv_fold.eq(fold)]
        require(set(train.protocol).isdisjoint(valid.protocol), f"Fold {fold}: protocol overlap")
    source = BASE / "feature_experiments/cell_features.csv"
    features = pd.read_csv(source, dtype={"cell_id": str})
    checked = manifest.merge(features, on=["batch", "cell_id"], validate="one_to_one")
    require(len(checked) == 36 and checked.input_max_cycle.le(100).all()
            and checked.change_feature_max_cycle.le(100).all(), "Invalid early-feature horizon")
    plan = json.loads((folder / "plan.json").read_text())
    require(hashlib.sha256(source.read_bytes()).hexdigest() == plan["source_sha256"],
            "Frozen input features changed")
    selection = json.loads((folder / "selection.json").read_text())
    for model, expected in selection["model_hashes"].items():
        require(hashlib.sha256((folder / f"{model}.joblib").read_bytes()).hexdigest() == expected,
                f"Frozen model changed: {model}")
    comparison = pd.read_csv(folder / "model_comparison.csv")
    winner = comparison.loc[comparison.method.eq("protocol")].sort_values(
        ["cv_mape_pct", "n_features"], kind="stable").iloc[0].candidate
    require(winner == selection["primary"], "Selected model is not the protocol-CV winner")
    for experiment in ("ablation_loss", "protocol_validation"):
        values = pd.read_csv(BASE / experiment / "performance_reporting.csv")["MAPE (%)"]
        train, valid, test = values.iloc[:3]
        np.testing.assert_allclose(values.iloc[3:], [valid - train, test - valid, test - 9.1], atol=1e-9)


def verify_report_links():
    for path in [ROOT / "README.md", ROOT / "outputs/final/DAY1_REPORT.md",
                 BASE / "DAY2_REPORT.md", BASE / "DAY2_PROTOCOL_VALIDATION.md",
                 BASE / "DAY2_BATCH3_TEST.md", BASE / "DAY2_BUSINESS_APPENDIX.md",
                 ROOT / "docs/PROJECT_GUIDE.md", ROOT / "data/README.md"]:
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text()):
            if target.startswith(("http://", "https://", "#")):
                continue
            require((path.parent / target.split("#")[0]).exists(),
                    f"Broken link in {path.name}: {target}")


def verify_business_appendix():
    """Check retained evidence from discarded experiments without rerunning them."""
    folder = BASE / "business_appendix"
    provenance = json.loads((folder / "provenance.json").read_text())
    for name, expected in provenance["evidence_sha256"].items():
        require(hashlib.sha256((folder / name).read_bytes()).hexdigest() == expected,
                f"Appendix evidence changed: {name}")
    for name, expected in provenance["frozen_main_model_sha256"].items():
        require(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected,
                f"Main model changed after appendix: {name}")
    predictions = pd.read_csv(folder / "predictions.csv", dtype={"cell_id": str})
    keys = ["experiment", "split", "candidate", "evaluation"]
    require(not predictions.duplicated(keys + ["cell_id"]).any(),
            "Appendix has duplicate prediction cells")
    require(np.isfinite(predictions[["cycle_life", "prediction"]]).all().all()
            and predictions.cycle_life.gt(100).all(), "Invalid appendix targets/predictions")
    np.testing.assert_allclose(predictions.ape_pct,
                               100 * abs(predictions.prediction - predictions.cycle_life)
                               / predictions.cycle_life, atol=1e-9)
    rows = []
    for key, group in predictions.groupby(keys):
        masks = {"all": np.ones(len(group), dtype=bool),
                 "short_lt500": group.cycle_life.lt(500).to_numpy(),
                 "other_ge500": group.cycle_life.ge(500).to_numpy()}
        for segment, mask in masks.items():
            data = group.loc[mask]
            if data.empty:
                continue
            delta = data.prediction - data.cycle_life
            rows.append(dict(zip(keys, key), segment=segment, n=len(data),
                             mape_pct=float(data.ape_pct.mean()), mae=float(abs(delta).mean()),
                             rmse=float(np.sqrt(np.mean(delta**2))), bias_cycles=float(delta.mean()),
                             mean_positive_overprediction=float(delta.clip(lower=0).mean()),
                             overprediction_pct=float(100 * (delta > 0).mean())))
    checked = pd.DataFrame(rows)
    saved = pd.read_csv(folder / "metrics.csv")
    joined = checked.merge(saved, on=keys + ["segment"], suffixes=("_check", "_saved"),
                           validate="one_to_one")
    require(len(joined) == len(checked) == len(saved), "Appendix metric rows differ")
    for metric in ("n", "mape_pct", "mae", "rmse", "bias_cycles",
                   "mean_positive_overprediction", "overprediction_pct"):
        np.testing.assert_allclose(joined[metric + "_check"], joined[metric + "_saved"],
                                   rtol=1e-9, atol=1e-9)
    folds = pd.read_csv(folder / "fold_metrics.csv")
    require(not folds.duplicated(["experiment", "split", "candidate", "regime", "fold"]).any()
            and np.isfinite(folds.mape_pct).all(), "Invalid appendix fold evidence")
    objectives = folds.loc[folds.experiment.eq("shortlife_objective")]
    means = objectives.groupby(["split", "regime", "candidate"]).mape_pct.mean()
    order = {name: i for i, name in enumerate(("baseline", "inverse", "q50", "q30"))}
    for split, regimes in provenance["objective_selection"].items():
        for regime, expected in regimes.items():
            scores = means.loc[(split, regime)]
            winner = min(scores.index, key=lambda name: (scores[name], order[name]))
            require(winner == expected, "Appendix source-only selection mismatch")
    for split in provenance["stress_splits"]:
        require(split["protocol_overlap"] == 0 and split["train_min"] > split["valid_max"]
                and set(split["train_ids"]).isdisjoint(split["valid_ids"]),
                "Invalid appendix stress split")
    return len(predictions)


def verify_final_recommendation():
    """Audit the post-evaluation recommendation without changing historical selection."""
    folder = BASE / "final_recommendation"
    record = json.loads((folder / "recommendation.json").read_text())
    require(record["candidate"] == "S0_linear" and record["selection_uses_external_results"]
            and not record["independent_final_test"] and not record["refit"],
            "Final recommendation context differs")
    require(record["features"] == ["log10_delta_Q_var"]
            and record["target"] == "raw_cycle_life"
            and record["n_development"] == 29 and record["n_holdout"] == 7,
            "Final recommendation specification differs")
    model = ROOT / record["model_path"]
    require(hashlib.sha256(model.read_bytes()).hexdigest() == record["model_sha256"],
            "Recommended model changed")
    for name, expected in record["source_sha256"].items():
        require(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected,
                f"Recommendation evidence changed: {name}")
    historical = json.loads((BASE / "protocol_validation/selection.json").read_text())
    require(historical["primary"] == record["historical_cv_winner"]
            and historical["model_hashes"][record["candidate"]] == record["model_sha256"],
            "Historical selection or frozen model differs")
    row = pd.read_csv(BASE / "protocol_validation/summary.csv").set_index("candidate").loc[record["candidate"]]
    cv, valid, test = row.cv_mape_pct, row["Valid (Batch 1 Hold-out)"], row["Test (Batch 2)"]
    metrics = pd.read_csv(BASE / "batch3_test/evaluation_metrics.csv")
    batch3 = metrics.loc[metrics.candidate.eq(record["candidate"])
                        & metrics.evaluation.eq("Test (Batch 3)"), "mape_pct"].item()
    expected = [cv, valid, test, valid-cv, test-valid, test-9.1,
                batch3, batch3-test, batch3-9.1]
    saved = pd.read_csv(folder / "performance_reporting.csv")
    np.testing.assert_allclose(saved["MAPE (%)"], expected, atol=1e-9)


def main():
    total = 0
    for name in EXPERIMENTS:
        total += verify_experiment(name)
        checks = "saved prediction metrics and frozen test inputs" if name.startswith("batch3_") else "saved prediction metrics and CV means"
        print(f"PASS: {name} — {checks}")
    verify_protocol_split()
    appendix_rows = verify_business_appendix()
    verify_report_links()
    verify_final_recommendation()
    print("PASS: final recommendation — existing S0 model, historical CV winner and all reporting gaps")
    print(f"PASS: business appendix — {appendix_rows} saved predictions, source selections and model hashes")
    print(f"PASS: {total} prediction rows; protocol split, frozen models, gaps and links")
    print("No training, raw-data access, model unpickling or file writes.")


if __name__ == "__main__":
    main()
