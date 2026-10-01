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
               "capacity_penalty", "curve_models", "protocol_validation")


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
        if "split" in saved:
            saved["evaluation"] = saved["split"].map(
                {"Valid": "Valid (Batch 1 Hold-out)", "Test Batch2": "Test (Batch 2)"})
        joined = recalculated.merge(saved, on=keys, suffixes=("_check", "_saved"),
                                     validate="one_to_one")
        require(len(joined) == len(saved) == len(recalculated), f"{name}: metric rows differ")
        for metric in ("n", "mape_pct", "mae", "rmse", "r2"):
            np.testing.assert_allclose(joined[metric + "_check"], joined[metric + "_saved"],
                                       rtol=1e-9, atol=1e-9)
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
                 BASE / "DAY2_REPORT.md", BASE / "DAY2_PROTOCOL_VALIDATION.md"]:
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text()):
            if target.startswith(("http://", "https://", "#")):
                continue
            require((path.parent / target.split("#")[0]).exists(),
                    f"Broken link in {path.name}: {target}")


def main():
    total = 0
    for name in EXPERIMENTS:
        total += verify_experiment(name)
        print(f"PASS: {name} — saved prediction metrics and CV means")
    verify_protocol_split()
    verify_report_links()
    print(f"PASS: {total} prediction rows; protocol split, frozen models, gaps and links")
    print("No training, raw-data access, model unpickling or file writes.")


if __name__ == "__main__":
    main()
