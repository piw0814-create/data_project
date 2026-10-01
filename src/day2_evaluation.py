"""Score frozen CV-selected models without using evaluation scores for selection."""

import hashlib
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PAPER_TARGET_MAPE = 9.1  # Assignment reference, not an exactly reproduced experiment.


def _score(frame, pipeline, columns):
    x = frame[columns].apply(pd.to_numeric, errors="raise")
    if np.isinf(x.to_numpy()).any() or x.isna().all(axis=1).any():
        raise ValueError("Evaluation inputs contain infinity or entirely missing features")
    y = frame["cycle_life"].to_numpy(dtype=float)
    if not np.isfinite(y).all() or (y <= 100).any():
        raise ValueError("Evaluation needs finite targets beyond the prediction horizon")
    prediction = pipeline.predict(x)
    if not np.isfinite(prediction).all():
        raise ValueError("Nonfinite evaluation predictions")
    records = frame[["batch", "cell_id", "cycle_life", "charging_policy"]].copy()
    records["prediction"] = prediction
    records["residual"] = y-prediction
    records["ape_pct"] = 100*np.abs(y-prediction)/y
    records["negative_prediction"] = prediction < 0
    scores = {
        "n": len(y), "mape_pct": float(records.ape_pct.mean()),
        "mae": float(mean_absolute_error(y, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(y, prediction))),
        "r2": float(r2_score(y, prediction)),
    }
    return records, scores


def evaluate_frozen_model(features, modeling_dir, *, allow_retest=False):
    """Never select/refit a model from validation or test scores.

    The SAME pipeline, fitted only on Batch1 development, predicts holdout and
    Batch2. Batch3 remains an optional reserved dataset. Same-run reruns reuse
    saved evaluation; changed inputs/model need a distinct run directory.
    """
    output = Path(modeling_dir)
    metadata = json.loads((output / "metadata.json").read_text())
    if not metadata.get("final_evaluation_authorized", False) and not allow_retest:
        raise ValueError("This revised run permits holdout evaluation only; Batch2 was already evaluated.")
    spec_path = output / "model_spec.json"
    spec = json.loads(spec_path.read_text())
    if spec.get("fit_scope") != "Batch1 development only":
        raise ValueError("Expected a CV-selected model fitted only on development")
    manifest = pd.read_csv(output / "split_manifest.csv", dtype={"cell_id": str})
    frame = features.copy()
    frame["cell_id"] = frame["cell_id"].astype(str)
    frame = frame.merge(manifest[["batch", "cell_id", "role"]],
                        on=["batch", "cell_id"], validate="one_to_one")
    columns = spec["feature_names"]
    valid = frame.loc[frame["batch"].eq("Batch1") & frame["role"].eq("holdout")].copy()
    label_ok = frame["label_eligible"]
    if label_ok.dtype != bool:
        label_ok = label_ok.astype(str).str.lower().map({"true": True, "false": False})
        if label_ok.isna().any():
            raise ValueError("label_eligible needs explicit booleans")
    test = frame.loc[frame["batch"].eq("Batch2") & label_ok
                     & frame["delta_status"].eq("ok")].copy()
    if not len(valid) or not len(test):
        raise ValueError("Both Batch1 holdout and labeled Batch2 test cells are required")
    is_cell_split = metadata["split"]["method"] == "train_test_split"
    if not is_cell_split and set(valid["charging_policy"]) & set(
            frame.loc[frame.role.eq("development"), "charging_policy"]):
        raise ValueError("A charging policy crosses development/holdout")
    comparison = pd.read_csv(output / "model_comparison.csv")
    selected = comparison.loc[comparison["candidate"].eq(spec["selected_name"])].iloc[0]
    if comparison.iloc[0]["candidate"] != spec["selected_name"]:
        raise ValueError("Frozen model does not match the CV recommendation")
    signature = hashlib.sha256(
        spec_path.read_bytes()
        + (output / "selected_pipeline.joblib").read_bytes()
        + manifest.to_csv(index=False).encode()
        + pd.concat([valid, test])[["batch", "cell_id", "cycle_life"]+columns].to_csv(
            index=False, float_format="%.12g").encode()
    ).hexdigest()
    eval_path = output / "final_evaluation.json"
    if eval_path.exists():
        cached = json.loads(eval_path.read_text())
        if cached["frozen_run_sha256"] != signature:
            raise ValueError("This run already used its test set with another model/input; use a new run explicitly")
        return {"report": pd.read_csv(output / "performance_reporting.csv"), "scores": cached,
                "predictions": pd.read_csv(output / "evaluation_predictions.csv"), "cached": True}
    pipeline = joblib.load(output / "selected_pipeline.joblib")
    if is_cell_split:
        heldout = evaluate_holdout_model(features, output)
        valid_pred, valid_scores = heldout["predictions"].copy(), heldout["scores"]["valid"]
    else:
        valid_pred, valid_scores = _score(valid, pipeline, columns)
    test_pred, test_scores = _score(test, pipeline, columns)
    valid_pred["evaluation"] = "Valid (Batch 1 Hold-out)"
    test_pred["evaluation"] = "Test (Batch 2)"
    predictions = pd.concat([valid_pred, test_pred], ignore_index=True)
    train = float(selected["cv_mean_mape_pct"])
    valid_mape, test_mape = valid_scores["mape_pct"], test_scores["mape_pct"]
    report = pd.DataFrame([
        ["Train (Batch 1 CV)", train, "셀 단위 5-fold MAPE 산술 평균" if is_cell_split else "프로토콜별 5-fold MAPE의 산술 평균"],
        ["Valid (Batch 1 Hold-out)", valid_mape, f"{len(valid)}개 셀; 같은 최종 모델"],
        ["Test (Batch 2)", test_mape, f"{len(test)}개 셀; 고정 모델 재평가" if is_cell_split else f"{len(test)}개 셀; CV 선택 후 1회 평가"],
        ["Gap (Train-Valid)", valid_mape-train, "Valid - Train; (+): 과적합 또는 조건 차이 의심"],
        ["Gap (Valid-Test)", test_mape-valid_mape, "Test - Valid; (+): 배치 간 일반화 저하 의심"],
        ["Gap (Target-Test)", test_mape-PAPER_TARGET_MAPE, "Test - 9.1%; 과제에서 제시한 원논문 참고값"],
    ], columns=["구분", "MAPE (%)", "비고"])
    scores = {
        "evaluation_status": "completed", "selection_is_provisional": False,
        "selected_name": spec["selected_name"], "frozen_run_sha256": signature,
        "train_cv_mean_mape_pct": train,
        "train_pooled_oof_mape_pct": float(selected["oof_mape_pct"]),
        "valid": valid_scores, "test_batch2": test_scores,
        "target_mape_pct": PAPER_TARGET_MAPE, "batch3_evaluation": "not run (optional)",
        "final_fit_scope": "Batch1 development only; same pipeline for Valid/Test",
        "refit_after_holdout": False, "selected_using": "Batch1 mean CV MAPE only",
        "gap_units": "percentage points; later-stage error minus earlier-stage error",
        "benchmark_limit": "local Batch2 file, cohort, feature set and target transform differ from original paper",
        "label_limit": "Batch1 uses supplied stopping-boundary labels, not directly observed below-0.88 crossings",
        "batch2_context": "Previously evaluated with the protocol-split model; this is a user-requested reevaluation of the frozen cell-split model." if is_cell_split else "Already inspected in DAY1 EDA; no DAY2 model selection uses its score",
        "evaluation_kind": "user_requested_retest" if is_cell_split else "initial_evaluation",
        "test_used_for_selection": False,
    }
    predictions.to_csv(output / "evaluation_predictions.csv", index=False)
    report.to_csv(output / "performance_reporting.csv", index=False)
    eval_path.write_text(json.dumps(scores, ensure_ascii=False, indent=2, allow_nan=False))
    pd.DataFrame([{"split": "Valid", **valid_scores}, {"split": "Test Batch2", **test_scores}]).to_csv(
        output / "evaluation_metrics.csv", index=False)
    bins = pd.cut(predictions.cycle_life, [-np.inf, 500, 1000, np.inf],
                  labels=["<=500", "500-1000", ">1000"])
    segment = predictions.assign(life_band=bins).groupby(["evaluation", "life_band"], observed=True).agg(
        n=("cell_id", "size"), mape_pct=("ape_pct", "mean"), mean_residual=("residual", "mean"))
    segment.to_csv(output / "evaluation_segments.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, (label, group) in zip(axes, predictions.groupby("evaluation", sort=False)):
        ax.scatter(group.cycle_life, group.prediction, s=28, alpha=.8)
        low = min(group.cycle_life.min(), group.prediction.min())
        high = max(group.cycle_life.max(), group.prediction.max())
        ax.plot([low, high], [low, high], "--", color="gray", lw=1)
        ax.set(title=label, xlabel="Actual cycle life", ylabel="Estimated cycle life")
    fig.savefig(output / "final_evaluation.png", dpi=150)
    plt.close(fig)
    return {"report": report, "scores": scores, "predictions": predictions, "cached": False}


def evaluate_holdout_model(features, modeling_dir):
    """Evaluate only the frozen winner on Batch1 holdout; never predict Batch2/3.

    The previous run already evaluated Batch2. Its score cannot be assigned to
    this revised model or used to select new candidates. Reuse identical saved
    holdout predictions on reruns and reject changed evaluation inputs.
    """
    output = Path(modeling_dir)
    spec_path = output / "model_spec.json"
    spec = json.loads(spec_path.read_text())
    if spec.get("fit_scope") != "Batch1 development only":
        raise ValueError("Expected a CV-selected model fitted only on development")
    manifest = pd.read_csv(output / "split_manifest.csv", dtype={"cell_id": str})
    valid_keys = manifest.loc[manifest.batch.eq("Batch1") & manifest.role.eq("holdout"),
                              ["batch", "cell_id"]]
    dev_keys = manifest.loc[manifest.role.eq("development"), ["batch", "cell_id"]]
    if set(map(tuple, valid_keys.to_numpy())) & set(map(tuple, dev_keys.to_numpy())):
        raise ValueError("A cell appears in both development and holdout")
    frame = features.loc[features.batch.eq("Batch1")].copy()
    frame["cell_id"] = frame["cell_id"].astype(str)
    valid = valid_keys.merge(frame, on=["batch", "cell_id"], validate="one_to_one")
    if len(valid) != len(valid_keys) or not len(valid):
        raise ValueError("All holdout cells must be present")
    columns = spec["feature_names"]
    comparison = pd.read_csv(output / "model_comparison.csv")
    if comparison.iloc[0]["candidate"] != spec["selected_name"]:
        raise ValueError("Frozen model does not match the CV recommendation")
    selected = comparison.iloc[0]
    signature = hashlib.sha256(
        spec_path.read_bytes() + (output / "selected_pipeline.joblib").read_bytes()
        + manifest.to_csv(index=False).encode()
        + valid[["batch", "cell_id", "cycle_life"] + columns].to_csv(
            index=False, float_format="%.12g").encode()
    ).hexdigest()
    eval_path = output / "holdout_evaluation.json"
    if eval_path.exists():
        cached = json.loads(eval_path.read_text())
        if cached["frozen_run_sha256"] != signature:
            raise ValueError("Holdout already evaluated with different model/input; use a distinct run")
        report = pd.read_csv(output / "performance_reporting.csv")
        mask = report["구분"].isin(["Test (Batch 2)", "Gap (Valid-Test)", "Gap (Target-Test)"])
        report.loc[mask, "MAPE (%)"] = np.nan
        report.loc[mask, "비고"] = "이 단계에서는 Hold-out만 확인; Batch2는 최종 평가에서 확인"
        return {"report": report, "scores": cached,
                "predictions": pd.read_csv(output / "holdout_predictions.csv"), "cached": True}
    pipeline = joblib.load(output / "selected_pipeline.joblib")
    predictions, scores = _score(valid, pipeline, columns)
    cv_mape = float(selected["cv_mean_mape_pct"])
    report = pd.DataFrame([
        ["Train (Batch 1 CV)", cv_mape, "셀 단위 5-fold MAPE 산술 평균"],
        ["Valid (Batch 1 Hold-out)", scores["mape_pct"], f"{len(valid)}개 셀; CV로 선택한 동일 모델"],
        ["Test (Batch 2)", np.nan, "수정 모델 미평가; 기존 Test 평가 결과와 구분"],
        ["Gap (Train-Valid)", scores["mape_pct"] - cv_mape, "Valid - CV; 단위 %p"],
        ["Gap (Valid-Test)", np.nan, "수정 모델의 Test 결과 필요"],
        ["Gap (Target-Test)", np.nan, "수정 모델의 Test 결과 필요; 참고 Target 9.1%"],
    ], columns=["구분", "MAPE (%)", "비고"])
    evaluation = {
        "evaluation_status": "holdout_completed_external_test_not_repeated",
        "selected_name": spec["selected_name"], "frozen_run_sha256": signature,
        "train_cv_mean_mape_pct": cv_mape,
        "train_pooled_oof_mape_pct": float(selected["oof_mape_pct"]),
        "valid": scores, "test_batch2": None, "test_batch3": None,
        "batch2_predictions_run": False, "batch3_predictions_run": False,
        "selected_using": "Batch1 mean CV MAPE only",
        "refit_after_holdout": False, "target_mape_pct": PAPER_TARGET_MAPE,
        "previous_test_results": "../modeling/final_evaluation.json (different split/model)",
        "revision_reason": "Use cell splitting for the assignment goal; no requirement to exclude shared policies",
        "honesty_note": "The same Batch1 cells were used in an earlier development run; holdout is a development check, not a new independent test.",
    }
    predictions.to_csv(output / "holdout_predictions.csv", index=False)
    report.to_csv(output / "performance_reporting.csv", index=False)
    pd.DataFrame([{"split": "Valid", **scores}]).to_csv(output / "holdout_metrics.csv", index=False)
    eval_path.write_text(json.dumps(evaluation, ensure_ascii=False, indent=2, allow_nan=False))
    fig, ax = plt.subplots(figsize=(5, 4), layout="constrained")
    ax.scatter(predictions.cycle_life, predictions.prediction, s=28, alpha=.8)
    low = min(predictions.cycle_life.min(), predictions.prediction.min())
    high = max(predictions.cycle_life.max(), predictions.prediction.max())
    ax.plot([low, high], [low, high], "--", color="gray", lw=1)
    ax.set(title="Batch1 cell holdout", xlabel="Actual cycle life", ylabel="Estimated cycle life")
    fig.savefig(output / "holdout_evaluation.png", dpi=150)
    plt.close(fig)
    return {"report": report, "scores": evaluation, "predictions": predictions, "cached": False}
