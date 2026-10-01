"""User-requested Batch3 test of the original 28-cell, 26.19% Batch2 model.

No refit or tuning. The earlier two-model Batch3 test remains a separate record.
Run: python -m src.day2_batch3_original
"""
from datetime import datetime, timezone
from pathlib import Path
import json
import platform

import joblib
import numpy as np
import pandas as pd

from .day2_batch3 import BASE, ROOT, sha
from .day2_evaluation import _score, evaluate_frozen_model
from .day2_experiments import write_json

PARENT = BASE / "modeling_cell_split"
OUT = BASE / "batch3_original_test"


def validate(plan):
    for name, expected in plan["input_hashes"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Frozen input changed: {name}")


def main():
    if (OUT / "results.json").is_file():
        result = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
        validate(result["plan"])
        print("Saved original-model Batch3 test exists; no repeated fitting/scoring.")
        print(pd.read_csv(OUT / "evaluation_metrics.csv").round(3).to_string(index=False))
        return
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError("Partial evaluation exists; inspect it before restarting")
    if not (PARENT / "final_evaluation.json").is_file():
        raise ValueError("Original Batch2 evaluation must already exist")
    spec = json.loads((PARENT / "model_spec.json").read_text(encoding="utf-8"))
    if spec["selected_name"] != "linear_F0" or spec["n_fit_cells"] != 28:
        raise ValueError("Not the original 28-cell variance model")
    frame = pd.read_csv(PARENT / "cell_features.csv", dtype={"cell_id": str})
    # The cached evaluator verifies the original model/input signature. It must
    # return saved Batch2 predictions, not score Batch2 again.
    previous = evaluate_frozen_model(frame, PARENT, allow_retest=True)
    if not previous["cached"]:
        raise ValueError("Expected the original saved evaluation")
    np.testing.assert_allclose(previous["scores"]["test_batch2"]["mape_pct"],
                               26.18567870657303, atol=1e-10)
    paths = [PARENT / name for name in ["selected_pipeline.joblib", "model_spec.json",
             "cell_features.csv", "split_manifest.csv", "final_evaluation.json"]]
    paths += [BASE / "batch3_test" / name for name in ["input_audit.csv", "cell_features.csv"]]
    plan = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                candidate="linear_F0", fit_cells=28, features=spec["feature_names"],
                input_hashes={p.relative_to(ROOT).as_posix(): sha(p) for p in paths},
                reason="User requested the historical 26.19% model after seeing the first Batch3 results",
                batch3_context="Follow-up evaluation of an existing model; not a new untouched test",
                model_fit=False, tuning=False, selected_from_batch3=False,
                input_cutoff_cycle=100, seed=42, code_sha256=sha(Path(__file__)),
                python=platform.python_version(), model_versions=spec["versions"])
    validate(plan)
    # Use exactly the 44 cells whose raw early inputs were already audited.
    audited = pd.read_csv(BASE / "batch3_test/cell_features.csv", dtype={"cell_id": str})
    test = frame.loc[frame.batch.eq("Batch3")].merge(
        audited[["batch", "cell_id"]], on=["batch", "cell_id"], validate="one_to_one")
    test = test.sort_values("cell_id").reset_index(drop=True)
    audited = audited.sort_values("cell_id").reset_index(drop=True)
    if len(test) != 44 or not test.cell_id.equals(audited.cell_id):
        raise ValueError("Different Batch3 test cells")
    np.testing.assert_allclose(test[["cycle_life"] + spec["feature_names"]],
                               audited[["cycle_life"] + spec["feature_names"]], atol=1e-12)
    OUT.mkdir(parents=True)
    write_json(OUT / "plan.json", plan)  # Frozen before this model's Batch3 predictions.
    model = joblib.load(PARENT / "selected_pipeline.joblib")
    predictions, metrics = _score(test, model, spec["feature_names"])
    predictions["candidate"], predictions["evaluation"] = "linear_F0", "Test (Batch 3)"
    metrics.update(bias_cycles=float(-predictions.residual.mean()),
                   underprediction_n=int(predictions.residual.gt(0).sum()))
    validate(plan)
    predictions.to_csv(OUT / "evaluation_predictions.csv", index=False)
    rows = [dict(candidate="linear_F0", evaluation=evaluation, **score)
            for evaluation, score in [("Valid (Batch 1 Hold-out)", previous["scores"]["valid"]),
                                      ("Test (Batch 2)", previous["scores"]["test_batch2"]),
                                      ("Test (Batch 3)", metrics)]]
    pd.DataFrame(rows).to_csv(OUT / "evaluation_metrics.csv", index=False)
    result = dict(plan=plan, completed_utc=datetime.now(timezone.utc).isoformat(),
                  new_model_fits=0, new_predictions=44, batch3_metrics=metrics,
                  batch2_mape_pct=previous["scores"]["test_batch2"]["mape_pct"],
                  model_sha256_after=sha(PARENT / "selected_pipeline.joblib"))
    write_json(OUT / "results.json", result)
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
