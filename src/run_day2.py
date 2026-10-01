"""Run revised DAY2 cell-level CV and holdout; do not repeat external tests."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

from .day2_features import BATCH_FILES, extract_cell_features, attach_label_audit
from .day2_modeling import run_development
from .day2_evaluation import evaluate_holdout_model


def save_provenance(root, output, audit_path, seed):
    feature_path = output / "cell_features.csv"
    sources = []
    for batch, filename in BATCH_FILES.items():
        info = (root / "data" / filename).stat()
        sources.append({"batch": batch, "file": filename,
                        "size_bytes": info.st_size, "mtime_ns": info.st_mtime_ns})
    provenance = {
        "raw_files": sources,
        "feature_csv_sha256": hashlib.sha256(feature_path.read_bytes()).hexdigest(),
        "label_audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        "python": sys.version.split()[0], "seed": seed,
        "code_sha256": {name: hashlib.sha256((root / "src" / name).read_bytes()).hexdigest()
                        for name in ("day2_features.py", "day2_modeling.py", "day2_evaluation.py", "run_day2.py")},
        "input_cutoff_cycle": 100,
        "target": "as-provided cycle_life, eligibility checked separately; no target imputation",
        "heldout_evaluation": "Batch1 holdout only; previously evaluated Batch2 is not repeated",
        "split_unit": "one independent cell per row",
    }
    (output / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / "outputs" / "day2" / "modeling_cell_split"
    audit_path = root / "outputs" / "day2" / "label_audit.csv"
    # A shared checkout contains the published run. Viewing it must not replace
    # its model before the frozen-evaluation signature is checked.
    if (output / "results.json").exists():
        required = ("metadata.json", "model_spec.json", "selected_pipeline.joblib",
                    "model_comparison.csv", "split_manifest.csv", "performance_reporting.csv")
        missing = [name for name in required if not (output / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Incomplete saved run: {missing}")
        metadata = json.loads((output / "metadata.json").read_text())
        if metadata["seed"] != args.seed:
            raise ValueError("Saved run uses a different seed; use a separate project copy for a new run")
        print("Saved run exists; no feature extraction, training or evaluation repeated.")
        comparison = pd.read_csv(output / "model_comparison.csv")
        print(comparison[["candidate", "feature_names", "cv_mean_mape_pct"]].round(4).to_string(index=False))
        print(pd.read_csv(output / "performance_reporting.csv").round(4).to_string(index=False))
        return
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("Partial run exists; inspect it before starting a new run")
    if not audit_path.exists():
        raise FileNotFoundError("Complete the target label audit before training: outputs/day2/label_audit.csv")
    output.mkdir(parents=True, exist_ok=True)
    raw_features = extract_cell_features(root / "data")
    features = attach_label_audit(raw_features, pd.read_csv(audit_path))
    feature_path = output / "cell_features.csv"
    features.to_csv(feature_path, index=False)
    results = run_development(features, output, seed=args.seed)
    evaluation = evaluate_holdout_model(features, output)
    save_provenance(root, output, audit_path, args.seed)
    print(results["comparison_df"].round(4).to_string(index=False))
    print("CV recommendation:", results["selected_name"])
    print(evaluation["report"].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
