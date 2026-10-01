"""Build one row per cell using only measurements available by cycle 100.

Raw MAT files are opened read-only. Late-life audit fields are never inputs.
"""

from pathlib import Path
import re

import h5py
import numpy as np
import pandas as pd

BATCH_FILES = {
    "Batch1": "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
    "Batch2": "2018-02-20_batchdata_updated_struct_errorcorrect.mat",
    "Batch3": "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
}
F0 = ["log10_delta_Q_var"]
F1 = ["log10_delta_Q_var", "median_chargetime", "mean_Tavg"]
POLICY_PATTERN = re.compile(
    r"(?P<C1>\d+(?:\.\d+)?)C\((?P<switch_SOC>\d+(?:\.\d+)?)%\)"
    r"\s*[-–]\s*(?P<C2>\d+(?:\.\d+)?)C"
)


def _vector(dataset):
    if dataset.attrs.get("MATLAB_empty", 0):
        return np.array([], dtype=float)
    return np.asarray(dataset[()], dtype=float).reshape(-1)


def _curve(file, cycle_group, position):
    refs = cycle_group["Qdlin"]
    # MATLAB structs in these files use an N x 1 reference array.
    if refs.ndim != 2 or refs.shape[1] != 1:
        raise ValueError(f"Unexpected Qdlin reference shape: {refs.shape}")
    return _vector(file[refs[position, 0]])


def extract_cell_features(data_dir):
    """Read summaries and exactly the 10th/100th Qdlin curves, not whole MATs.

    Missing labels remain missing here so exclusion is explicit in the audit.
    Features are not imputed or scaled until the model's training fold.
    """
    data_dir = Path(data_dir)
    rows = []
    reference_voltage = None
    for batch, filename in BATCH_FILES.items():
        with h5py.File(data_dir / filename, "r") as file:
            bg = file["batch"]
            for cell_id in range(bg["summary"].size):
                summary = file[bg["summary"][cell_id, 0]]
                cycle = _vector(summary["cycle"])
                arrays = {
                    key: _vector(summary[source]) for key, source in {
                        "QD": "QDischarge", "IR": "IR", "Tavg": "Tavg",
                        "Tmax": "Tmax", "chargetime": "chargetime",
                    }.items()
                }
                if any(len(a) != len(cycle) for a in arrays.values()):
                    raise ValueError(f"{batch} cell {cell_id}: summary length mismatch")
                if not np.isfinite(cycle).all() or len(np.unique(cycle)) != len(cycle):
                    raise ValueError(f"{batch} cell {cell_id}: invalid cycle numbers")
                early = pd.DataFrame({"cycle": cycle, **arrays})
                early = early.loc[early["cycle"].between(1, 100)].copy()
                measurements = ["QD", "IR", "Tavg", "Tmax", "chargetime"]
                empty = early[measurements].eq(0).all(axis=1)
                empty_count = int(empty.sum())
                early = early.loc[~empty].sort_values("cycle")
                early["IR"] = early["IR"].replace(0, np.nan)
                early[measurements] = early[measurements].replace([np.inf, -np.inf], np.nan)
                life = _vector(file[bg["cycle_life"][cell_id, 0]])
                if life.size > 1:
                    raise ValueError(f"{batch} cell {cell_id}: non-scalar cycle_life")
                policy = "".join(chr(int(v)) for v in file[bg["policy_readable"][cell_id, 0]][()].ravel())
                row = {
                    "batch": batch, "cell_id": cell_id,
                    "cycle_life": float(life[0]) if life.size else np.nan,
                    "charging_policy": policy, "source_file": filename,
                    "early_records": len(early), "empty_early_records": empty_count,
                    "input_max_cycle": float(early["cycle"].max()) if len(early) else np.nan,
                    "mean_QD": early["QD"].mean(), "std_QD": early["QD"].std(),
                    "mean_IR": early["IR"].mean(), "valid_IR_records": early["IR"].count(),
                    "mean_Tavg": early["Tavg"].mean(), "mean_Tmax": early["Tmax"].mean(),
                    "mean_chargetime": early["chargetime"].mean(),
                    "median_chargetime": early["chargetime"].median(),
                    "delta_Q_mean": np.nan, "delta_Q_min": np.nan,
                    "delta_Q_var": np.nan, "log10_delta_Q_var": np.nan,
                    "delta_status": "pending",
                }
                match = POLICY_PATTERN.search(policy)
                row.update({key: float(match[key]) if match else np.nan for key in ["C1", "switch_SOC", "C2"]})
                row["has_newstructure"] = "newstructure" in policy.lower()
                # Ancillary candidate only; do not use late-cycle slopes.
                slope_data = early.loc[early["cycle"].between(10, 100) & early["QD"].gt(0)].dropna(subset=["QD"])
                if len(slope_data) >= 3:
                    smooth = slope_data["QD"].rolling(7, center=True, min_periods=1).median()
                    row["early_drop_per100"] = -100 * np.polyfit(slope_data["cycle"], smooth, 1)[0]
                else:
                    row["early_drop_per100"] = np.nan
                try:
                    indices = [np.flatnonzero(cycle == n) for n in [10, 100]]
                    if any(len(i) != 1 for i in indices):
                        raise ValueError("cycle_10_or_100_missing")
                    cg = file[bg["cycles"][cell_id, 0]]
                    if cg["Qdlin"].size != len(cycle):
                        raise ValueError("summary_cycles_length_mismatch")
                    q10, q100 = [_curve(file, cg, int(i[0])) for i in indices]
                    voltage = _vector(file[bg["Vdlin"][cell_id, 0]])
                    if q10.size == 0 or not (q10.shape == q100.shape == voltage.shape):
                        raise ValueError("empty_or_mismatched_curve")
                    if not all(np.isfinite(a).all() for a in [q10, q100, voltage]):
                        raise ValueError("nonfinite_curve")
                    if reference_voltage is None:
                        reference_voltage = voltage.copy()
                    if reference_voltage.shape != voltage.shape or not np.allclose(voltage, reference_voltage):
                        raise ValueError("voltage_grid_mismatch")
                    delta = q100 - q10
                    variance = float(np.var(delta, ddof=0))
                    if variance <= 0:
                        raise ValueError("nonpositive_delta_variance")
                    row.update(delta_Q_mean=float(np.mean(delta)), delta_Q_min=float(np.min(delta)),
                               delta_Q_var=variance, log10_delta_Q_var=float(np.log10(variance)),
                               delta_status="ok")
                except (ValueError, IndexError, KeyError) as error:
                    row["delta_status"] = str(error)
                rows.append(row)
    features = pd.DataFrame(rows)
    if features.duplicated(["batch", "cell_id"]).any():
        raise ValueError("Duplicate cell identifiers")
    if features["input_max_cycle"].dropna().gt(100).any():
        raise ValueError("Future measurements entered features")
    return features


def attach_label_audit(features, audit):
    """Join a separately documented target eligibility audit; never alter Y."""
    required = {"batch", "cell_id", "cycle_life", "model_eligible", "label_status"}
    if not required.issubset(audit.columns):
        raise ValueError(f"Audit lacks {sorted(required - set(audit.columns))}")
    if audit.duplicated(["batch", "cell_id"]).any():
        raise ValueError("Duplicate audit identifiers")
    # Endpoint/crossing/continuation numbers stay in label_audit.csv, not X.
    audit_columns = ["batch", "cell_id", "cycle_life", "model_eligible", "label_status"]
    audit_columns += [name for name in ["label_eligible", "audit_reason"] if name in audit.columns]
    output = features.merge(audit[audit_columns], on=["batch", "cell_id"], how="left",
                            validate="one_to_one", suffixes=("", "_audit"))
    y, audited = output["cycle_life"], output["cycle_life_audit"]
    if not np.allclose(y, audited, equal_nan=True):
        raise ValueError("Label audit altered targets or does not cover every cell")
    if output["model_eligible"].isna().any() or output["label_status"].isna().any():
        raise ValueError("Every cell needs an explicit label audit status")
    # CSV bool parsing must not convert the string 'False' to True.
    raw = output["model_eligible"]
    if raw.dtype != bool:
        parsed = raw.astype(str).str.lower().map({"true": True, "false": False})
        if parsed.isna().any():
            raise ValueError("model_eligible must contain true/false")
        output["model_eligible"] = parsed.astype(bool)
    output["model_eligible"] = (
        output["model_eligible"] & output["cycle_life"].gt(100)
        & np.isfinite(output["cycle_life"]) & output["delta_status"].eq("ok")
    )
    return output.drop(columns="cycle_life_audit")
