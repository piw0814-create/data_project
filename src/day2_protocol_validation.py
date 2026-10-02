"""Re-evaluate early features using protocol-disjoint Batch1 validation.

Run: python -m src.day2_protocol_validation
Reuse per-cell feature calculations; fit every preprocessing step within a fold.
Freeze the complete selection before evaluating holdout or previously seen Batch2.
"""
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
import hashlib
import json
import platform
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import QuantileRegressor
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .day2_experiments import STAGES, settings, make_model, write_json
from .day2_ablation import md_table
from .day2_evaluation import _score
from .day2_modeling import _metrics

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/day2/protocol_validation"
FEATURE_SOURCE = ROOT / "outputs/day2/feature_experiments/cell_features.csv"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protocol(frame):
    columns = ["C1", "switch_SOC", "C2"]
    if not np.isfinite(frame[columns].to_numpy()).all():
        raise ValueError("Every evaluated cell needs a parsed charging protocol")
    return frame[columns].apply(lambda row: "|".join(format(x, ".12g") for x in row), axis=1)


def stage_specs():
    rows = []
    stages = STAGES + [("F1", "DAY1: +충전시간·온도", "raw",
                       ["log10_delta_Q_var", "median_chargetime", "mean_Tavg"])]
    for stage, title, target, features in stages:
        for setting in list(settings())[:1] if stage == "S0" else list(settings()):
            rows.append(dict(candidate=f"{stage}_{setting['name']}", phase="features",
                             stage=stage, title=title, family="standard", target=target,
                             features=list(features), removed=None, capacity_penalty=1,
                             **setting))
    return rows


def refinement_specs(best):
    """Adaptive rule fixed before CV: full set, one-at-a-time removals, loss/penalty."""
    rows = []
    removals = [None] + [c for c in best["features"] if c != "log10_delta_Q_var"]
    for removed in removals:
        features = [c for c in best["features"] if c != removed]
        tag = "full" if removed is None else f"minus_{removed}"
        for alpha in [0.1, 1.0, 10.0]:
            rows.append(dict(candidate=f"R_ridge_{tag}_a{alpha:g}", phase="refinement",
                             stage="R", family="standard", estimator="Ridge", alpha=alpha,
                             l1_ratio=None, target="log10", features=features,
                             removed=removed, capacity_penalty=1))
        for family in ["mae_raw", "mape_raw"]:
            for alpha in [0.0, 0.01, 0.1]:
                rows.append(dict(candidate=f"R_{family}_{tag}_a{alpha:g}", phase="refinement",
                                 stage="R", family=family, estimator="QuantileRegressor",
                                 alpha=alpha, l1_ratio=None, target="raw", features=features,
                                 removed=removed, capacity_penalty=1))
    if "qd_initial_median" in best["features"]:
        for penalty in [10, 100]:
            rows.append(dict(candidate=f"R_initial_penalty_{penalty}", phase="refinement",
                             stage="R", family="standard", estimator="Ridge", alpha=1.0,
                             l1_ratio=None, target="log10", features=best["features"],
                             removed=None, capacity_penalty=penalty))
    return rows


def build(spec):
    if spec["family"] in ["mae_raw", "mape_raw"]:
        return Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("scaler", StandardScaler()),
                         ("model", QuantileRegressor(quantile=0.5,
                                  alpha=spec["alpha"], solver="highs"))])
    model = make_model(spec, spec["target"])
    if spec["capacity_penalty"] != 1:
        factors = np.ones(len(spec["features"]))
        factors[spec["features"].index("qd_initial_median")] = 1 / np.sqrt(spec["capacity_penalty"])
        model.regressor.steps.insert(2, ("capacity_penalty", FunctionTransformer(
            partial(np.multiply, factors))))
    return model


def fit(model, spec, x, y):
    kwargs = {}
    if spec["family"] == "mape_raw":
        weight = 1 / y
        kwargs["model__sample_weight"] = weight / weight.mean()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x, y, **kwargs)
    return sum(issubclass(w.category, ConvergenceWarning) for w in caught)


def cv_compare(specs, dev, folds_by_method, comparisons, fold_rows, oofs):
    y = dev.cycle_life.to_numpy(float)
    for spec in specs:
        x = dev[spec["features"]]
        for method, folds in folds_by_method.items():
            prediction = np.full(len(y), np.nan)
            scores, warning_count = [], 0
            for fold, (tr, va) in enumerate(folds, 1):
                if x.iloc[tr].isna().all().any() or np.isinf(x.to_numpy()).any():
                    raise ValueError("A candidate has an unusable training feature")
                model = build(spec)
                warning_count += fit(model, spec, x.iloc[tr], y[tr])
                prediction[va] = np.asarray(model.predict(x.iloc[va])).reshape(-1)
                score = _metrics(y[va], prediction[va])
                overlap = len(set(dev.protocol.iloc[tr]) & set(dev.protocol.iloc[va]))
                if method == "protocol" and overlap:
                    raise AssertionError("Protocol leakage in CV")
                fold_rows.append(dict(candidate=spec["candidate"], method=method, fold=fold,
                    n_train=len(tr), n_valid=len(va), protocol_overlap=overlap,
                    n_valid_protocols=dev.protocol.iloc[va].nunique(), **score))
                scores.append(score["mape_pct"])
            pooled = _metrics(y, prediction)
            comparisons.append(dict(candidate=spec["candidate"], method=method,
                phase=spec["phase"], stage=spec["stage"], family=spec["family"],
                estimator=spec["estimator"], alpha=spec["alpha"], target=spec["target"],
                n_features=len(spec["features"]), removed=spec["removed"],
                capacity_penalty=spec["capacity_penalty"], cv_mape_pct=np.mean(scores),
                cv_std_pct=np.std(scores, ddof=1), oof_mape_pct=pooled["mape_pct"],
                convergence_warnings=warning_count))
            for row, p in zip(dev.itertuples(), prediction):
                oofs.append(dict(candidate=spec["candidate"], method=method,
                    batch=row.batch, cell_id=row.cell_id, protocol=row.protocol,
                    actual=row.cycle_life, prediction=p,
                    ape_pct=100 * abs(p-row.cycle_life)/row.cycle_life))


def main():
    if (OUT / "results.json").exists():
        print("Completed protocol validation exists; no repeated training or scoring.")
        print(pd.read_csv(OUT / "evaluation_metrics.csv").round(3).to_string(index=False))
        render_saved()
        return
    if (OUT / "selection.json").exists():
        evaluate_saved()
        return
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError("Partial run exists; inspect it before restarting")
    OUT.mkdir(parents=True)
    frame = pd.read_csv(FEATURE_SOURCE, dtype={"cell_id": str})
    if frame.duplicated(["batch", "cell_id"]).any():
        raise ValueError("Duplicate cell identifiers")
    eligible = frame.model_eligible & frame.label_eligible & frame.delta_status.eq("ok")
    batch1 = frame.loc[frame.batch.eq("Batch1") & eligible].sort_values(
        ["batch", "cell_id"]).reset_index(drop=True)
    batch1["protocol"] = protocol(batch1)
    assert len(batch1) == 36
    assert batch1.input_max_cycle.le(100).all() and batch1.change_feature_max_cycle.le(100).all()
    train_ids, valid_ids = next(GroupShuffleSplit(n_splits=1, test_size=0.2,
        random_state=SEED).split(batch1, groups=batch1.protocol))
    dev, valid = batch1.iloc[train_ids].reset_index(drop=True), batch1.iloc[valid_ids].copy()
    assert not set(dev.protocol) & set(valid.protocol)
    folds_by_method = {
        "cell": list(KFold(5, shuffle=True, random_state=SEED).split(dev)),
        "protocol": list(GroupKFold(5, shuffle=False).split(dev, groups=dev.protocol)),
    }
    manifest = batch1[["batch", "cell_id", "charging_policy", "protocol"]].copy()
    manifest["role"] = "holdout"
    manifest.loc[train_ids, "role"] = "development"
    for method, folds in folds_by_method.items():
        manifest[f"{method}_cv_fold"] = pd.Series(pd.NA, index=manifest.index, dtype="Int64")
        for fold, (_, va) in enumerate(folds, 1):
            manifest.loc[train_ids[va], f"{method}_cv_fold"] = fold
    manifest.to_csv(OUT / "split_manifest.csv", index=False)
    initial = stage_specs()
    plan = dict(created_utc=datetime.now(timezone.utc).isoformat(), seed=SEED,
        group_key=["C1", "switch_SOC", "C2"], newstructure_not_in_numeric_key=True,
        holdout="GroupShuffleSplit(test_size=0.2, seed=42); no split search",
        protocol_cv="GroupKFold(5, shuffle=False); deterministic balanced cell counts",
        cell_cv="KFold(5, shuffle=True, seed=42) on exactly the same development cells",
        n_development=len(dev), n_holdout=len(valid),
        n_development_protocols=dev.protocol.nunique(), n_holdout_protocols=valid.protocol.nunique(),
        initial_candidates=initial,
        refinement_rule="Choose feature-stage winner by protocol CV only; evaluate full set and each single-feature removal except core variance. Log Ridge alpha .1/1/10; raw MAE/MAPE alpha 0/.01/.1. Initial-capacity penalties10/100 if present.",
        final_selection="Lowest mean protocol-CV MAPE over initial/refinement candidates and fixed former controls; exact ties prefer fewer features, then fixed candidate order",
        evaluation_rule="Freeze primary winner, same-cell variance baseline, cell-CV-selected diagnostic and fixed former five-feature controls before holdout/Batch2 scoring",
        candidate_adaptation_uses="Batch1 development protocol CV only",
        prior_evaluation_context="Holdout cells and Batch2 previously seen in earlier work; follow-up exploration, not independent confirmatory test",
        batch3_evaluated=False, refit_including_holdout=False,
        source_sha256=digest(FEATURE_SOURCE), label_audit_sha256=digest(ROOT / "outputs/day2/label_audit.csv"),
        versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                      scikit_learn=sklearn.__version__, joblib=joblib.__version__))
    write_json(OUT / "plan.json", plan)
    comparisons, fold_rows, oofs = [], [], []
    cv_compare(initial, dev, folds_by_method, comparisons, fold_rows, oofs)
    stage_cv = pd.DataFrame(comparisons)
    winners = stage_cv.sort_values(["cv_mape_pct", "n_features"], kind="stable").groupby(
        ["stage", "method"], sort=False).head(1).sort_values(["stage", "method"])
    winners.to_csv(OUT / "stage_comparison.csv", index=False)
    best_name = stage_cv.loc[stage_cv.method.eq("protocol")].sort_values(
        ["cv_mape_pct", "n_features"], kind="stable").iloc[0].candidate
    spec_by_name = {s["candidate"]: s for s in initial}
    refine = refinement_specs(spec_by_name[best_name])
    write_json(OUT / "refinement_plan.json", dict(
        frozen_before_refinement=True, feature_winner=best_name, candidates=refine))
    cv_compare(refine, dev, folds_by_method, comparisons, fold_rows, oofs)
    spec_by_name.update({s["candidate"]: s for s in refine})
    cv = pd.DataFrame(comparisons)
    cv.to_csv(OUT / "model_comparison.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(OUT / "fold_metrics.csv", index=False)
    def winner(method):
        return cv.loc[cv.method.eq(method)].sort_values(
            ["cv_mape_pct", "n_features"], kind="stable").iloc[0].candidate
    # Fixed former controls, not selected from new external evaluation scores.
    full5 = list(STAGES[3][3])
    controls = [dict(candidate="control_five_log_ridge", phase="control", stage="control",
        family="standard", estimator="Ridge", alpha=1.0, l1_ratio=None,
        target="log10", features=full5, removed=None, capacity_penalty=1),
        dict(candidate="control_five_mape", phase="control", stage="control",
        family="mape_raw", estimator="QuantileRegressor", alpha=0.01, l1_ratio=None,
        target="raw", features=full5, removed=None, capacity_penalty=1)]
    # Controls have CV scores too, before scoring any holdout/external targets.
    cv_compare(controls, dev, folds_by_method, comparisons, fold_rows, oofs)
    spec_by_name.update({s["candidate"]: s for s in controls})
    cv = pd.DataFrame(comparisons)
    cv.to_csv(OUT / "model_comparison.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(OUT / "fold_metrics.csv", index=False)
    selected, cell_selected = winner("protocol"), winner("cell")
    evaluated = list(dict.fromkeys([selected, "S0_linear", cell_selected] + [s["candidate"] for s in controls]))
    models = {}
    for name in evaluated:
        spec = spec_by_name[name]
        model = build(spec)
        fit(model, spec, dev[spec["features"]], dev.cycle_life.to_numpy(float))
        joblib.dump(model, OUT / f"{name}.joblib")
        models[name] = model
    selection = dict(frozen_utc=datetime.now(timezone.utc).isoformat(), primary=selected,
        cell_cv_diagnostic=cell_selected, feature_stage_winner=best_name,
        evaluated=evaluated, specs={n: spec_by_name[n] for n in evaluated},
        split_manifest_sha256=digest(OUT / "split_manifest.csv"),
        model_hashes={n: digest(OUT / f"{n}.joblib") for n in evaluated},
        primary_selection_metric="protocol CV MAPE only", selected_before_evaluation=True,
        refit_including_holdout=False)
    write_json(OUT / "selection.json", selection)
    print("Frozen protocol-CV winner:", selected, flush=True)
    print("Feature-stage winner:", best_name, "; cell-CV diagnostic:", cell_selected, flush=True)
    evaluate_saved()


def evaluate_saved():
    """Complete evaluation of a frozen run without repeating fits or selection."""
    if any((OUT / name).exists() for name in ("evaluation_metrics.csv", "evaluation_predictions.csv")):
        raise RuntimeError("Saved evaluation exists; refusing to score it again")
    plan = json.loads((OUT / "plan.json").read_text())
    selection = json.loads((OUT / "selection.json").read_text())
    if digest(OUT / "split_manifest.csv") != selection.get("split_manifest_sha256"):
        raise ValueError("Missing or changed frozen split signature; evaluation cannot resume")
    if digest(FEATURE_SOURCE) != plan["source_sha256"]:
        raise ValueError("Feature source changed after selection")
    evaluated, selected = selection["evaluated"], selection["primary"]
    for name in evaluated:
        if digest(OUT / f"{name}.joblib") != selection["model_hashes"][name]:
            raise ValueError("Frozen model changed before evaluation")
    models = {name: joblib.load(OUT / f"{name}.joblib") for name in evaluated}
    spec_by_name = selection["specs"]
    manifest = pd.read_csv(OUT / "split_manifest.csv", dtype={"cell_id": str})
    frame = pd.read_csv(FEATURE_SOURCE, dtype={"cell_id": str})
    batch1 = manifest[["batch", "cell_id", "protocol", "role"]].merge(
        frame, on=["batch", "cell_id"], validate="one_to_one")
    dev = batch1.loc[batch1.role.eq("development")]
    valid = batch1.loc[batch1.role.eq("holdout")]
    assert not set(dev.protocol) & set(valid.protocol)
    cv = pd.read_csv(OUT / "model_comparison.csv")
    winners = pd.read_csv(OUT / "stage_comparison.csv")
    # model_eligible marks Batch1 development eligibility; external labels have
    # their own label_eligible flag and must not use that Batch1-only flag.
    test = frame.loc[frame.batch.eq("Batch2") & frame.label_eligible
                     & frame.delta_status.eq("ok") & frame.cycle_life.gt(100)].copy()
    test["protocol"] = protocol(test)
    assert len(test) == 39
    test["seen_training_protocol"] = test.protocol.isin(dev.protocol)
    scores, predictions, segments = [], [], []
    for name in evaluated:
        model, spec = models[name], spec_by_name[name]
        for label, data in [("Valid (Batch 1 Hold-out)", valid), ("Test (Batch 2)", test)]:
            pred, score = _score(data, model, spec["features"])
            pred = pred.merge(data[["batch", "cell_id", "protocol"]], on=["batch", "cell_id"], validate="one_to_one")
            pred["candidate"], pred["evaluation"] = name, label
            pred["seen_training_protocol"] = pred.protocol.isin(dev.protocol)
            predictions.append(pred)
            scores.append(dict(candidate=name, evaluation=label, **score))
            if label == "Test (Batch 2)":
                for group, g in pred.groupby("seen_training_protocol"):
                    segments.append(dict(candidate=name, grouping="protocol", segment="seen" if group else "unseen",
                        n=len(g), mape_pct=g.ape_pct.mean(), mean_overprediction=-g.residual.mean()))
                for group, g in pred.groupby(pd.cut(pred.cycle_life, [0, 500, 1000, np.inf], labels=["<=500", "500-1000", ">1000"]), observed=True):
                    segments.append(dict(candidate=name, grouping="life", segment=str(group), n=len(g),
                        mape_pct=g.ape_pct.mean(), mean_overprediction=-g.residual.mean()))
    scores = pd.DataFrame(scores)
    scores.to_csv(OUT / "evaluation_metrics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(OUT / "evaluation_predictions.csv", index=False)
    pd.DataFrame(segments).to_csv(OUT / "evaluation_segments.csv", index=False)
    primary_cv = float(cv.loc[cv.candidate.eq(selected) & cv.method.eq("protocol"), "cv_mape_pct"].iloc[0])
    primary_scores = scores.loc[scores.candidate.eq(selected)].set_index("evaluation")
    vm = primary_scores.loc["Valid (Batch 1 Hold-out)", "mape_pct"]
    tm = primary_scores.loc["Test (Batch 2)", "mape_pct"]
    reporting = pd.DataFrame([
        ["Train (Batch 1 CV)", primary_cv, "프로토콜 단위 5-fold 검증 평균"],
        ["Valid (Batch 1 Hold-out)", vm, f"{len(valid)}셀·{valid.protocol.nunique()}개 미관측 프로토콜"],
        ["Test (Batch 2)", tm, "39셀; 이전 열람 이력이 있는 후속 평가"],
        ["Gap (Train-Valid)", vm-primary_cv, "Valid − CV; %p"],
        ["Gap (Valid-Test)", tm-vm, "Test − Valid; %p"],
        ["Gap (Target-Test)", tm-9.1, "Test − 과제 참고값9.1%; %p"]],
        columns=["구분", "MAPE (%)", "비고"])
    reporting.to_csv(OUT / "performance_reporting.csv", index=False)
    summary = cv.loc[cv.method.eq("protocol") & cv.candidate.isin(evaluated), ["candidate", "n_features", "cv_mape_pct"]].merge(
        scores.pivot(index="candidate", columns="evaluation", values="mape_pct"), on="candidate")
    summary.to_csv(OUT / "summary.csv", index=False)
    refinement_plan = json.loads((OUT / "refinement_plan.json").read_text())
    result = dict(**selection, n_stage_candidates=len(plan["initial_candidates"]),
        n_refinement_candidates=len(refinement_plan["candidates"]),
        n_total_cv_fits=len(pd.read_csv(OUT / "fold_metrics.csv")),
        n_development=len(dev), n_holdout=len(valid),
        n_batch2_seen_protocol=int(test.seen_training_protocol.sum()),
        n_batch2_unseen_protocol=int((~test.seen_training_protocol).sum()),
        cv_mape_pct=primary_cv, holdout_mape_pct=float(vm), batch2_mape_pct=float(tm),
        feature_source_sha256=digest(FEATURE_SOURCE), versions=plan["versions"],
        batch3_evaluated=False, input_cutoff_cycle=100,
        interpretation="Many development comparisons; reused holdout cells and previously seen Batch2. CV is selected performance, not unbiased generalization proof.")
    write_json(OUT / "results.json", result)
    draw_and_report(result, cv, winners, summary, reporting, pd.DataFrame(segments), manifest)
    print(summary.round(3).to_string(index=False))
    print(reporting.round(3).to_string(index=False))


def render_saved():
    result = json.loads((OUT / "results.json").read_text())
    draw_and_report(result, pd.read_csv(OUT / "model_comparison.csv"),
        pd.read_csv(OUT / "stage_comparison.csv"), pd.read_csv(OUT / "summary.csv"),
        pd.read_csv(OUT / "performance_reporting.csv"),
        pd.read_csv(OUT / "evaluation_segments.csv"), pd.read_csv(OUT / "split_manifest.csv"))


def draw_and_report(result, cv, stages, summary, reporting, segments, manifest):
    saved_model = joblib.load(OUT / f"{result['primary']}.joblib")
    pipe = saved_model.regressor_ if hasattr(saved_model, "regressor_") else saved_model
    features = result["specs"][result["primary"]]["features"]
    coefficients = pd.DataFrame(dict(feature=features,
        standardized_coefficient=pipe.named_steps["model"].coef_,
        nonzero=np.abs(pipe.named_steps["model"].coef_) > 1e-12))
    coefficients.to_csv(OUT / "selected_coefficients.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), layout="constrained")
    order = ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "F1"]
    for method, label in [("cell", "Cell CV"), ("protocol", "Protocol CV")]:
        values = stages.loc[stages.method.eq(method)].set_index("stage").loc[order]
        axes[0].plot(order, values.cv_mape_pct, "o-", label=label)
    axes[0].set(xlabel="Feature stage", ylabel="MAPE (%)", title="Same development cells; different validation splits")
    axes[0].legend(); axes[0].grid(alpha=.2)
    x = np.arange(len(summary)); width=.25
    for offset, col, label in [(-1, "cv_mape_pct", "Protocol CV"), (0, "Valid (Batch 1 Hold-out)", "Protocol holdout"), (1, "Test (Batch 2)", "Batch2")]:
        axes[1].bar(x+offset*width, summary[col], width, label=label)
    labels = ["Selected" if n == result["primary"] else "Variance" if n == "S0_linear" else "Cell-CV pick" if n == result["cell_cv_diagnostic"] else "Five log Ridge" if n == "control_five_log_ridge" else "Five MAPE" for n in summary.candidate]
    axes[1].set(xticks=x, xticklabels=labels, ylabel="MAPE (%)", title="Frozen candidates; external results")
    axes[1].tick_params(axis="x", labelrotation=20); axes[1].legend(fontsize=8)
    fig.savefig(OUT / "comparison.png", dpi=160); plt.close(fig)
    stages_table = stages.pivot(index="stage", columns="method", values="cv_mape_pct").loc[order].reset_index()
    spec = result["specs"][result["primary"]]
    same_model_cv = cv.loc[cv.candidate.eq(result["primary"])].set_index("method").cv_mape_pct
    baseline = summary.loc[summary.candidate.eq("S0_linear")].iloc[0]
    refinement = cv.loc[cv.method.eq("protocol") & cv.phase.eq("refinement")].sort_values("cv_mape_pct").head(8)
    group_counts = manifest.groupby("role").agg(cells=("cell_id", "size"), protocols=("protocol", "nunique")).reset_index()
    report = f"""# DAY2 부록 — 충전 프로토콜 분리 검증

전체 개발 과정과 두 분리 방식의 비교는 [DAY2 보고서](DAY2_REPORT.md)에 정리했다. 이 부록은 프로토콜 분리 실험의 상세 기록이다.

기존 셀 단위 검증에 이어, 학습에서 보지 못한 충전 조합에 대한 검증으로 특징과 모델을 다시 선택했다. **이 추가 검증의 CV 선택 후보는 `{result['primary']}`이며 프로토콜 CV {result['cv_mape_pct']:.2f}%, Hold-out {result['holdout_mape_pct']:.2f}%, Batch2 {result['batch2_mape_pct']:.2f}%다.**

## 1. 왜 검증을 바꿨는가

기존 Batch2에는 처음 보는 충전 조합이 많았다. 같은 충전 조합의 다른 셀을 검증하는 방식과 처음 보는 조합을 검증하는 방식은 평가 대상이 다르다. 동일 프로토콜 중복을 곧바로 데이터 누수라고 단정하지 않고, 새로운 프로토콜 일반화와의 정합성을 확인했다.

그룹은 C1·전환 SOC·C2의 정확한 수치 조합이며 newstructure는 별도 실험 표시다. Batch1에서는 이 표시가 일정하다. seed42의 단일 GroupShuffleSplit으로 20%의 프로토콜을 Hold-out에 보관했다. 결과에 맞춰 분할을 변경하지 않았다.

{md_table(group_counts, list(group_counts.columns))}

학습/Hold-out 및 프로토콜 CV 각 학습/검증 폴드의 프로토콜 교집합은 모두0이다. 같은 개발 셀에서 셀 KFold와 GroupKFold를 모두 실행했다. 두 CV는 검증 목적이 다르며, 차이를 기계적인 누수량으로 해석하지 않는다.

## 2. 기존 특징 추가 과정 재평가

셀별 초기100사이클 특징 계산은 기존 값을 재사용했다. 결측 대체·표준화는 각 학습 폴드에서 새로 fit했다. S0는 원 수명과 분산 하나, S1은 로그 수명, S2는 최솟값, S3은 초기 용량·변화율·기울기, S4는 IR 변화율, S5는 온도 변화·폭, S6은 충전시간, S7은 C1·SOC·C2 누적 추가다. F1은 DAY1의 분산＋충전시간＋평균 온도 제안이다.

{md_table(stages_table, list(stages_table.columns))}

각 단계에서 기존 선형·Ridge·ElasticNet 후보를 재비교했다. 셀 CV와 프로토콜 CV는 단계별로 서로 다른 설정을 선택할 수 있다. 각 설정을 정확히 짝지은 비교는 `model_comparison.csv`의 phase=features 행에 있다. 특징과 설정을 함께 선택하므로 단계 차이를 개별 특징의 인과효과로 보지 않는다.

프로토콜 CV의 특징 단계 선택: `{result['feature_stage_winner']}`. 셀 CV 전체 선택 진단: `{result['cell_cv_diagnostic']}`.

## 3. 특징 제거·규제·학습 손실

프로토콜 CV에서 고른 특징 집합에 대해 핵심 분산 외 특징을 하나씩 제거했다. 초기 용량은 초기 상태·측정 조건, 변화율·기울기는 작은 변동과 중복, 최솟값은 국소 변동, 저항·온도·충전시간은 실험 조건을 함께 반영할 수 있다는 가설이다. 이 가설이 각 특징의 부정확함을 미리 확정하는 것은 아니다.

같은 특징 집합에서 로그 Ridge alpha0.1/1/10, 원 수명 MAE 및 MAPE 가중 회귀 alpha0/0.01/0.1을 비교했다. MAPE 가중치는 학습 수명의 역수로 해당 폴드에서만 계산했다. 초기 용량이 포함되어 있으면 그 계수 규제10/100배도 비교했다. 모든 후보와 추가 비교 규칙은 Hold-out/Batch2 평가 전에 고정했다.

{md_table(refinement, ['candidate','n_features','cv_mape_pct','cv_std_pct'])}

선택 특징: `{', '.join(spec['features'])}`. 타깃: `{spec['target']}`. 모델: `{spec['estimator']}`, alpha={spec['alpha']}.

{md_table(coefficients, list(coefficients.columns))}

입력 명세는 {len(features)}개이고 개발 셀 전체로 학습한 모델의 0이 아닌 계수는 {int(coefficients.nonzero.sum())}개다. 중복 ΔQ 통계 중 어느 것을 쓰는지는 규제와 데이터에 따라 달라질 수 있다. 계수는 모델 내부의 조건부 관계이며 열화 원인의 독립적 물리 효과로 해석하지 않는다. 다른 CV 폴드에서는 0이 아닌 계수 구성이 달라질 수도 있다.

## 4. 최종 평가

![검증 방식과 최종 평가 비교](protocol_validation/comparison.png)

{md_table(summary, list(summary.columns))}

단일 분산 기준 모델과 최종 후보는 같은 개발 셀로 학습했다. 따라서 이 비교는 학습 셀이 다른 기존26.19%와 직접 대조하는 것보다 적절하다. 기존 셀 분할의 모델에 비해 새 분할은 학습/검증 셀과 학습 범위도 달라져, 외부 점수 변화 전체를 분할 방식의 인과효과로 볼 수 없다.

{md_table(reporting, list(reporting.columns))}

Gap의 단위는 %p이며 Train은 훈련 재예측 오차가 아니라 CV 검증 평균이다. 원논문9.1%는 과제 참고값이며 로컬 파일·학습 구성이 달라 동일 조건 재현이 아니다.

## 5. Batch2의 익숙한 조합과 새로운 조합

이번 개발 셀 기준 Batch2의 기존 조합은 {result['n_batch2_seen_protocol']}셀, 새로운 조합은 {result['n_batch2_unseen_protocol']}셀이다. 이전 학습 집합과 달라 새로 계산했다. 숫자 조합이 같아도 newstructure·배치·수명 분포가 다를 수 있다.

{md_table(segments.loc[segments.grouping.eq('protocol')], ['candidate','segment','n','mape_pct','mean_overprediction'])}

양의 mean_overprediction은 과대 예측이다. 집단별 오차 차이만으로 프로토콜 신규성의 독립적 효과를 증명하지 않는다. 단수명 범위·실험 표시·배치 차이가 같이 존재한다.

## 6. 결과에서 확인한 내용

- 같은 선택 후보의 셀 CV는 {same_model_cv['cell']:.2f}%, 프로토콜 CV는 {same_model_cv['protocol']:.2f}%다. 두 검증 방식이 고른 전체 후보도 {'같았다' if result['primary'] == result['cell_cv_diagnostic'] else '달랐다'}. 프로토콜 검증이 반드시 더 높은 오차를 만드는 것은 아니며, 폴드별 셀·조건 구성에도 영향을 받는다. 이번 결과로 기존 셀 CV의 우수한 점수가 프로토콜 중복 때문이었다고 확정할 수 없다.
- 초기 용량·변화율·기울기까지 추가한 S3가 내부에서 유망했다. 저항·온도·충전시간·충전 조건을 누적 추가하는 단계는 더 좋은 프로토콜 CV 대표를 만들지 못했다. 모든 단독 활용 가능성을 부정하는 결과는 아니다.
- 최종 후보의 Hold-out {result['holdout_mape_pct']:.2f}%는 같은 학습 셀의 단일 분산 모델 {baseline['Valid (Batch 1 Hold-out)']:.2f}%보다 낮았지만, Batch2 {result['batch2_mape_pct']:.2f}%는 분산 모델 {baseline['Test (Batch 2)']:.2f}%보다 높았다. 새로운 프로토콜의 Batch1 셀에서 확인한 내부 개선도 다른 배치의 개선을 보장하지 못했다.
- Batch2의 익숙한 조합은 {result['n_batch2_seen_protocol']}셀뿐이라 새 조합 집단과 안정적으로 비교하기 어렵다. 같은 수치 조합의 셀도 오차가 컸으며, 프로토콜 신규성만으로 성능 저하를 설명할 수 없다.
- 개발 수명은534~1054사이클인데 Batch2에는500이하28셀이 있다. 새 검증 방식으로도 단수명 학습 범위 부족·배치 상태·특징 관계의 차이를 해결하지 못했다. 이 요인들의 개별 인과효과를 분리해 입증한 것은 아니다.
- **이 프로토콜 분리 실험 시점의 후보 범위에서는** 기존 26.19%보다 낮은 Batch2 MAPE를 얻지 못했고, 해당 실행에서는 평가 후 모델을 다시 선택하지 않았다. 이후 [비즈니스 목적 부록](DAY2_BUSINESS_APPENDIX.md)에서 추가 탐색을 했으며, 예전 28셀 분리의 분위수 0.5는 Batch2 23.84%를 기록했다. 현재의 단일 분산 모델 추천은 이러한 후속 결과까지 검토한 사후 판단으로 [DAY2 보고서](DAY2_REPORT.md)에 구분해 기록했다.

## 7. 평가 한계와 실행

새 분할도 이전에 사용한 Batch1 셀을 재사용했고 Batch2도 이미 여러 번 확인했다. 이번 실험은 후속 탐색이며 새 독립적인 최종 검증이라고 주장하지 않는다. 개발 CV를 많은 후보 선택에 재사용했으므로 선택된 CV 최저값도 일반화 성능의 불편 추정량은 아니다.

Batch2 결과로 후보를 다시 선택하거나 특징을 추가하지 않았다. Hold-out 포함 전체36셀 재학습 및 Batch3 추가 평가는 하지 않았다. 초기100사이클만 사용했으며 후기 knee·종료 용량·실제 총수명은 입력에 없다.

실행: `.venv/bin/python -m src.day2_protocol_validation`. 기존 결과가 있으면 재학습·재평가를 반복하지 않는다. `plan.json`, `refinement_plan.json`, `selection.json`, 분할표·CV 폴드 점수·예측·모델·환경 버전은 `protocol_validation/`에 있다.
"""
    (ROOT / "outputs/day2/DAY2_PROTOCOL_VALIDATION.md").write_text(report)


if __name__ == "__main__":
    main()
