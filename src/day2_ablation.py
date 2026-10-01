"""Controlled feature removal and MAPE-aligned loss, on the existing DAY2 split.

Run: python -m src.day2_ablation. Completed runs are read from disk, not repeated.
Select only on Batch1 CV; freeze three family representatives before evaluation.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import platform
import shutil

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import QuantileRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .day2_experiments import make_model, write_json
from .day2_modeling import _metrics
from .day2_evaluation import _score

SEED = 42
FEATURES = ["log10_delta_Q_var", "delta_Q_min", "qd_initial_median",
            "qd_fraction_change", "early_drop_per100"]
REMOVALS = [None, "qd_initial_median", "qd_fraction_change",
            "early_drop_per100", "delta_Q_min"]
LABELS = {None: "5개 특징 유지", "qd_initial_median": "초기 용량 제거",
          "qd_fraction_change": "용량 변화율 제거", "early_drop_per100": "초기 감소 속도 제거",
          "delta_Q_min": "ΔQ 최솟값 제거"}
HYPOTHESES = {
    "qd_initial_median": "초기 용량은 초기 상태·측정 조건도 반영하므로 배치별 수준 차이를 학습했을 가능성",
    "qd_fraction_change": "초기 용량 증가는 이후 열화와 공존할 수 있으므로 증가율과 수명의 관계가 배치마다 달라질 가능성",
    "early_drop_per100": "초기 용량 변화가 작아 기울기가 일시적 변동에 민감하고, 변화율과 정보가 겹칠 가능성",
    "delta_Q_min": "국소적인 최대 곡선 변화는 잡음에도 민감할 수 있고, 분산과 중복될 가능성",
}
FAMILIES = ["ridge_log", "mae_raw", "mape_raw"]


def specs():
    for removed in REMOVALS:
        tag = "full" if removed is None else "minus_" + removed
        cols = [c for c in FEATURES if c != removed]
        yield dict(candidate=f"ridge_{tag}", family="ridge_log", removed=removed,
                   title=LABELS[removed], features=cols, alpha=1.0, target="log10")
        for family in ["mae_raw", "mape_raw"]:
            for alpha in [0.0, 0.01, 0.1]:
                yield dict(candidate=f"{family}_{tag}_a{alpha:g}", family=family,
                           removed=removed, title=LABELS[removed], features=cols,
                           alpha=alpha, target="raw")


def model_for(spec):
    if spec["family"] == "ridge_log":
        return make_model(dict(estimator="Ridge", alpha=1.0, l1_ratio=None), "log10")
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", QuantileRegressor(quantile=0.5, alpha=spec["alpha"], solver="highs")),
    ])


def fit_model(model, spec, x_train, y_train):
    # Targets enter the loss only, never the predictor matrix or preprocessing.
    if spec["family"] == "mape_raw":
        w = 1.0 / y_train
        w = w / w.mean()  # Training-fold normalization; proportional to MAPE.
        model.fit(x_train, y_train, model__sample_weight=w)
    else:
        model.fit(x_train, y_train)
    return model


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    parent = root / "outputs/day2/feature_experiments"
    output = root / "outputs/day2/ablation_loss"
    if (output / "results.json").exists():
        print("Completed run exists; use saved results, no repeated fitting or evaluation.")
        return
    output.mkdir(parents=True, exist_ok=True)
    if (output / "selection.json").exists():
        raise RuntimeError("An incomplete frozen run exists; inspect it before restarting.")
    all_specs = list(specs())
    write_json(output / "experiment_plan.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(), "seed": SEED,
        "hypotheses": HYPOTHESES, "candidates": all_specs,
        "selection": "Mean Batch1 five-fold validation MAPE; ties follow fixed candidate order",
        "evaluation_plan": "Freeze one CV winner in each family and the overall CV winner before scoring Hold-out and previously seen Batch2",
        "fit_cells": 28, "holdout_cells": 8, "batch2_cells": 39,
        "refit_with_holdout": False, "batch3_evaluation": False,
        "mape_objective": "0.5 * normalized weighted absolute error + alpha * L1; weights proportional to 1 / training target",
        "alphas": [0.0, 0.01, 0.1], "target_log_for_quantile": False,
    })
    frame = pd.read_csv(parent / "cell_features.csv", dtype={"cell_id": str})
    manifest = pd.read_csv(parent / "split_manifest.csv", dtype={"cell_id": str})
    if frame.duplicated(["batch", "cell_id"]).any():
        raise ValueError("Duplicate cells")
    joined = manifest[["batch", "cell_id", "role", "cv_fold"]].merge(
        frame, on=["batch", "cell_id"], validate="one_to_one", sort=False)
    dev = joined.loc[joined.role.eq("development")].reset_index(drop=True)
    if len(dev) != 28 or not dev.batch.eq("Batch1").all():
        raise ValueError("Development split changed")
    if not dev.input_max_cycle.le(100).all() or not dev.change_feature_max_cycle.le(100).all():
        raise ValueError("Feature beyond cycle100")
    y = dev.cycle_life.to_numpy(float)
    if not np.isfinite(y).all() or not (y > 100).all():
        raise ValueError("Invalid training lifetime")
    fold_ids = dev.cv_fold.astype(int).to_numpy()
    if set(fold_ids) != set(range(1, 6)):
        raise ValueError("Unexpected folds")
    rows, folds, oofs = [], [], []
    for spec in all_specs:
        x = dev[spec["features"]]
        pred = np.full(len(y), np.nan)
        fold_mapes = []
        for fold in range(1, 6):
            tr, va = np.flatnonzero(fold_ids != fold), np.flatnonzero(fold_ids == fold)
            if x.iloc[tr].isna().all().any() or np.isinf(x.to_numpy(float)).any():
                raise ValueError("Unusable input feature")
            model = fit_model(model_for(spec), spec, x.iloc[tr], y[tr])
            pred[va] = model.predict(x.iloc[va])
            m = _metrics(y[va], pred[va])
            fold_mapes.append(m["mape_pct"])
            folds.append(dict(candidate=spec["candidate"], fold=fold, n_train=len(tr), n_valid=len(va), **m))
        m = _metrics(y, pred)
        rows.append(dict(candidate=spec["candidate"], family=spec["family"],
                         removal=spec["removed"] or "none", title=spec["title"],
                         alpha=spec["alpha"], features=",".join(spec["features"]),
                         cv_mape_pct=float(np.mean(fold_mapes)),
                         cv_std_pp=float(np.std(fold_mapes, ddof=1)),
                         oof_mape_pct=m["mape_pct"], oof_mae=m["mae"], oof_rmse=m["rmse"],
                         oof_r2=m["r2"], mean_prediction_minus_actual=float((pred-y).mean()),
                         underprediction_pct=float(100*(pred < y).mean()),
                         nonpositive_predictions=int((pred <= 0).sum())))
        oofs.extend(dict(candidate=spec["candidate"], cell_id=dev.iloc[i].cell_id,
                         fold=int(fold_ids[i]), actual=y[i], prediction=pred[i],
                         ape_pct=100*abs(pred[i]-y[i])/y[i]) for i in range(len(y)))
    comparison = pd.DataFrame(rows).sort_values("cv_mape_pct", kind="stable").reset_index(drop=True)
    comparison.insert(0, "rank", np.arange(1, len(comparison)+1))
    lookup = {s["candidate"]: s for s in all_specs}
    old = json.loads((parent / "results.json").read_text())
    base = comparison.loc[comparison.candidate.eq("ridge_full")].iloc[0]
    if not np.isclose(base.cv_mape_pct, old["cv_mape_pct"], atol=1e-9):
        raise ValueError("Baseline CV failed to reproduce")
    chosen = {family: comparison.loc[comparison.family.eq(family)].iloc[0].candidate for family in FAMILIES}
    selected = comparison.iloc[0].candidate
    frozen = {"overall_cv_winner": selected, "family_cv_winners": chosen,
              "selected_using": "Batch1 CV only", "frozen_utc": datetime.now(timezone.utc).isoformat(),
              "model_specs": {n: lookup[n] for n in chosen.values()},
              "holdout_used_for_selection": False, "batch2_used_for_selection": False,
              "historical_holdout_and_batch2_results_already_seen": True,
              "new_hypotheses_informed_by_previous_batch2_diagnosis": True}
    write_json(output / "selection.json", frozen)
    comparison.to_csv(output / "model_comparison.csv", index=False)
    fold_frame = pd.DataFrame(folds)
    fold_frame.to_csv(output / "fold_metrics.csv", index=False)
    pd.DataFrame(oofs).to_csv(output / "oof_predictions.csv", index=False)
    ablation = comparison.loc[comparison.family.eq("ridge_log")].copy()
    ablation["cv_change_vs_full_pp"] = ablation.cv_mape_pct-base.cv_mape_pct
    ablation.to_csv(output / "ablation_comparison.csv", index=False)
    paired = comparison.loc[comparison.family.eq("mae_raw")].merge(
        comparison.loc[comparison.family.eq("mape_raw")], on=["removal", "alpha"],
        suffixes=("_mae", "_mape"), validate="one_to_one")
    paired["cv_weighting_change_pp"] = paired.cv_mape_pct_mape-paired.cv_mape_pct_mae
    paired["oof_mae_change"] = paired.oof_mae_mape-paired.oof_mae_mae
    paired["oof_rmse_change"] = paired.oof_rmse_mape-paired.oof_rmse_mae
    paired.to_csv(output / "paired_loss_comparison.csv", index=False)
    # The same development-trained objects score both reserved datasets.
    valid = joined.loc[joined.batch.eq("Batch1") & joined.role.eq("holdout")].copy()
    test = joined.loc[joined.batch.eq("Batch2") & joined.label_eligible
                      & joined.delta_status.eq("ok") & joined.cycle_life.gt(100)].copy()
    if len(valid) != 8 or len(test) != 39:
        raise ValueError("Evaluation cohort changed")
    score_rows, predictions, model_hashes = [], [], {}
    for family, name in chosen.items():
        spec = lookup[name]
        model = fit_model(model_for(spec), spec, dev[spec["features"]], y)
        path = output / (name + ".joblib")
        joblib.dump(model, path)
        model = joblib.load(path)
        model_hashes[name] = digest(path)
        for evaluation, subset in [("Valid (Batch 1 Hold-out)", valid), ("Test (Batch 2)", test)]:
            records, m = _score(subset, model, spec["features"])
            records["candidate"], records["family"], records["evaluation"] = name, family, evaluation
            predictions.append(records)
            score_rows.append(dict(candidate=name, family=family, evaluation=evaluation, **m,
                                   mean_prediction_minus_actual=float(-records.residual.mean()),
                                   underprediction_pct=float(100*records.residual.gt(0).mean()),
                                   nonpositive_predictions=int(records.prediction.le(0).sum())))
    scores = pd.DataFrame(score_rows)
    scores.to_csv(output / "evaluation_metrics.csv", index=False)
    pred = pd.concat(predictions, ignore_index=True)
    pred.to_csv(output / "evaluation_predictions.csv", index=False)
    pred["life_band"] = pd.cut(pred.cycle_life, [-np.inf, 500, 1000, np.inf], labels=["<=500", "500-1000", ">1000"])
    pred.groupby(["candidate", "evaluation", "life_band"], observed=True).agg(
        n=("cell_id", "size"), mape_pct=("ape_pct", "mean"),
        mean_prediction_minus_actual=("residual", lambda a: -a.mean())).to_csv(output / "evaluation_segments.csv")
    winner = comparison.iloc[0]
    win_scores = scores.loc[scores.candidate.eq(selected)].set_index("evaluation")
    vm, tm = float(win_scores.loc["Valid (Batch 1 Hold-out)", "mape_pct"]), float(win_scores.loc["Test (Batch 2)", "mape_pct"])
    reporting = pd.DataFrame([
        ["Train (Batch 1 CV)", winner.cv_mape_pct, "28개 학습 셀의 동일 5-fold 검증 평균"],
        ["Valid (Batch 1 Hold-out)", vm, "8개; 현재 선택에 사용하지 않음; 이전 평가 이력 있음"],
        ["Test (Batch 2)", tm, "39개; 고정 모델 후속 평가; 새 독립 Test 아님"],
        ["Gap (Train-Valid)", vm-winner.cv_mape_pct, "Valid − CV; %p"],
        ["Gap (Valid-Test)", tm-vm, "Test − Valid; %p"],
        ["Gap (Target-Test)", tm-9.1, "Test − 9.1; %p; 원논문과 조건 다른 참고 비교"],
    ], columns=["구분", "MAPE (%)", "비고"])
    reporting.to_csv(output / "performance_reporting.csv", index=False)
    result = dict(**frozen, cv_mape_pct=float(winner.cv_mape_pct), n_candidates=len(all_specs),
                  n_cv_fits=len(all_specs)*5, seed=SEED, evaluation=scores.to_dict("records"),
                  refit_including_holdout=False, batch3_evaluated=False,
                  weights_min=float((1/y).min()), weights_max=float((1/y).max()),
                  weight_max_min_ratio=float(y.max()/y.min()),
                  uncertainty="Fixed small development set; repeated prior CV and holdout use; exploration, not independent proof",
                  versions=dict(python=platform.python_version(), numpy=np.__version__,
                                pandas=pd.__version__, sklearn=sklearn.__version__, joblib=joblib.__version__),
                  sha256=dict(code=digest(Path(__file__)), features=digest(parent / "cell_features.csv"),
                              manifest=digest(parent / "split_manifest.csv"), models=model_hashes))
    write_json(output / "results.json", result)
    plot_results(output, comparison, chosen, scores)
    write_report(root, output, comparison, ablation, paired, chosen, scores, result, reporting)
    print(comparison.head(8)[["candidate", "cv_mape_pct", "oof_mae", "oof_rmse"]].round(4).to_string(index=False))
    print(scores[["candidate", "evaluation", "mape_pct", "mae", "rmse"]].round(4).to_string(index=False))


def md_table(frame, columns):
    f = frame[columns].copy()
    return "| " + " | ".join(columns) + " |\n| " + " | ".join(["---"]*len(columns)) + " |\n" + "\n".join(
        "| " + " | ".join(f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v) for v in row) + " |"
        for row in f.itertuples(index=False, name=None))


def plot_results(output, comparison, chosen, scores):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
    ridge = comparison.loc[comparison.family.eq("ridge_log")].set_index("removal")
    ordered = ["none", "qd_initial_median", "qd_fraction_change", "early_drop_per100", "delta_Q_min"]
    axes[0].barh(["Full 5 features", "Remove initial QD", "Remove QD change", "Remove QD slope", "Remove delta Q min"], ridge.loc[ordered, "cv_mape_pct"])
    axes[0].invert_yaxis()
    axes[0].set(xlabel="Batch1 CV MAPE (%)", title="Fixed log-target Ridge (alpha=1)")
    axes[0].axvline(ridge.loc["none", "cv_mape_pct"], color="gray", linestyle="--")
    pos = np.arange(3)
    for offset, kind, label in [(-0.25, "cv", "CV"), (0, "Valid (Batch 1 Hold-out)", "Hold-out"), (0.25, "Test (Batch 2)", "Batch2 follow-up")]:
        values = [float(comparison.set_index("candidate").loc[chosen[f], "cv_mape_pct"]) if kind == "cv" else float(scores.loc[(scores.family.eq(f)) & scores.evaluation.eq(kind), "mape_pct"].iloc[0]) for f in FAMILIES]
        axes[1].bar(pos+offset, values, width=0.25, label=label)
    axes[1].set(xticks=pos, xticklabels=["Log Ridge", "MAE", "MAPE weighted"], ylabel="MAPE (%)", title="CV-selected family representatives")
    axes[1].legend(fontsize=8)
    fig.savefig(output / "comparison.png", dpi=150)
    plt.close(fig)


def write_report(root, output, comparison, ablation, paired, chosen, scores, result, reporting):
    removed_initial = ablation.loc[ablation.removal.eq("qd_initial_median")].iloc[0]
    control_pair = paired.loc[paired.removal.eq("delta_Q_min") & paired.alpha.eq(0.01)].iloc[0]
    full_pair = paired.loc[paired.removal.eq("none") & paired.alpha.eq(0.01)].iloc[0]
    best_cv_std = comparison.iloc[0].cv_std_pp
    parts = ["# DAY2 특징 제거·MAPE 학습 비교\n",
             f"현재 비교의 CV 선택은 **{result['overall_cv_winner']}**이며 CV MAPE는 **{result['cv_mape_pct']:.3f}%**다. 선택은 Batch1 CV로만 했고, 후속 평가 결과로 모델을 바꾸지 않았다.\n",
             "## 1. 무엇을 고정했는가?\n",
             "Batch1 학습 28개·Hold-out 8개, 기존 5개 CV 폴드와 seed42, 수명값과 셀 제외 기준을 유지했다. Hold-out을 포함한 36개 재학습은 하지 않았다. 입력은 초기 100사이클 이내이며, 각 CV 학습 폴드에서만 중앙값 대체와 표준화를 fit했다. 원본 데이터와 이전 실험 결과를 변경하지 않았다.\n",
             "## 2. 도메인 근거에서 세운 가설\n",
             "원논문에서는 초기 용량과 로그 수명의 관계가 약하고, 100사이클까지 용량이 증가하는 셀이 많았다. 용량 변화가 곧바로 열화량을 뜻하지 않으므로 총 용량과 전압 곡선 변화의 정보를 구분해야 한다. ΔQ 통계는 곡선 변화를 요약하지만 특정 열화 기전을 확정하는 지표는 아니다. [Severson·Attia 원논문](https://www.nature.com/articles/s41560-019-0356-8)\n",
             md_table(pd.DataFrame([{"제거 특징": k, "검증할 가설": v} for k, v in HYPOTHESES.items()]), ["제거 특징", "검증할 가설"]),
             "\n배치 차이는 이전 Batch2 진단에서 발견했으므로 이 가설은 외부 평가 정보를 이미 본 이후의 탐색이다. 아래 CV 결과가 도메인 가설의 물리적 인과를 입증하지는 않는다.\n",
             "## 3. 특징 하나씩 제거한 결과\n",
             "5개 특징 기준에서 하나씩만 제거했다. Target은 log10 수명, Ridge alpha=1로 고정했다. 나머지 특징·모델·분할을 유지했으며 제거마다 규제 강도를 다시 튜닝하지 않았다. 변화값은 제거 모델 CV − 전체 특징 CV이며 음수일수록 개선이다.\n",
             md_table(ablation.sort_values("rank"), ["title", "cv_mape_pct", "cv_change_vs_full_pp", "oof_mae", "oof_rmse"]),
             f"\n**실제 결과:** 초기 용량을 빼면 CV가 7.698%에서 {removed_initial.cv_mape_pct:.3f}%로 악화했다. 초기 용량은 Batch1 내부에서 유용했고, 배치 차이가 있다는 이유만으로 제거할 근거는 부족하다. ΔQ 최솟값·변화율·기울기 제거의 개선은 모두 0.012%p 이내로 아주 작다. 불필요하다고 확정하기보다 단순화 후보로 볼 수 있다. 각 제거 모델을 모두 Batch2에서 평가하지 않았으므로 초기 용량 제거가 배치 일반화를 개선한다는 가설 자체는 미확인이다.\n",
             "\n초기 용량 제거 후 CV가 악화하면 Batch1 내부에서는 유용하다는 뜻이다. 그것만으로 다른 배치에서도 안정적이라고 판단할 수 없다. 제거 후 개선하면 해당 모델에서 중복·잡음 가능성을 지지하지만 실제 배터리 기전이나 모든 모델에서의 무용성을 확정하지 않는다.\n",
             "## 4. MAPE에 맞춘 학습은 어떻게 비교했는가?\n",
             "두 모델 모두 원 수명 Target, QuantileRegressor quantile=0.5, 동일 전처리와 L1 alpha 후보 0·0.01·0.1을 사용했다. 무가중 모델은 절대오차를 최소화하며, 가중 모델은 각 학습 폴드에서 1/수명을 계산해 가중 절대오차를 최소화한다. 가중치는 해당 학습 폴드 평균이 1이 되게 정규화했다. alpha=0이면 MAPE와 비례하는 목적함수, alpha>0이면 여기에 L1 규제가 추가된다. [MAPE 회귀 연구](https://arxiv.org/abs/1605.02541), [QuantileRegressor 공식 문서](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.QuantileRegressor.html)\n",
             "수명은 학습 손실의 가중치 계산에만 사용하며 입력 X·결측 대체·표준화에는 사용하지 않는다. 검증/테스트 예측에는 실제 수명이 필요하지 않다. 로그 수명 오차에 1/수명을 곱하지 않았고, 예측값을 임의로 잘라내거나 보정하지 않았다.\n",
             f"학습 수명 최소/최대에 따른 가중치 최대/최소 비율은 **{result['weight_max_min_ratio']:.3f}배**다. 0에 가까운 수명은 없어 분모 폭발은 없으나 단수명 오차를 더 중시하는 특성은 남는다. 장수명 MAE·RMSE와 과소 예측도 함께 확인했다.\n",
             "가중치 효과는 동일 특징·동일 alpha를 짝지어 비교해야 한다. 다음 값은 가중 모델 − 무가중 모델이며, 아래 표의 CV 변화가 음수면 MAPE 개선이다.\n",
             md_table(paired.sort_values(["removal", "alpha"]), ["removal", "alpha", "cv_mape_pct_mae", "cv_mape_pct_mape", "cv_weighting_change_pp", "oof_mae_change", "oof_rmse_change"]),
             f"\n**실제 결과:** 15개의 동일 특징·alpha 짝 모두 가중 모델의 CV MAPE가 낮았다. 전체 특징·alpha=0.01에서는 {full_pair.cv_mape_pct_mae:.3f}% → {full_pair.cv_mape_pct_mape:.3f}%로 개선했다. 그러나 ΔQ 최솟값을 제외한 같은 alpha에서는 {control_pair.cv_mape_pct_mae:.3f}% → {control_pair.cv_mape_pct_mape:.3f}%로 차이가 {abs(control_pair.cv_weighting_change_pp):.3f}%p뿐이었다. 즉 가중치 개선 효과는 특징 구성에 따라 달라졌으며, 잘 구성한 무가중 모델에 비해 압도적인 이득이라고 볼 수 없다. 선택 모델의 CV 폴드간 표준편차는 {best_cv_std:.3f}%p다.\n",
             "\nRidge와 QuantileRegressor 비교에는 손실뿐 아니라 Target 로그 변환·규제 방식도 달라지므로 그 차이를 전부 가중치 효과로 해석하지 않는다.\n",
             "## 5. CV 선택 후 고정한 후속 평가\n",
             "총 35설정·175회 CV 학습을 진행했다. 평균 CV MAPE로 전체 최저 모델과 세 계열의 대표를 미리 고정하고, 각 대표를 학습용 28개로 학습했다. 같은 저장 모델을 Hold-out과 Batch2에 적용했다. 세 대표는 방법별 진단용이며 테스트 최저 모델을 다시 선택하지 않는다.\n",
             md_table(scores, ["family", "candidate", "evaluation", "mape_pct", "mae", "rmse", "mean_prediction_minus_actual", "underprediction_pct"]),
             "\nmean_prediction_minus_actual은 예측 − 실제의 평균(양수면 과대 예측)이다. underprediction_pct는 과소 예측 셀 비율이다.\n",
             "**실제 해석:** CV 선택 MAPE 모델은 Hold-out 4.410%, Batch2 33.876%였다. 기존 5개 특징 Ridge의 3.421%·38.198%와 비교하면 Hold-out은 악화하고 Batch2는 개선했다. ΔQ 최솟값을 제거한 Ridge는 CV가 0.012%p 개선됐지만 Batch2는 43.588%로 악화했다. 작은 내부 개선이 외부 개선을 보장하지 않는다. 무가중 계열 대표는 Hold-out 3.549%·Batch2 33.147%였으나, 이를 보고 현재 CV 선택을 바꾸지 않았다.\n",
             "MAPE 모델에서 초기 감소 속도의 최종 계수는 0이고, 해당 특징을 제거한 CV도 같았다. L1 규제가 그 특징을 사용하지 않는 해를 선택했다는 의미이며, 모든 데이터에서 불필요한 특징이라는 의미는 아니다. Batch2 500 이하 28개에서 MAPE 37.246%·평균 과대 예측 162.860사이클이 남았다. 장수명 3개는 평균 107.221사이클 과소 예측됐지만 소수 표본이므로 가중치의 인과 효과로 단정하지 않는다.\n",
             "![동일 조건 특징 제거와 계열별 고정 모델 평가](ablation_loss/comparison.png)\n",
             "## 6. 과제 성능 표 — 현재 CV 선택 모델\n",
             md_table(reporting, ["구분", "MAPE (%)", "비고"]),
             "\n이전 Hold-out과 Batch2 결과를 이미 확인한 상태이며, 현재 후보 선택에는 두 점수를 사용하지 않았다. 가설 생성 과정은 이전 외부 진단의 영향을 받았으므로 새로운 독립 최종 테스트라고 표현하지 않는다. 반복 CV의 최저 점수도 선택 편향이 있고, 8개 Hold-out에서 작은 차이를 확정적 개선으로 해석하지 않는다.\n",
             "## 7. 제출 시 해석의 범위\n",
             "초기 용량·초기 변화 신호의 도메인 가설을 세우고 고정 모델의 개별 제거 실험으로 검토했다. MAPE 학습은 정상적인 지도학습이며 학습 폴드 내 가중치로 누수를 막았지만, 단수명 중심 평가가 장수명 오차와 충돌할 수 있어 보조 지표를 함께 보고했다. Batch1에 500 미만 학습 셀이 없고 제공 수명에는 종료 경계 레이블 한계가 있어, 학습 목적 변경만으로 배치 차이를 해결했다고 주장할 수 없다.\n",
             "9.1%는 과제 참고 기준이다. 원논문의 학습·테스트 구성과 로컬 Batch2 파일이 달라 정확한 재현 비교가 아니다. Batch3는 이번 실험에서 평가하지 않았다. 기존 단일 특징 모델과 기존 특징 추가 결과도 그대로 보존했다.\n",
             "**모델링 시사점:** 초기 용량은 내부 예측에 유용하지만 배치 간 안정성은 별개이고, MAPE 정렬은 내부 목적함수 개선에 도움이 됐으나 일반화 문제를 해결하지 못했다. 현재 CV 선택 모델도 기존 단일 분산 모델의 Batch2 26.186%를 넘지 못했다. 외부 결과로 최종 모델을 재선택하는 대신, 어떤 선택 기준을 사용했는지와 실패한 일반화 결과를 함께 보고한다. 이번 결과만으로 전체 모델 개발의 최종 모델을 새로 확정하지 않는다.\n",
             "## 8. 재현 파일\n",
             "실행: `.venv/bin/python -m src.day2_ablation`. 완료된 결과가 있으면 재학습·재평가하지 않는다. 결과 폴더 `outputs/day2/ablation_loss/`에 계획·가설·전체 후보·동일 조건 짝 비교·폴드별 결과·OOF·선택 명세·모델·예측·성능 표·버전·해시를 저장했다. 노트북 `notebooks/42-ESSHealth-DAY2-ablation-loss.ipynb`에서 저장 결과와 해석을 확인할 수 있다.\n"]
    (root / "outputs/day2/DAY2_ABLATION_LOSS.md").write_text("\n".join(parts))


if __name__ == "__main__":
    main()
