"""Last, preset diagnostic: penalize initial QD only in the original log Ridge.

No external-score tuning or final model reselection. Run as a module.
"""
from pathlib import Path
from datetime import datetime, timezone
from functools import partial
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
from sklearn.preprocessing import FunctionTransformer

from .day2_experiments import make_model, write_json
from .day2_ablation import md_table
from .day2_modeling import _metrics
from .day2_evaluation import _score

FEATURES = ["log10_delta_Q_var", "delta_Q_min", "qd_initial_median",
            "qd_fraction_change", "early_drop_per100"]
INITIAL = "qd_initial_median"
CONDITIONS = [("baseline", "기존 규제", 1.0), ("penalty10", "초기 용량 규제 10배", 10.0),
              ("penalty100", "초기 용량 규제 100배", 100.0), ("removed", "초기 용량 완전 제거", None)]


def build_model(penalty):
    columns = [c for c in FEATURES if penalty is not None or c != INITIAL]
    model = make_model(dict(estimator="Ridge", alpha=1.0, l1_ratio=None), "log10")
    factors = np.ones(len(columns))
    if penalty is not None:
        factors[columns.index(INITIAL)] = 1 / np.sqrt(penalty)
    # After standardization. Re-standardizing after this would cancel the penalty.
    model.regressor.steps.insert(2, ("capacity_penalty", FunctionTransformer(
        partial(np.multiply, factors))))
    return model, columns, factors


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    parent = root / "outputs/day2/feature_experiments"
    prior = root / "outputs/day2/ablation_loss"
    out = root / "outputs/day2/capacity_penalty"
    if (out / "results.json").exists():
        print("Saved diagnostic exists; no repeated training or evaluation.")
        return
    out.mkdir(parents=True, exist_ok=True)
    if (out / "evaluation_metrics.csv").exists():
        required = ("comparison.csv", "segments.csv", "coefficients.csv", "evaluation_predictions.csv")
        if not all((out / name).is_file() for name in required):
            raise RuntimeError("Partial evaluation exists; inspect it before finalizing")
        complete_saved(root, out, parent)
        return
    if any(out.iterdir()):
        raise RuntimeError("Partial experiment exists; refusing to overwrite it")
    write_json(out / "plan.json", dict(
        frozen_utc=datetime.now(timezone.utc).isoformat(), seed=42,
        hypothesis="Reduced initial-capacity dependence may worsen Batch1 CV but reduce Batch2 overprediction",
        conditions=[dict(name=n, title=t, initial_capacity_penalty=p) for n, t, p in CONDITIONS],
        other_feature_penalties=1.0, target="log10 provided cycle_life",
        fit_scope="Batch1 development28 only; same CV5 and holdout8",
        baseline="Reuse existing model and saved scores",
        removal="Reuse existing five-fold scores; fit one development-only control for previously unscored holdout/Batch2",
        new_cv_models=["penalty10", "penalty100"],
        batch2_context="Already seen; preset follow-up diagnostic, not an untouched independent test",
        select_using_batch2=False, final_model_reselection=False, more_tuning=False))
    frame = pd.read_csv(parent / "cell_features.csv", dtype={"cell_id": str})
    manifest = pd.read_csv(parent / "split_manifest.csv", dtype={"cell_id": str})
    joined = manifest[["batch", "cell_id", "role", "cv_fold"]].merge(
        frame, on=["batch", "cell_id"], validate="one_to_one", sort=False)
    dev = joined.loc[joined.role.eq("development")].reset_index(drop=True)
    valid = joined.loc[joined.batch.eq("Batch1") & joined.role.eq("holdout")].copy()
    test = joined.loc[joined.batch.eq("Batch2") & joined.label_eligible
                      & joined.delta_status.eq("ok") & joined.cycle_life.gt(100)].copy()
    assert len(dev) == 28 and len(valid) == 8 and len(test) == 39
    assert dev.batch.eq("Batch1").all() and dev.input_max_cycle.le(100).all()
    assert dev.change_feature_max_cycle.le(100).all()
    y = dev.cycle_life.to_numpy(float)
    fold_id = dev.cv_fold.astype(int).to_numpy()
    assert set(fold_id) == set(range(1, 6)) and np.isfinite(y).all() and (y > 100).all()
    old_results = json.loads((parent / "results.json").read_text())
    old_comparison = pd.read_csv(prior / "ablation_comparison.csv")
    removal_cv = old_comparison.loc[old_comparison.removal.eq(INITIAL)].iloc[0]
    old_fold = pd.read_csv(parent / "fold_metrics.csv")
    old_remove_fold = pd.read_csv(prior / "fold_metrics.csv")
    folds, cv_rows, oofs, eval_rows, pred_rows, coef_rows, hashes = [], [], [], [], [], [], {}
    for name, title, penalty in CONDITIONS:
        if name == "baseline":
            cv = old_results["cv_mape_pct"]
            fr = old_fold.loc[old_fold.candidate.eq("S3_ridge_1")].copy()
            fr["condition"] = name
            folds.extend(fr.to_dict("records"))
            cv_sd = float(fr.mape_pct.std(ddof=1))
            model = joblib.load(parent / "selected_pipeline.joblib")
            cols, factors = FEATURES, np.ones(len(FEATURES))
            old_pred = pd.read_csv(parent / "evaluation_predictions.csv", dtype={"cell_id": str})
            for evaluation, m in old_results["scores"].items():
                records = old_pred.loc[old_pred.evaluation.eq(evaluation)].copy()
                records["condition"] = name
                pred_rows.append(records)
                eval_rows.append(dict(condition=name, title=title, evaluation=evaluation, **m,
                    mean_prediction_minus_actual=float(-records.residual.mean())))
            hashes[name] = sha(parent / "selected_pipeline.joblib")
        else:
            model, cols, factors = build_model(penalty)
            x = dev[cols]
            assert not np.isinf(x.to_numpy(float)).any()
            if name == "removed":
                cv, cv_sd = float(removal_cv.cv_mape_pct), float(removal_cv.cv_std_pp)
                fr = old_remove_fold.loc[old_remove_fold.candidate.eq("ridge_minus_qd_initial_median")].copy()
                fr["condition"] = name
                folds.extend(fr.to_dict("records"))
            else:
                pred = np.full(len(y), np.nan)
                mapes = []
                for fold in range(1, 6):
                    tr, va = np.flatnonzero(fold_id != fold), np.flatnonzero(fold_id == fold)
                    assert not x.iloc[tr].isna().all().any()
                    fitted, _, _ = build_model(penalty)
                    fitted.fit(x.iloc[tr], y[tr])
                    pred[va] = fitted.predict(x.iloc[va])
                    m = _metrics(y[va], pred[va])
                    mapes.append(m["mape_pct"])
                    folds.append(dict(condition=name, fold=fold, n_train=len(tr), n_valid=len(va), **m))
                cv, cv_sd = float(np.mean(mapes)), float(np.std(mapes, ddof=1))
                oofs.extend(dict(condition=name, cell_id=dev.iloc[i].cell_id, fold=int(fold_id[i]),
                    actual=y[i], prediction=pred[i], ape_pct=100*abs(pred[i]-y[i])/y[i]) for i in range(len(y)))
            model.fit(x, y)
            # Verify feature-specific penalties against the corresponding normal equations.
            pipe = model.regressor_
            z = pipe.named_steps["scaler"].transform(pipe.named_steps["imputer"].transform(x))
            beta = pipe.named_steps["model"].coef_ * factors
            direct = np.linalg.solve(z.T @ z + np.diag(1/factors**2), z.T @ (np.log10(y)-np.log10(y).mean()))
            assert np.allclose(beta, direct, atol=1e-10)
            path = out / f"{name}.joblib"
            joblib.dump(model, path)
            model = joblib.load(path)
            hashes[name] = sha(path)
            for evaluation, subset in [("Valid (Batch 1 Hold-out)", valid), ("Test (Batch 2)", test)]:
                records, m = _score(subset, model, cols)
                records["condition"], records["evaluation"] = name, evaluation
                pred_rows.append(records)
                eval_rows.append(dict(condition=name, title=title, evaluation=evaluation, **m,
                    mean_prediction_minus_actual=float(-records.residual.mean())))
        cv_rows.append(dict(condition=name, title=title, initial_penalty=penalty,
                            cv_mape_pct=cv, cv_std_pp=cv_sd))
        pipe = model.regressor_
        beta = pipe.named_steps["model"].coef_ * factors
        raw_beta = beta / pipe.named_steps["scaler"].scale_
        for col, b, raw in zip(cols, beta, raw_beta):
            coef_rows.append(dict(condition=name, feature=col, coefficient_standardized=float(b),
                                 coefficient_original_units=float(raw)))
    cv_frame, evaluation = pd.DataFrame(cv_rows), pd.DataFrame(eval_rows)
    prediction = pd.concat(pred_rows, ignore_index=True)
    assert np.isfinite(prediction.prediction).all() and prediction.prediction.gt(0).all()
    cv_frame.to_csv(out / "cv_comparison.csv", index=False)
    evaluation.to_csv(out / "evaluation_metrics.csv", index=False)
    prediction.to_csv(out / "evaluation_predictions.csv", index=False)
    pd.DataFrame(folds).to_csv(out / "fold_metrics.csv", index=False)
    pd.DataFrame(oofs).to_csv(out / "new_oof_predictions.csv", index=False)
    pd.DataFrame(coef_rows).to_csv(out / "coefficients.csv", index=False)
    prediction["life_band"] = pd.cut(prediction.cycle_life, [-np.inf, 500, 1000, np.inf], labels=["<=500", "500-1000", ">1000"])
    segments = prediction.groupby(["condition", "evaluation", "life_band"], observed=True).agg(
        n=("cell_id", "size"), mape_pct=("ape_pct", "mean"),
        mean_prediction_minus_actual=("residual", lambda a: -a.mean())).reset_index()
    segments.to_csv(out / "segments.csv", index=False)
    combined = cv_frame.copy()
    for short, key in [("holdout", "Valid (Batch 1 Hold-out)"), ("batch2", "Test (Batch 2)")]:
        combined = combined.merge(evaluation.loc[evaluation.evaluation.eq(key), ["condition", "mape_pct", "mean_prediction_minus_actual"]].rename(
            columns={"mape_pct":f"{short}_mape_pct", "mean_prediction_minus_actual":f"{short}_bias_cycles"}), on="condition", validate="one_to_one")
    combined.to_csv(out / "comparison.csv", index=False)
    complete_saved(root, out, parent)


def complete_saved(root, out, parent):
    """Finalize stored scores after an output error, without fitting/scoring again."""
    combined = pd.read_csv(out / "comparison.csv")
    segments = pd.read_csv(out / "segments.csv")
    coefficients = pd.read_csv(out / "coefficients.csv")
    assert list(combined.condition) == [c[0] for c in CONDITIONS]
    assert len(pd.read_csv(out / "evaluation_metrics.csv")) == 8
    assert len(pd.read_csv(out / "evaluation_predictions.csv")) == 4*(8+39)
    base = combined.iloc[0]
    matched = combined.loc[combined.cv_mape_pct.gt(base.cv_mape_pct)
                           & combined.batch2_mape_pct.lt(base.batch2_mape_pct), "condition"].tolist()
    hashes = {name: sha(parent / "selected_pipeline.joblib" if name == "baseline"
                       else out / f"{name}.joblib") for name, _, _ in CONDITIONS}
    clean = combined.astype(object).where(pd.notna(combined), None)
    write_json(out / "results.json", dict(
        comparison=clean.to_dict("records"), observed_cv_worse_batch2_better=matched,
        purpose="Diagnostic tradeoff, not selection from Batch2", additional_tuning=False,
        final_model_changed=False, seed=42, new_cv_fits=10, new_development_fits=3,
        raw_data_changed=False, batch3_evaluated=False,
        versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__),
        sha256=dict(code=sha(Path(__file__)), features=sha(parent / "cell_features.csv"),
                    manifest=sha(parent / "split_manifest.csv"), models=hashes)))
    plot(out, combined)
    write_report(root, combined, segments, coefficients)
    print(combined[["title", "cv_mape_pct", "holdout_mape_pct", "batch2_mape_pct", "batch2_bias_cycles"]].round(4).to_string(index=False))
    print("Outputs finalized from stored predictions; no repeated fitting/scoring.")


def plot(out, d):
    fig, ax = plt.subplots(figsize=(7.5, 3.6), layout="constrained")
    x = np.arange(4)
    for off, col, label in [(-.24,"cv_mape_pct","Batch1 CV"), (0,"holdout_mape_pct","Hold-out"), (.24,"batch2_mape_pct","Batch2 follow-up")]:
        bars = ax.bar(x+off, d[col], width=.24, label=label)
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
    ax.set(xticks=x, xticklabels=["Baseline", "Initial QD penalty10", "Initial QD penalty100", "Initial QD removed"], ylabel="MAPE (%)")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out / "comparison.png", dpi=160)
    plt.close(fig)


def write_report(root, comparison, segments, coefficients):
    report = """# DAY2 마지막 진단 — 초기 용량 의존 억제

## 가설과 근거

Batch1에는 500 미만 학습 셀이 없고 Batch2는 다수 단수명 셀을 포함한다. 초기 용량은 Batch1 내부 예측에 도움이 됐지만 배치별 분포가 달랐다. 따라서 **초기 용량에 대한 의존만 줄이면 내부 오차는 증가하더라도 외부 과대 예측이 감소할 수 있다**는 가설을 검토했다. 초기 총 용량과 수명의 약한 관계는 [원논문](https://www.nature.com/articles/s41560-019-0356-8)에서도 확인된 동기이며 현재 데이터의 인과를 확정하는 근거는 아니다.

## 사전에 고정한 비교

기존 로그 수명 Ridge(alpha=1)와 동일한 5개 특징을 유지하고, 초기 QD 중앙값 계수의 L2 벌점만 10배·100배로 증가시켰다. 다른 특징 벌점은 1, Target은 log10 수명, 개발28/Hold-out8/CV5/seed42를 유지했다. 각 폴드의 학습 셀에서만 결측 대체와 표준화를 fit했다.

표준화 후 초기 용량 열에 1/√벌점을 곱하고 Ridge를 학습했다. 표준화된 원 특징의 계수로 환산하면 `잔차제곱합 + 다른 계수제곱합 + 벌점 × 초기 용량 계수제곱`이다. 이 조정 후 다시 표준화하지 않아 벌점이 상쇄되지 않도록 했다. 같은 벌점을 모두에게 주는 전체 규제 강화와 구분한다. [Ridge 공식 문서](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)

새 규제 조건은 두 개뿐이며 추가 강도 탐색을 하지 않았다. 기존 모델 점수는 재사용했고, 완전 제거 조건의 CV도 이전 결과를 재사용했다. 완전 제거 조건의 Hold-out/Batch2는 이번에 학습용28개로만 학습한 비교 모델로 확인했다. 새 CV 학습10회와 학습용28개 최종fit3회를 수행했다. 저장 모델을 다시 불러와 평가했고, 특성별 규제는 대응하는 정상방정식과 계수가 일치하는지 확인했다.

## 결과

"""
    report += md_table(comparison, ["title", "cv_mape_pct", "holdout_mape_pct", "batch2_mape_pct", "batch2_bias_cycles"])
    report += "\n\nBatch2 bias는 예측 − 실제 수명의 평균이며 양수는 과대 예측이다. CV는 5개 검증 폴드 MAPE의 평균이다.\n\n"
    report += "![초기 용량 의존 억제 비교](capacity_penalty/comparison.png)\n\n"
    report += "### Batch2 수명 구간별 결과\n\n" + md_table(segments.loc[segments.evaluation.eq("Test (Batch 2)")], ["condition", "life_band", "n", "mape_pct", "mean_prediction_minus_actual"])
    report += "\n\n### 초기 용량 계수 변화\n\n" + md_table(coefficients.loc[coefficients.feature.eq(INITIAL)], ["condition", "coefficient_standardized", "coefficient_original_units"])
    report += "\n\n계수는 log10 수명을 예측하는 계수이며, 완전 제거 조건은 초기 용량 계수가 없다. 다른 특징 계수도 함께 재추정되므로 성능 변화 전체를 해당 계수 하나의 독립적 인과효과로 보지 않는다.\n\n"
    report += "## 가설과 실제 결과의 연결\n\n"
    report += "**이번 사전 고정 조건에서는 내부 정확도를 양보하고 외부 오차를 줄이는 패턴이 관측됐다.** 초기 용량 규제를 강화하면서 CV는 7.698% → 7.832% → 9.135%, Hold-out은 3.421% → 3.931% → 5.849%로 악화했지만 Batch2는 38.198% → 35.433% → 29.830%로 개선했다. 완전 제거 조건은 CV9.805%·Hold-out6.608%·Batch2 27.924%였다.\n\n"
    report += "초기 용량 표준화 계수는 0.0290 → 0.0218 → 0.0063으로 감소했고 Batch2의 평균 과대 예측은 190.73 → 173.41 → 138.43사이클로 줄었다. Batch1 내부에서 도움이 되는 초기 용량 관계에 덜 의존하면 현재 Batch2로 이전하는 오차를 줄일 수 있다는 가설과 부합한다. 특징 분포 차이·계수 억제·예측 편향 감소의 연결을 확인했지만, 초기 용량이 물리적으로 수명을 결정하는 기전이나 단수명 데이터 부재가 유일한 원인이라고 증명한 것은 아니다.\n\n"
    report += "Batch2의 500 이하 28개에서 평균 과대 예측은 179.43 → 169.38 → 148.90사이클, 완전 제거 시141.10사이클이었다. 단수명 예측의 편향은 여전히 크다. 완전 제거 조건도 기존 단일 분산 모델의 Batch2 26.186%보다 나빴고 목표9.1%에 도달하지 못했다. 따라서 성공적인 일반화 해결로 표현하지 않는다.\n\n"
    report += "## 해석의 범위\n\n이전 Batch2 진단을 본 후 세운 가설의 사전 고정 후속 비교다. 독립 최종 Test가 아니며 외부 최저 점수로 모델을 다시 선택하지 않는다. 단수명 학습 데이터가 없는 한계는 규제로 해결되지 않는다. 기존 CV 선택 MAPE 모델은 유지하고, 이 실험은 배치 의존을 줄이는 가설의 진단으로 기록한다. 추가 튜닝·Batch3 평가·전체36개 재학습은 하지 않았다. 결과가 개선돼도 감소 범위와 개선되지 않은 부분을 함께 보고한다.\n\n"
    report += "실행 코드: `src/day2_capacity_penalty.py`. 결과: `outputs/day2/capacity_penalty/`. 실행 명령: `.venv/bin/python -m src.day2_capacity_penalty`. 완료된 실행은 저장 결과를 사용하며 재학습하지 않는다.\n"
    (root / "outputs/day2/DAY2_CAPACITY_PENALTY.md").write_text(report)


if __name__ == "__main__":
    main()
