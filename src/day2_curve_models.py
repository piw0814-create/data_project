"""Preset log-IQR and low-component PLS comparison using the existing cell split.

Run once as `python -m src.day2_curve_models`. Raw data and prior runs are kept.
"""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import platform
import shutil

import h5py
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .day2_features import BATCH_FILES, _vector, _curve
from .day2_modeling import _metrics
from .day2_evaluation import _score
from .day2_experiments import write_json
from .day2_ablation import md_table

PAPER = "https://arxiv.org/abs/2101.01885"
CANDIDATES = [("linear_iqr", 0), ("pls_1", 1), ("pls_2", 2), ("pls_3", 3)]
TOLERANCE_PP = 0.1  # A preset practical tie; use fewer PLS components.


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_model(components):
    # StandardScaler and supervised PLS loadings are fitted inside each fold.
    estimator = (PLSRegression(n_components=components, scale=False,
                               max_iter=1000, tol=1e-6)
                 if components else LinearRegression())
    return Pipeline([("scaler", StandardScaler()), ("model", estimator)])


def read_curves(root, selected):
    rows, reference, indices = [], None, None
    for batch, group in selected.groupby("batch", sort=False):
        with h5py.File(root / "data" / BATCH_FILES[batch], "r") as file:
            bg = file["batch"]
            for record in group.itertuples():
                cell = int(record.cell_id)
                summary = file[bg["summary"][cell, 0]]
                cycle = _vector(summary["cycle"])
                positions = [np.flatnonzero(cycle == n) for n in (10, 100)]
                if any(len(p) != 1 for p in positions):
                    raise ValueError(f"Missing actual cycle 10/100: {batch}/{cell}")
                cg = file[bg["cycles"][cell, 0]]
                if cg["Qdlin"].size != len(cycle):
                    raise ValueError("Summary/curve positions do not match")
                q10, q100 = [_curve(file, cg, int(p[0])) for p in positions]
                voltage = _vector(file[bg["Vdlin"][cell, 0]])
                if not (q10.shape == q100.shape == voltage.shape):
                    raise ValueError("Mismatched voltage curves")
                delta = q100 - q10
                if not np.isfinite(delta).all() or not np.isfinite(voltage).all():
                    raise ValueError("Nonfinite curve")
                if reference is None:
                    reference = voltage.copy()
                    indices = np.linspace(0, len(delta)-1, 100).round().astype(int)
                    if len(np.unique(indices)) != 100:
                        raise ValueError("Curve has fewer than 100 distinct positions")
                if voltage.shape != reference.shape or not np.allclose(voltage, reference):
                    raise ValueError("Voltage grid mismatch")
                # Check against the same raw curves used by the frozen baseline.
                if not np.isclose(np.log10(np.var(delta)), record.log10_delta_Q_var,
                                  rtol=0, atol=1e-10):
                    raise ValueError("Re-extracted variance disagrees with baseline")
                iqr = float(np.percentile(delta, 75)-np.percentile(delta, 25))
                if iqr <= 0:
                    raise ValueError("Nonpositive IQR")
                row = dict(batch=batch, cell_id=str(cell), delta_Q_iqr=iqr,
                           log10_delta_Q_iqr=float(np.log10(iqr)))
                row.update({f"dq_{i:03d}": float(v) for i, v in enumerate(delta[indices])})
                rows.append(row)
    grid = pd.DataFrame(dict(feature=[f"dq_{i:03d}" for i in range(100)],
                             original_position=indices, voltage=reference[indices]))
    return pd.DataFrame(rows), grid, len(reference)


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / "outputs/day2/curve_models"
    if (out / "results.json").exists():
        print("Completed run exists; using saved results, no repeated fits/evaluations.")
        print(pd.read_csv(out / "comparison.csv").to_string(index=False))
        return
    if out.exists() and any(out.iterdir()):
        raise RuntimeError("Partial run exists: inspect saved state before running again")
    out.mkdir(parents=True, exist_ok=True)
    prior = root / "outputs/day2/feature_experiments"
    baseline = root / "outputs/day2/modeling_cell_split"
    write_json(out / "plan.json", dict(
        frozen_utc=datetime.now(timezone.utc).isoformat(), seed=42,
        hypothesis="Represent early discharge-curve dispersion/shape without absolute initial capacity",
        candidates=[dict(name=n, components=c) for n, c in CANDIDATES],
        target="Provided cycle_life, raw cycles; same as original linear variance baseline",
        input="Q100(V)-Q10(V); IQR on all voltage points; PLS on 100 fixed equally spaced positions",
        fit_scope="Batch1 development28 only; existing holdout8 and same CV5",
        selection="Fold-mean MAPE; PLS smallest component count within 0.1pp of its CV minimum",
        pls_practical_tie_pp=TOLERANCE_PP,
        evaluation="One fixed IQR and one CV-selected PLS representative; reuse historical baseline scores",
        batch2_used_for_selection=False, batch2_context="Already examined; exploratory follow-up",
        paper=PAPER, replicate_paper=False, batch3_evaluated=False))
    original = pd.read_csv(prior / "cell_features.csv", dtype={"cell_id": str})
    manifest = pd.read_csv(prior / "split_manifest.csv", dtype={"cell_id": str})
    data = manifest[["batch", "cell_id", "role", "cv_fold"]].merge(
        original, on=["batch", "cell_id"], validate="one_to_one", sort=False)
    use = (data.batch.eq("Batch1") & data.role.isin(["development", "holdout"])) | (
        data.batch.eq("Batch2") & data.label_eligible & data.delta_status.eq("ok")
        & data.cycle_life.gt(100))
    data = data.loc[use].copy()
    curves, grid, full_grid_count = read_curves(root, data)
    data = data.merge(curves, on=["batch", "cell_id"], validate="one_to_one", sort=False)
    data.to_csv(out / "cell_features.csv", index=False)
    grid.to_csv(out / "voltage_grid.csv", index=False)
    dev = data.loc[data.role.eq("development")].reset_index(drop=True)
    valid = data.loc[data.batch.eq("Batch1") & data.role.eq("holdout")]
    test = data.loc[data.batch.eq("Batch2")]
    assert (len(dev), len(valid), len(test)) == (28, 8, 39)
    assert dev.batch.eq("Batch1").all() and dev.input_max_cycle.le(100).all()
    assert set(dev.cell_id).isdisjoint(valid.cell_id)
    columns = dict(iqr=["log10_delta_Q_iqr"], pls=grid.feature.tolist())
    y, fold_ids = dev.cycle_life.to_numpy(float), dev.cv_fold.to_numpy(int)
    folds, summaries, oof = [], [], []
    for name, components in CANDIDATES:
        cols = columns["pls" if components else "iqr"]
        x = dev[cols]
        assert np.isfinite(x.to_numpy()).all()
        predictions, values = np.full(len(y), np.nan), []
        for fold in range(1, 6):
            tr, va = np.flatnonzero(fold_ids != fold), np.flatnonzero(fold_ids == fold)
            model = make_model(components)
            model.fit(x.iloc[tr], y[tr])
            prediction = model.predict(x.iloc[va]).reshape(-1)
            predictions[va] = prediction
            metrics = _metrics(y[va], prediction)
            values.append(metrics["mape_pct"])
            folds.append(dict(candidate=name, fold=fold, n_train=len(tr), n_valid=len(va), **metrics))
        assert np.isfinite(predictions).all()
        summaries.append(dict(candidate=name, components=components, input_features=len(cols),
                              cv_mape_pct=float(np.mean(values)), cv_std_pp=float(np.std(values, ddof=1))))
        for i, p in enumerate(predictions):
            oof.append(dict(candidate=name, batch="Batch1", cell_id=dev.iloc[i].cell_id,
                            fold=int(fold_ids[i]), actual=float(y[i]), prediction=float(p)))
    cv = pd.DataFrame(summaries)
    cv.to_csv(out / "cv_comparison.csv", index=False)
    pd.DataFrame(folds).to_csv(out / "fold_metrics.csv", index=False)
    pd.DataFrame(oof).to_csv(out / "oof_predictions.csv", index=False)
    pls_rows = cv.loc[cv.components.gt(0)]
    best_pls = pls_rows.loc[pls_rows.cv_mape_pct.le(pls_rows.cv_mape_pct.min()+TOLERANCE_PP)]
    pls_name = best_pls.sort_values("components").iloc[0].candidate
    representatives = ["linear_iqr", str(pls_name)]
    old = json.loads((baseline / "final_evaluation.json").read_text())
    baseline_cv = float(old["train_cv_mean_mape_pct"])
    # Freeze representatives and the recommended comparison model before holdout/B2 scores.
    scores_for_selection = {"linear_variance": baseline_cv, **{
        name: float(cv.set_index("candidate").loc[name, "cv_mape_pct"]) for name in representatives}}
    selected = min(scores_for_selection, key=scores_for_selection.get)
    write_json(out / "selection.json", dict(
        frozen_utc=datetime.now(timezone.utc).isoformat(), representatives=representatives,
        comparison_selected=selected, comparison_scope="Existing variance and two new representatives",
        cv_scores=scores_for_selection, holdout_used=False, batch2_used=False,
        overall_previous_final_candidate_automatically_replaced=False))
    evaluation, records, model_hashes, coefficients = [], [], {}, []
    for name in representatives:
        components = int(cv.set_index("candidate").loc[name, "components"])
        cols = columns["pls" if components else "iqr"]
        model = make_model(components).fit(dev[cols], y)
        path = out / f"{name}.joblib"
        joblib.dump(model, path)
        saved = joblib.load(path)
        assert np.allclose(saved.predict(dev[cols]), model.predict(dev[cols]))
        model_hashes[name] = sha(path)
        beta = np.asarray(model.named_steps["model"].coef_).reshape(-1)
        for col, b, scale in zip(cols, beta, model.named_steps["scaler"].scale_):
            coefficients.append(dict(candidate=name, feature=col, coefficient_standardized=float(b),
                                     coefficient_original_units=float(b/scale)))
        for label, subset in [("Valid (Batch 1 Hold-out)", valid), ("Test (Batch 2)", test)]:
            prediction, metrics = _score(subset, saved, cols)
            prediction["candidate"], prediction["evaluation"] = name, label
            records.append(prediction)
            evaluation.append(dict(candidate=name, evaluation=label, **metrics,
                                   bias_cycles=float(-prediction.residual.mean())))
    historic = pd.read_csv(baseline / "evaluation_predictions.csv", dtype={"cell_id": str})
    historic["candidate"] = "linear_variance"
    records.append(historic)
    for label, key in [("Valid (Batch 1 Hold-out)", "valid"), ("Test (Batch 2)", "test_batch2")]:
        subset = historic.loc[historic.evaluation.eq(label)]
        evaluation.append(dict(candidate="linear_variance", evaluation=label, **old[key],
                               bias_cycles=float(-subset.residual.mean())))
    metrics, predictions = pd.DataFrame(evaluation), pd.concat(records, ignore_index=True)
    assert np.isfinite(predictions.prediction).all()
    metrics.to_csv(out / "evaluation_metrics.csv", index=False)
    predictions.to_csv(out / "evaluation_predictions.csv", index=False)
    pd.DataFrame(coefficients).to_csv(out / "coefficients.csv", index=False)
    predictions["life_band"] = pd.cut(predictions.cycle_life, [-np.inf,500,1000,np.inf],
                                      labels=["<=500", "500-1000", ">1000"])
    segments = predictions.groupby(["candidate", "evaluation", "life_band"], observed=True).agg(
        n=("cell_id", "size"), mape_pct=("ape_pct", "mean"),
        bias_cycles=("residual", lambda a: -a.mean())).reset_index()
    segments.to_csv(out / "segments.csv", index=False)
    comparison = pd.DataFrame([
        dict(candidate="linear_variance", cv_mape_pct=baseline_cv),
        *cv.loc[cv.candidate.isin(representatives), ["candidate", "cv_mape_pct"]].to_dict("records")])
    for short, label in [("holdout", "Valid (Batch 1 Hold-out)"), ("batch2", "Test (Batch 2)")]:
        comparison = comparison.merge(metrics.loc[metrics.evaluation.eq(label), ["candidate", "mape_pct", "bias_cycles"]].rename(
            columns={"mape_pct":f"{short}_mape_pct", "bias_cycles":f"{short}_bias_cycles"}), on="candidate", validate="one_to_one")
    comparison.to_csv(out / "comparison.csv", index=False)
    chosen = comparison.set_index("candidate").loc[selected]
    reporting = pd.DataFrame([
        ["Train (Batch 1 CV)", chosen.cv_mape_pct, "5-fold mean; development28"],
        ["Valid (Batch 1 Hold-out)", chosen.holdout_mape_pct, "Same model; 8 cells"],
        ["Test (Batch 2)", chosen.batch2_mape_pct, "Exploratory follow-up; 39 cells"],
        ["Gap (Train-Valid)", chosen.holdout_mape_pct-chosen.cv_mape_pct, "Valid - CV, percentage points"],
        ["Gap (Valid-Test)", chosen.batch2_mape_pct-chosen.holdout_mape_pct, "Batch2 - Valid, percentage points"],
        ["Gap (Target-Test)", chosen.batch2_mape_pct-9.1, "Batch2 - 9.1%, assignment reference"]],
        columns=["구분", "MAPE (%)", "비고"])
    reporting.to_csv(out / "performance_reporting.csv", index=False)
    versions = dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                    sklearn=sklearn.__version__, h5py=h5py.__version__, joblib=joblib.__version__)
    write_json(out / "results.json", dict(
        comparison=comparison.to_dict("records"), selected=selected, pls_selected=pls_name,
        new_cv_fits=20, new_development_fits=2, target_transform="none", versions=versions,
        full_voltage_points=full_grid_count, raw_data_changed=False, batch3_read=False,
        negative_prediction_counts=predictions.groupby("candidate").negative_prediction.sum().astype(int).to_dict(),
        prior_final_candidate_changed=False, selection_before_external_scoring=True,
        source_files={b:dict(filename=BATCH_FILES[b], size=(root/"data"/BATCH_FILES[b]).stat().st_size)
                      for b in ("Batch1", "Batch2")},
        hashes=dict(code=sha(Path(__file__)), prior_features=sha(prior/"cell_features.csv"),
                    manifest=sha(prior/"split_manifest.csv"), model_hashes=model_hashes)))
    plot(out, comparison)
    write_report(root, cv, comparison, reporting, segments, selected, pls_name)
    print(comparison.round(4).to_string(index=False))
    print("Selected by Batch1 CV before evaluation:", selected, "; PLS representative:", pls_name)


def plot(out, comparison):
    fig, ax = plt.subplots(figsize=(7.5, 4), layout="constrained")
    x = np.arange(len(comparison))
    for offset, col, label in [(-.24,"cv_mape_pct","Batch1 CV"),
                               (0,"holdout_mape_pct","Hold-out"),
                               (.24,"batch2_mape_pct","Batch2 follow-up")]:
        bars = ax.bar(x+offset, comparison[col], width=.24, label=label)
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=3)
    ax.set(xticks=x, xticklabels=comparison.candidate.tolist(), ylabel="MAPE (%)")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top","right"]].set_visible(False)
    fig.savefig(out / "comparison.png", dpi=160)
    plt.close(fig)


def write_report(root, cv, comparison, reporting, segments, selected, pls_name):
    baseline = comparison.iloc[0]
    iqr = comparison.set_index("candidate").loc["linear_iqr"]
    pls = comparison.set_index("candidate").loc[pls_name]
    report = """# DAY2 — ΔQ 특징 표현 변경: 로그 IQR과 PLS

## 1. 가설과 논문 근거

초기 용량 등 추가 특징은 Batch1 CV를 개선했지만 Batch2에서는 기존 단일 분산 모델보다 나빴다. 따라서 다른 센서 신호를 추가하지 않고, 이미 배치마다 일관된 수명 관계가 확인된 ΔQ100−10(V)의 표현을 바꾼다. 로그 IQR은 극단적인 전압 지점의 영향에 덜 민감한 퍼짐, PLS는 분산 하나로 사라지는 전압별 곡선 형태를 활용한다.

[Attia·Severson·Witmer (2021)](https://arxiv.org/abs/2101.01885)은 동일 계열 데이터에서 로그 IQR과 ΔQ 곡선 PLS를 비교했다. 논문 Table II의 1차/2차 테스트 RMSE는 분산138/196, IQR124/190, PLS100/176사이클이다. 논문의 혼합 배치 학습·분할과 로그 수명 타깃은 현재와 다르며 RMSE를 MAPE로 해석하지 않는다. 원문 방법의 정확한 재현이 아니라 표현 방식을 참고한 제한된 비교다.

## 2. 이번에 변경한 부분과 유지한 부분

- 로그 IQR = log10(ΔQ 값의 75백분위수 − 25백분위수). 전체1000개 전압 지점으로 계산하고 특징1개 선형회귀에 사용했다. 분산과 함께 넣지 않았다.
- PLS는 같은 ΔQ 곡선에서 고정된 등간격100개 위치를 뽑았다. 전압 해상도100개와 성분 수를 구분한다. 성분1·2·3개만 비교했다. 성분이 적어도 전압별 가중치는 학습되므로 자유도가 단순히1~3개뿐이라고 주장하지 않는다.
- 수명은 로그 변환하지 않았다. 기존 분산 선형 모델과 동일하게 원 cycle_life를 예측해 특징 표현의 효과를 비교했다. 초기 용량·충전 시간·온도·IR·후기 열화값은 넣지 않았다.
- 기존 Batch1 개발28/Hold-out8, 동일5fold 및 seed42, 기존 레이블 제외 기준을 유지했다. 모든 표준화와 PLS의 지도학습 압축은 각 학습 폴드 안에서만 fit했다.
- PLS CV 최저와0.1%p 이내면 성분이 적은 설정을 고르는 규칙을 평가 전에 고정했다. IQR과 PLS 대표를 먼저 고정하고 Hold-out/Batch2를 각각 평가했다. 기존 분산은 저장 점수를 재사용했다.
- 새 학습은 CV20회와 개발28개 fit2회이다. 전체36개 재학습, Batch3 읽기·평가, Batch2 결과를 보고 추가 설정 탐색은 하지 않았다.

## 3. Batch1 CV에서 성분 선택

"""
    report += md_table(cv, ["candidate", "components", "input_features", "cv_mape_pct", "cv_std_pp"])
    report += f"\n\nPLS 대표는 `{pls_name}`이며, 기존 분산과 두 대표의 이번 비교에서 CV로 선택된 모델은 `{selected}`이다. 이전 전체 개발의 최종 후보를 자동으로 교체하지 않았다. CV 표준편차는5개 폴드 간 변동이며 신뢰구간이 아니다.\n\n## 4. 고정 대표의 평가 결과\n\n"
    report += md_table(comparison, ["candidate", "cv_mape_pct", "holdout_mape_pct", "batch2_mape_pct", "batch2_bias_cycles"])
    report += "\n\n![ΔQ 표현 비교](curve_models/comparison.png)\n\nBias는 예측 − 실제 수명의 평균이다. 양수는 과대 예측이다.\n\n"
    report += f"IQR의 Batch2 변화는 기존 대비 {iqr.batch2_mape_pct-baseline.batch2_mape_pct:+.3f}%p, PLS 대표는 {pls.batch2_mape_pct-baseline.batch2_mape_pct:+.3f}%p였다. 음수는 개선이다.\n\n"
    report += "## 5. 과제 성능 표 — 이번 비교의 CV 선택 모델\n\n" + md_table(reporting, list(reporting.columns))
    report += "\n\n## 6. Batch2 수명 구간별 해석 자료\n\n" + md_table(
        segments.loc[segments.evaluation.eq("Test (Batch 2)")], ["candidate", "life_band", "n", "mape_pct", "bias_cycles"])
    report += "\n\n### 관측 결과와 가설의 연결\n\n"
    report += f"IQR은 CV {baseline.cv_mape_pct:.3f}% → {iqr.cv_mape_pct:.3f}%와 Hold-out {baseline.holdout_mape_pct:.3f}% → {iqr.holdout_mape_pct:.3f}%로 내부 점수가 좋아졌다. 하지만 Batch2는 {iqr.batch2_mape_pct:.3f}%로 악화했다. CV 차이는 작으며 유의한 개선이라고 단정하지 않는다. 극단값에 덜 민감한 특징이라는 동기가 현재 배치 일반화 개선으로 이어지지는 않았다.\n\n"
    report += f"PLS는 성분1/2/3개의 CV가 각각10.283/12.031/10.584%로 성분을 늘리는 이점이 확인되지 않았다. 1성분 대표는 Batch2 평균 과대 예측을 {baseline.batch2_bias_cycles:.2f} → {pls.batch2_bias_cycles:.2f}사이클로 줄였지만 MAPE는 {pls.batch2_mape_pct:.3f}%였다. 평균 편향 감소와 개별 셀 상대 오차 개선은 같지 않다. 500이하 구간 오차는 분산29.389%, PLS33.559%였고, 500~1000 구간은22.484% →20.141%로 좋아졌지만 전체 개선에는 부족했다.\n\n"
    report += "따라서 이번 두 설계가 기존26.186%보다 낫다는 가설은 지지되지 않았다. 같은 계열 데이터에서 검증된 문헌 근거가 있어도 현재의 Batch1만 학습하는 조건에 그대로 성능이 이전되지 않았다. 원인은 학습 범위·배치 조건·표현/타깃 차이 등 여러 가능성이 있어 단독 원인을 확정하지 않는다. 추가 성분의 Batch2 점수를 보고 고르거나 이번 결과를 본 뒤 타깃 변환을 추가하는 작업은 진행하지 않았다.\n\n"
    report += "\n\n## 7. 해석의 한계\n\n이번 비교는 표현 변경이 현재 배치 이전에 도움이 되는지를 보는 후속 탐색이다. Batch2는 이전에도 확인했으므로 완전히 새로운 독립 최종 Test 성능으로 제시하지 않는다. 내부와 외부 개선을 따로 해석하고, 외부 점수가 가장 낮다는 이유로 후보를 사후 선택하지 않는다. 단수명 학습 셀이 부족한 문제나 배치 시작 조건 차이가 특징 변경만으로 해결됐다고 단정하지 않는다. 기존 레이블 종료 경계 및 원논문과의 파일·분할 차이도 유지된다.\n\n실행: `.venv/bin/python -m src.day2_curve_models`. 완료 후 재실행은 저장 결과를 보여주며 학습·평가를 반복하지 않는다. 모델/버전/분할/전압 위치/폴드 점수/예측은 `outputs/day2/curve_models/`에 저장했다.\n"
    (root / "outputs/day2/DAY2_CURVE_MODELS.md").write_text(report)


if __name__ == "__main__":
    main()
