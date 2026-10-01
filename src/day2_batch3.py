"""One additional Batch3 test of two previously frozen Batch1 models.

Run: python -m src.day2_batch3. No fitting, tuning, clipping or test-based removal.
Completed runs reuse saved scores; partial runs are never silently overwritten.
"""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import platform
import subprocess

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn

from .day2_evaluation import _score
from .day2_experiments import extract_changes, write_json
from .day2_features import BATCH_FILES, extract_cell_features
from .day2_process_review import configure_font, table

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/day2"
PARENT = BASE / "protocol_validation"
OUT = BASE / "batch3_test"
SOURCE = BASE / "feature_experiments/cell_features.csv"
LABELS = {"S3_elasticnet_0.001_0.5": "CV 선택 ElasticNet", "S0_linear": "분산 1개 선형회귀"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_frozen(plan):
    for name, expected in plan["input_hashes"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Frozen input changed: {name}")
    for name, expected in plan["model_hashes"].items():
        if sha(PARENT / f"{name}.joblib") != expected:
            raise ValueError(f"Frozen model changed: {name}")


def audit_inputs(frame, columns):
    """Recompute deterministic early features from read-only MATs, without fitting.

    The existing extraction checks actual cycle numbers, equal finite voltage
    grids and finite cycle10/100 curves. Per-cell subtraction is kept unchanged.
    Out-of-training-range values are reported, never removed as outliers.
    """
    raw = extract_cell_features(ROOT / "data")
    changes = extract_changes(ROOT / "data")
    raw["cell_id"] = raw.cell_id.astype(str)
    raw = raw.merge(changes, on=["batch", "cell_id"], validate="one_to_one")
    cached = frame.loc[frame.batch.eq("Batch3")].sort_values("cell_id").reset_index(drop=True)
    fresh = raw.loc[raw.batch.eq("Batch3")].sort_values("cell_id").reset_index(drop=True)
    if len(cached) != 46 or not cached.cell_id.equals(fresh.cell_id):
        raise ValueError("Unexpected Batch3 cell identities")
    check_columns = ["cycle_life", "input_max_cycle", "change_feature_max_cycle"] + columns
    np.testing.assert_allclose(cached[check_columns], fresh[check_columns],
                               rtol=1e-10, atol=1e-12, equal_nan=True)
    rows = cached[["batch", "cell_id", "cycle_life", "label_eligible", "label_status",
                   "input_max_cycle", "change_feature_max_cycle", "delta_status"]].copy()
    rows["raw_features_match_cache"] = True
    rows["finite_model_inputs"] = np.isfinite(cached[columns]).all(axis=1)
    rows["input_horizon_ok"] = (cached.input_max_cycle.le(100)
                               & cached.change_feature_max_cycle.le(100))
    rows["included"] = (cached.label_eligible & cached.cycle_life.gt(100)
                        & cached.delta_status.eq("ok") & rows.input_horizon_ok)
    rows["exclusion_reason"] = np.where(rows.included, "included", cached.label_status)
    if not rows.loc[rows.included, "finite_model_inputs"].all():
        raise ValueError("Nonfinite model inputs; inspect before evaluation")
    rows.to_csv(OUT / "input_audit.csv", index=False)
    cached.loc[rows.included].to_csv(OUT / "cell_features.csv", index=False)
    return cached.loc[rows.included].copy(), rows


def main():
    if (OUT / "results.json").is_file():
        result = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
        validate_frozen(result["plan"])
        print("Saved Batch3 test exists; no repeated training or scoring.")
        print(pd.read_csv(OUT / "evaluation_metrics.csv").round(3).to_string(index=False))
        render_saved()
        return
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError("Partial Batch3 test exists; inspect it before restarting")
    selection = json.loads((PARENT / "selection.json").read_text(encoding="utf-8"))
    candidates = [selection["primary"], "S0_linear"]
    if candidates != list(LABELS):
        raise ValueError("Unexpected frozen primary; do not choose models from Batch3")
    inputs = [SOURCE, BASE / "label_audit.csv", PARENT / "selection.json",
              PARENT / "split_manifest.csv", PARENT / "model_comparison.csv",
              PARENT / "evaluation_metrics.csv", PARENT / "evaluation_predictions.csv"]
    raw_path = ROOT / "data" / BATCH_FILES["Batch3"]
    info = raw_path.stat()
    plan = dict(
        created_utc=datetime.now(timezone.utc).isoformat(), seed=42,
        primary=selection["primary"], candidates=candidates,
        candidate_choice="Existing protocol-CV winner plus preset one-feature control; fixed before scoring",
        specs={n: selection["specs"][n] for n in candidates},
        model_hashes={n: selection["model_hashes"][n] for n in candidates},
        input_hashes={p.relative_to(ROOT).as_posix(): sha(p) for p in inputs},
        input_cutoff_cycle=100, fit_cells=29, holdout_cells=7,
        exclusion_rule="Existing valid provided labels, cycle_life>100 and valid early curves; no error-based deletion",
        raw_source=dict(file=raw_path.name, size_bytes=info.st_size, mtime_ns=info.st_mtime_ns),
        raw_curve_policy="Same finite voltage grid; within-cell Q100(V)-Q10(V); no Batch3-derived alignment or scaling",
        target="Provided total cycle_life; exact EOL crossing is not redefined",
        batch3_prior_use="Previously inspected in EDA and feature distributions; first model scoring, not completely unseen data",
        model_fit=False, model_selection_changed=False, refit_including_holdout=False,
        versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                      sklearn=sklearn.__version__, joblib=joblib.__version__),
        code_sha256=sha(Path(__file__)),
        git_commit=(subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                   capture_output=True).stdout.strip() or None)
                    if (ROOT / ".git").exists() else None)
    validate_frozen(plan)
    OUT.mkdir(parents=True)
    write_json(OUT / "plan.json", plan)  # Written before seeing any Batch3 predictions.
    frame = pd.read_csv(SOURCE, dtype={"cell_id": str})
    columns = list(dict.fromkeys(c for n in candidates for c in plan["specs"][n]["features"]))
    test, audit = audit_inputs(frame, columns)
    manifest = pd.read_csv(PARENT / "split_manifest.csv", dtype={"cell_id": str})
    dev = manifest.loc[manifest.role.eq("development"), ["batch", "cell_id"]].merge(
        frame, on=["batch", "cell_id"], validate="one_to_one")
    if len(dev) != 29 or len(test) != 44:
        raise ValueError("Unexpected development/test count; inspect the audit")
    ranges = []
    for c in columns + ["cycle_life"]:
        ranges.append(dict(feature=c, development_min=dev[c].min(), development_max=dev[c].max(),
                           batch3_min=test[c].min(), batch3_max=test[c].max(),
                           below_development_n=int(test[c].lt(dev[c].min()).sum()),
                           above_development_n=int(test[c].gt(dev[c].max()).sum())))
    pd.DataFrame(ranges).to_csv(OUT / "range_diagnostics.csv", index=False)
    prior_metrics = pd.read_csv(PARENT / "evaluation_metrics.csv")
    prior_predictions = pd.read_csv(PARENT / "evaluation_predictions.csv", dtype={"cell_id": str})
    scores, predictions, segments = [], [], []
    maximum = float(dev.cycle_life.max())
    for candidate in candidates:
        model = joblib.load(PARENT / f"{candidate}.joblib")
        pred, metrics = _score(test, model, plan["specs"][candidate]["features"])
        pred["candidate"], pred["evaluation"] = candidate, "Test (Batch 3)"
        pred["above_training_life_max"] = pred.cycle_life.gt(maximum)
        predictions.append(pred)
        prior = prior_predictions.loc[prior_predictions.candidate.eq(candidate)]
        for evaluation, g in pd.concat([prior, pred]).groupby("evaluation"):
            if evaluation == "Test (Batch 3)":
                row = dict(candidate=candidate, evaluation=evaluation, **metrics)
            else:
                row = prior_metrics.loc[prior_metrics.candidate.eq(candidate)
                                        & prior_metrics.evaluation.eq(evaluation)].iloc[0].to_dict()
            row.update(bias_cycles=float((g.prediction - g.cycle_life).mean()),
                       underprediction_n=int(g.prediction.lt(g.cycle_life).sum()))
            scores.append(row)
        for label, g in pred.groupby("above_training_life_max"):
            segments.append(dict(candidate=candidate,
                segment="above training maximum" if label else "within training life range",
                n=len(g), mape_pct=g.ape_pct.mean(), mae=abs(g.residual).mean(),
                bias_cycles=-g.residual.mean(), underprediction_n=int(g.residual.gt(0).sum())))
    validate_frozen(plan)
    pd.concat(predictions, ignore_index=True).to_csv(OUT / "evaluation_predictions.csv", index=False)
    scores = pd.DataFrame(scores)
    scores.to_csv(OUT / "evaluation_metrics.csv", index=False)
    pd.DataFrame(segments).to_csv(OUT / "life_segments.csv", index=False)
    # Only Batch3 rows are new; copy existing Batch1 and Batch2 scores verbatim.
    cv = pd.read_csv(PARENT / "model_comparison.csv")
    primary = plan["primary"]
    train = float(cv.loc[cv.candidate.eq(primary) & cv.method.eq("protocol"), "cv_mape_pct"].iloc[0])
    values = scores.loc[scores.candidate.eq(primary)].set_index("evaluation").mape_pct
    valid, b2, b3 = [float(values[x]) for x in
                    ["Valid (Batch 1 Hold-out)", "Test (Batch 2)", "Test (Batch 3)"]]
    report = pd.DataFrame([
        ["Train (Batch 1 CV)", "", train, "기존 프로토콜 CV 평균; 재학습하지 않음"],
        ["Valid (Batch 1 Hold-out)", "", valid, "기존 7셀·4프로토콜"],
        ["Test (Batch 2)", "", b2, "기존 39셀 평가; 재계산하지 않음"],
        ["", "Gap (Train-Valid)", valid-train, "Valid − CV; %p"],
        ["", "Gap (Valid-Test)", b2-valid, "Batch2 − Valid; %p"],
        ["", "Gap (Target-Test)", b2-9.1, "Batch2 − 참고값 9.1; %p"],
        ["Test (Batch 3)", "", b3, "44셀; 같은 고정 모델의 추가 평가"],
        ["", "Gap (Batch2-Batch3)", b3-b2, "Batch3 − Batch2; (+)이면 Batch3 오차 증가; %p"],
        ["", "Gap (Target-Test)", b3-9.1, "Batch3 − 과제 공통 참고값 9.1; 논문의 Batch3 목표치 아님"]],
        columns=["구분", "비교", "MAPE (%)", "비고"])
    report.to_csv(OUT / "performance_reporting.csv", index=False)
    result = dict(plan=plan, completed_utc=datetime.now(timezone.utc).isoformat(),
                  n_raw=46, n_test=len(test), excluded_cell_ids=audit.loc[~audit.included, "cell_id"].tolist(),
                  new_predictions=len(test)*len(candidates), new_model_fits=0,
                  primary_batch3_mape_pct=b3, primary_batch2_to_batch3_gap_pp=b3-b2,
                  model_hashes_after={n: sha(PARENT / f"{n}.joblib") for n in candidates})
    write_json(OUT / "results.json", result)
    render_saved()
    print(scores.round(3).to_string(index=False))


def render_saved():
    result = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    scores = pd.read_csv(OUT / "evaluation_metrics.csv")
    pred = pd.read_csv(OUT / "evaluation_predictions.csv")
    segments = pd.read_csv(OUT / "life_segments.csv")
    primary = result["plan"]["primary"]
    b3 = scores.loc[scores.evaluation.eq("Test (Batch 3)")].copy()
    comparison = scores.loc[scores.evaluation.isin(["Test (Batch 2)", "Test (Batch 3)"])].copy()
    comparison["모델"] = comparison.candidate.map(LABELS)
    comparison = comparison.rename(columns={"evaluation": "평가", "n": "셀 수", "mape_pct": "MAPE (%)",
                                             "mae": "MAE (사이클)", "rmse": "RMSE (사이클)",
                                             "r2": "R²", "bias_cycles": "평균 예측−실제"})
    configure_font()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    for ax, candidate in zip(axes, LABELS):
        g = pred.loc[pred.candidate.eq(candidate)]
        ax.scatter(g.cycle_life, g.prediction, alpha=.8, s=35)
        ax.plot([350, 2000], [350, 2000], "--", color="gray", label="예측 = 실제")
        ax.axvline(1054, color="gray", alpha=.45, label="학습 최대 수명")
        ax.set(xlim=(350, 2000), ylim=(350, 2000), xlabel="실제 수명 (사이클)",
               ylabel="예측 수명 (사이클)", title=LABELS[candidate])
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.savefig(OUT / "predictions.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
    for ax, metric, title in [(axes[0], "mape_pct", "MAPE (%)"), (axes[1], "mae", "MAE (사이클)")]:
        for offset, evaluation, label in [(-.18, "Test (Batch 2)", "Batch2"), (.18, "Test (Batch 3)", "Batch3")]:
            vals = [scores.loc[scores.candidate.eq(n) & scores.evaluation.eq(evaluation), metric].iloc[0]
                    for n in LABELS]
            bars = ax.bar(np.arange(2)+offset, vals, width=.35, label=label)
            ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
        ax.set(xticks=np.arange(2), xticklabels=list(LABELS.values()), ylabel=title)
        ax.set_ylim(0, ax.get_ylim()[1]*1.16)
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUT / "batch_comparison.png", dpi=160)
    plt.close(fig)
    segment_table = segments.copy()
    segment_table["candidate"] = segment_table.candidate.map(LABELS)
    segment_table["segment"] = segment_table.segment.map({"above training maximum": "학습 최대 수명 초과 (>1054)",
                                                          "within training life range": "학습 수명 범위 내 (≤1054)"})
    segment_table = segment_table.rename(columns={"candidate": "모델", "segment": "집단", "n": "셀 수",
        "mape_pct": "MAPE (%)", "mae": "MAE (사이클)", "bias_cycles": "평균 예측−실제",
        "underprediction_n": "과소예측 셀"})
    main_score = b3.loc[b3.candidate.eq(primary)].iloc[0]
    baseline_score = b3.loc[b3.candidate.eq("S0_linear")].iloc[0]
    report = f"""# DAY2 — Batch3 추가 테스트

**기존 프로토콜 CV 선택 ElasticNet의 Batch3 MAPE는 {main_score.mape_pct:.2f}%다.** Batch1 개발 29셀로 이미 학습한 모델을 그대로 적용했다. 같은 29셀로 학습한 분산 1개 선형회귀를 사전에 비교 대상으로 정했다. 이번 결과로 후보·특징·설정을 바꾸거나 재학습하지 않았다.

## 1. 무엇을 확인했는가

- Batch3 원본 46셀 중 수명 결측 2셀 제외, **44셀 평가**. 제외 cell_id: {result['excluded_cell_ids']} (0부터 시작).
- 원본 MAT에서 초기 특징을 다시 계산해 저장된 입력과 일치하는지 확인했다. 실제 10·100사이클, 유한한 Qdlin, 동일 전압 격자, 초기 100사이클 이내 조건을 확인했다.
- 모델 입력은 동일 셀의 `Q100(V)−Q10(V)`와 기존 용량 요약이다. Batch3에 맞춰 새 정렬·표준화·보정을 학습하지 않았다. 곡선 원값을 서로 다른 셀 사이에서 단순 비교하는 방식과 구분한다. 동일 격자 확인이 배치 차이를 모두 해결한다는 뜻은 아니다.
- 학습 범위 밖 값을 이상치로 삭제하거나 오차가 큰 셀을 사후 제외하지 않았다. 기존 유효 수명 44셀의 모델 입력은 모두 유한했다. 원논문에서 제외한 특정 셀을 동일하게 제외한 재현 실험이라고 주장하지 않는다.
- 제공 `cycle_life`를 그대로 평가했다. 엄격한 EOL 교차를 새로 확정한 수명과는 구분한다.
- 모델 해시·특징 CSV·분할표를 평가 전에 고정하고 평가 후 다시 확인했다. 새 학습 횟수는 **0**이다.

## 2. Batch2와 비교

![Batch2·3 MAPE와 사이클 단위 오차](batch3_test/batch_comparison.png)

{table(comparison[['모델', '평가', '셀 수', 'MAPE (%)', 'MAE (사이클)', 'RMSE (사이클)', 'R²', '평균 예측−실제']])}

평균 `예측−실제`가 음수이면 수명을 짧게 예측하는 방향이다.ElasticNet은 Batch2에서는 평균 약 178사이클 **과대예측**, Batch3에서는 평균 약 182사이클 **과소예측**했다. 서로 다른 배치에서 오차 방향도 달라졌다.

Batch3 MAPE는 낮아졌지만 ElasticNet MAE는 약 **198→195사이클**로 거의 같고, 분산 모델 MAE는 약 **143→150사이클**로 조금 커졌다. RMSE는 두 모델 모두 증가했다. 긴 수명에서는 같은 사이클 오차의 비율이 작아지므로, 상대오차 개선과 사이클 단위 예측 개선을 구분해야 한다. 이 결과는 오차 감소 전부가 수명 분모 때문이라는 증명도 아니다.

비교표의 분산 기준 모델은 프로토콜 분리 개발 **29셀** 모델이다. 과거 **28셀** 모델의 Batch2 26.19%와 혼동하지 않는다.

## 3. 장수명에서의 오차

![Batch3 실제 수명과 예측 수명](batch3_test/predictions.png)

{table(segment_table)}

점이 대각선 아래에 있으면 과소예측이다. Batch3에는 학습 최대 수명 1,054사이클보다 긴 셀이 17개 있어 외삽 성능을 따로 확인한다. 집단을 나눈 것은 오차 진단용이며 셀 제외나 모델 선택에 사용하지 않았다.

**학습 수명 범위 안의 27셀에서는 ElasticNet MAPE 7.57%, 분산 모델 8.71%로 추가 특징이 유용했다. 반면 범위 밖 17셀에서는 ElasticNet 27.06%, 분산 모델 17.46%로 순서가 뒤집혔다.** ElasticNet은 이 17셀 전부의 수명을 짧게 예측했고, 평균 약 395사이클 부족했다. 이번 전체 결과 차이는 긴 수명 구간의 오차와 연결된다.

Batch3 초기 용량 중앙값은 약 1.068 Ah로 학습 셀의 약 1.083 Ah보다 낮다. 현재 모델의 초기 용량 계수는 양수이므로 다른 입력이 같다면 낮은 용량이 예측 수명을 낮춘다. 하지만 실제 Batch3는 더 긴 수명 셀이 많았다. 초기 용량 관계의 배치 간 이동이 과소예측에 기여했을 가능성은 있으며, 여러 입력의 분포 차이가 함께 있으므로 원인을 초기 용량 하나로 확정하지 않는다.

## 4. 과제 지정 성능표 — CV 선택 모델

{table(pd.read_csv(OUT / 'performance_reporting.csv').fillna(''))}

Gap은 오차 증가가 양수가 되도록 계산하며 단위는 %p다. 9.1%는 과제의 공통 논문 참고값이다. Batch3의 별도 논문 성능 목표나 동일 데이터 분할 재현값으로 해석하지 않는다.

## 5. 결론과 평가의 범위

**Batch3에서도 단일 분산 모델의 전체 MAPE({baseline_score.mape_pct:.2f}%)가 CV 선택 모델({main_score.mape_pct:.2f}%)보다 낮았다.** 추가 특징은 학습 범위와 비슷한 수명에는 도움이 됐지만, 장수명 외삽에서 더 큰 오차를 보였다. Batch1 내부 CV의 우위가 모든 배치·수명 구간으로 이어지지 않음을 추가로 확인했다. 이 결과로 새 모델을 선정하거나 재튜닝하지 않는다.

이번 실행은 Batch3의 **첫 모델 예측·점수 계산**이다. 다만 Batch3는 이미 DAY1 EDA와 특징 분포 확인에 사용했으므로 완전히 보지 않은 데이터라고 표현하지 않는다. Batch2의 반복 개발 이력도 유지한다. 사전에 고정한 두 모델의 추가 평가이며 Batch3 성능으로 최종 모델을 다시 선택하지 않는다.

실행: `python -m src.day2_batch3`. 완료된 결과가 있으면 저장된 점수를 읽으며 재평가하지 않는다. [고정 계획](batch3_test/plan.json), [입력 점검표](batch3_test/input_audit.csv), [셀별 예측](batch3_test/evaluation_predictions.csv), [학습 범위 비교](batch3_test/range_diagnostics.csv)에 근거를 남겼다.
"""
    original_path = BASE / "batch3_original_test/results.json"
    if original_path.is_file():
        original = json.loads(original_path.read_text(encoding="utf-8"))
        score = original["batch3_metrics"]
        report += f"""
## 6. 기존 26.19% 모델의 추가 확인

첫 두 모델 평가 후 사용자의 요청으로, 기존 셀 분리 **28셀**로 학습해 Batch2 MAPE 26.19%를 기록한 저장 모델 `linear_F0`도 같은 Batch3 44셀에 적용했다. 모델·전처리를 그대로 사용했고 재학습하지 않았다. 이미 Batch3 결과를 본 뒤 추가한 비교이므로 최초 두 모델의 사전 고정 평가와 구분한다.

| 저장 모델 | 학습 셀 수 | Batch2 MAPE (%) | Batch3 MAPE (%) | Batch3 MAE | Batch3 RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| 기존 셀 분리 분산 모델 | 28 | {original['batch2_mape_pct']:.2f} | {score['mape_pct']:.2f} | {score['mae']:.2f} | {score['rmse']:.2f} |
| 프로토콜 분리 분산 모델 | 29 | 28.68 | {baseline_score.mape_pct:.2f} | {baseline_score.mae:.2f} | {baseline_score.rmse:.2f} |
| 프로토콜 CV 선택 ElasticNet | 29 | 37.14 | {main_score.mape_pct:.2f} | {main_score.mae:.2f} | {main_score.rmse:.2f} |

**기존 28셀 모델은 Batch3 MAPE {score['mape_pct']:.2f}%로, 29셀 분산 모델보다 약 {baseline_score.mape_pct-score['mape_pct']:.2f}%p 낮았다.** 두 단일 분산 모델의 결과는 가깝다. 학습 셀 구성이 다르므로 이 작은 차이를 프로토콜 분리의 인과적 효과나 통계적으로 확정된 성능 차이로 해석하지 않는다. 관측상 단순 모델의 외부 성능이 더 좋았다는 기존 해석과 연결되지만, 이 추가 결과로 CV 선택 모델을 바꾸지는 않는다.

[추가 평가 계획](batch3_original_test/plan.json) · [셀별 예측](batch3_original_test/evaluation_predictions.csv) · [평가표](batch3_original_test/evaluation_metrics.csv). 재확인은 `python -m src.day2_batch3_original`로 수행하며 완료된 점수를 재사용한다.
"""
    (BASE / "DAY2_BATCH3_TEST.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
