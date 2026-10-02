"""Summarize saved DAY2 experiments and draw figures; never fit or rescore."""
from pathlib import Path
import json
import hashlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


def table(df):
    values = df.copy()
    return "| " + " | ".join(values.columns) + " |\n| " + " | ".join(["---"]*len(values.columns)) + " |\n" + "\n".join(
        "| " + " | ".join(f"{v:.2f}" if isinstance(v, (float, np.floating)) else str(v)
                             for v in row) + " |" for row in values.itertuples(index=False,name=None))


def main():
    root = Path(__file__).resolve().parents[1]
    base = root / "outputs/day2"
    out = base / "process_review"
    out.mkdir(exist_ok=True)
    configure_font()
    stage = pd.read_csv(base / "feature_experiments/stage_comparison.csv")
    penalty = pd.read_csv(base / "capacity_penalty/comparison.csv")
    curve = pd.read_csv(base / "curve_models/comparison.csv")
    weighted = json.loads((base / "ablation_loss/results.json").read_text())
    ridge = penalty.iloc[0]
    wvalid, wtest = [next(row for row in weighted["evaluation"]
                         if row["candidate"]=="mape_raw_full_a0.01" and row["evaluation"]==label)
                     for label in ("Valid (Batch 1 Hold-out)", "Test (Batch 2)")]
    summary = pd.DataFrame([
        ["분산1개 선형회귀",1,curve.iloc[0].cv_mape_pct,curve.iloc[0].holdout_mape_pct,curve.iloc[0].batch2_mape_pct],
        ["5개 특징 로그 Ridge",5,ridge.cv_mape_pct,ridge.holdout_mape_pct,ridge.batch2_mape_pct],
        ["5개 특징 MAPE 가중",5,weighted["cv_mape_pct"],wvalid["mape_pct"],wtest["mape_pct"]],
        ["초기 용량 제거 로그 Ridge",4,penalty.iloc[-1].cv_mape_pct,penalty.iloc[-1].holdout_mape_pct,penalty.iloc[-1].batch2_mape_pct],
        ["IQR1개 선형회귀",1,curve.iloc[1].cv_mape_pct,curve.iloc[1].holdout_mape_pct,curve.iloc[1].batch2_mape_pct],
        ["PLS1성분",100,curve.iloc[2].cv_mape_pct,curve.iloc[2].holdout_mape_pct,curve.iloc[2].batch2_mape_pct]],
        columns=["모델","입력 수","Batch1 CV (%)","Hold-out (%)","Batch2 (%)"])
    summary.to_csv(out / "model_summary.csv",index=False)
    draw_development(out,stage,summary)
    draw_baseline(base,out)
    draw_penalty(out,penalty)
    draw_protocol_results(base, out)
    draw_domain_shift(base, out)
    write_report(base,out,stage,summary,penalty)
    source_paths = [base / "feature_experiments/stage_comparison.csv",
                    base / "capacity_penalty/comparison.csv", base / "curve_models/comparison.csv",
                    base / "ablation_loss/results.json", base / "modeling_cell_split/evaluation_predictions.csv",
                    base / "ablation_loss/performance_reporting.csv",
                    base / "modeling_cell_split/model_comparison.csv",
                    base / "feature_experiments/cell_features.csv",
                    base / "protocol_validation/results.json",
                    base / "protocol_validation/summary.csv",
                    base / "protocol_validation/stage_comparison.csv",
                    base / "protocol_validation/performance_reporting.csv",
                    base / "protocol_validation/split_manifest.csv",
                    base / "protocol_validation/evaluation_predictions.csv"]
    batch3_completed = (base / "batch3_test/results.json").is_file()
    if batch3_completed:
        source_paths += [base / "batch3_test/results.json", base / "batch3_test/evaluation_metrics.csv"]
        if (base / "batch3_original_test/results.json").is_file():
            source_paths += [base / "batch3_original_test/results.json", base / "batch3_original_test/evaluation_metrics.csv"]
    (out / "provenance.json").write_text(json.dumps(dict(
        purpose="Saved-results synthesis only", model_fit=False, evaluation_repeated=False,
        prior_model_selection_changed=False, batch3_evaluated=batch3_completed,
        sources={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}),
        ensure_ascii=False,indent=2))
    print(summary.round(3).to_string(index=False))
    print("Saved report and figures; no new training or evaluation.")


def draw_development(out,stage,summary):
    fig,axes=plt.subplots(1,2,figsize=(13,4.6),layout="constrained")
    x=np.arange(len(stage))
    axes[0].plot(x,stage.cv_mape_pct,"o-",color="#2878a5",lw=1.5)
    for i,row in enumerate(stage.itertuples()):
        axes[0].annotate(f"{row.cv_mape_pct:.2f}%\n{row.n_features} input(s)",
                         (i,row.cv_mape_pct),xytext=(0,9),textcoords="offset points",
                         ha="center",fontsize=8)
    axes[0].set(xticks=x,xticklabels=["Variance","Log Y","+ Min","+ QD","+ IR","+ Temp","+ Time","+ Policy"],
                ylabel="Batch1 CV MAPE (%)",title="1. Cumulative feature development",ylim=(6.8,10.6))
    axes[0].tick_params(axis="x",rotation=35,labelsize=8)
    axes[0].axhline(stage.iloc[3].cv_mape_pct,color="gray",ls="--",lw=.8)
    y=np.arange(len(summary))
    for offset,col,label,color in [(-.23,"Batch1 CV (%)","Batch1 CV","#2878a5"),
                                   (0,"Hold-out (%)","Hold-out","#e49b45"),
                                   (.23,"Batch2 (%)","Batch2 follow-up","#4b9468")]:
        bars=axes[1].barh(y+offset,summary[col],height=.23,label=label,color=color)
        axes[1].bar_label(bars,fmt="%.2f",padding=3,fontsize=7)
    axes[1].set(yticks=y,yticklabels=["Variance (1)","Log Ridge (5)","MAPE-weighted (5)",
                                    "Initial QD removed (4)","IQR (1)","PLS (1 component)"],
                xlabel="MAPE (%)",title="2. Internal vs. external evaluation",xlim=(0,43))
    axes[1].invert_yaxis()
    axes[1].legend(frameon=False,fontsize=8,loc="lower right")
    for ax in axes:
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="x" if ax is axes[1] else "y",alpha=.15)
    fig.savefig(out / "development.png",dpi=170)
    plt.close(fig)


def draw_baseline(base,out):
    features=pd.read_csv(base / "modeling_cell_split/cell_features.csv",dtype={"cell_id":str})
    manifest=pd.read_csv(base / "modeling_cell_split/split_manifest.csv",dtype={"cell_id":str})
    data=manifest[["batch","cell_id","role","label_eligible"]].merge(
        features.drop(columns="label_eligible",errors="ignore"),on=["batch","cell_id"],validate="one_to_one")
    dev=data.loc[data.role.eq("development")]
    valid=data.loc[data.batch.eq("Batch1")&data.role.eq("holdout")]
    test=data.loc[data.batch.eq("Batch2")&data.label_eligible&data.delta_status.eq("ok")]
    prediction=pd.read_csv(base / "modeling_cell_split/evaluation_predictions.csv",dtype={"cell_id":str})
    pred=prediction.loc[prediction.evaluation.eq("Test (Batch 2)")]
    coef=pd.read_csv(base / "modeling_cell_split/feature_coefficients.csv").query('candidate == "linear_F0"').iloc[0]
    merged=pred.merge(test[["batch","cell_id","log10_delta_Q_var"]],on=["batch","cell_id"],validate="one_to_one")
    # Verify the plotted algebraic line against existing saved predictions only.
    assert np.allclose(merged.prediction,coef.intercept_original_units+
                       coef.coefficient_original_units*merged.log10_delta_Q_var)
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.3),layout="constrained")
    xmin,xmax=dev.log10_delta_Q_var.min(),dev.log10_delta_Q_var.max()
    axes[0].axvspan(xmin,xmax,color="#2878a5",alpha=.08,label="Batch1 training X range")
    for subset,label,color,marker in [(dev,"Batch1 development (28)","#2878a5","o"),
                                      (valid,"Batch1 hold-out (8)","#e49b45","^"),
                                      (test,"Batch2 (39)","#4b9468","x")]:
        axes[0].scatter(subset.log10_delta_Q_var,subset.cycle_life,label=label,
                        color=color,marker=marker,s=30,alpha=.8)
    plotted=pd.concat([dev,valid,test])
    domain=np.linspace(plotted.log10_delta_Q_var.min(),plotted.log10_delta_Q_var.max(),120)
    axes[0].plot(domain,coef.intercept_original_units+coef.coefficient_original_units*domain,
                 color="#333333",lw=1.5,label="Frozen Batch1 fitted line")
    axes[0].axhline(500,color="gray",ls="--",lw=.8)
    axes[0].set(xlabel="log10 variance of Delta Q",ylabel="Actual cycle life",
                title="Single-feature relationship across batches")
    axes[0].legend(frameon=False,fontsize=7)
    for mask,label,color in [(pred.cycle_life.le(500),"Life <=500 (28)","#bd634d"),
                             (pred.cycle_life.gt(500),"Life >500 (11)","#4b9468")]:
        axes[1].scatter(pred.loc[mask,"cycle_life"],pred.loc[mask,"prediction"],
                        label=label,color=color,s=32,alpha=.8)
    low,high=300,1250
    axes[1].plot([low,high],[low,high],color="gray",ls="--",label="Ideal prediction")
    axes[1].set(xlim=(low,high),ylim=(low,high),xlabel="Actual cycle life",
                ylabel="Predicted cycle life",title="Batch2: remaining overprediction")
    axes[1].legend(frameon=False,fontsize=8)
    for ax in axes:
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(alpha=.15)
    fig.savefig(out / "single_feature.png",dpi=170)
    plt.close(fig)


def draw_penalty(out,penalty):
    fig,axes=plt.subplots(1,2,figsize=(10.5,3.8),layout="constrained")
    x=np.arange(4)
    for col,label,color in [("cv_mape_pct","Batch1 CV","#2878a5"),
                             ("holdout_mape_pct","Hold-out","#e49b45")]:
        axes[0].plot(x,penalty[col],"o-",label=label,color=color)
        for i,v in enumerate(penalty[col]):
            axes[0].annotate(f"{v:.2f}",(i,v),xytext=(0,7),textcoords="offset points",ha="center",fontsize=8)
    axes[0].set(title="Internal error increases",ylabel="MAPE (%)",ylim=(2.5,11))
    axes[0].legend(frameon=False,fontsize=8)
    axes[1].plot(x,penalty.batch2_mape_pct,"o-",color="#4b9468",label="Batch2 follow-up")
    for i,v in enumerate(penalty.batch2_mape_pct):
        axes[1].annotate(f"{v:.2f}",(i,v),xytext=(0,8),textcoords="offset points",ha="center",fontsize=8)
    axes[1].axhline(26.18567870657303,color="gray",ls="--",label="Variance baseline: 26.19")
    axes[1].set(title="External error decreases, but not enough",ylabel="MAPE (%)",ylim=(24,41))
    axes[1].legend(frameon=False,fontsize=8)
    for ax in axes:
        ax.set(xticks=x,xticklabels=["Penalty 1","Penalty 10","Penalty 100","Removed"])
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y",alpha=.15)
    fig.savefig(out / "capacity_tradeoff.png",dpi=170)
    plt.close(fig)


def configure_font():
    """Use an installed Korean font where available; figures remain plain."""
    preferred = ["Apple SD Gothic Neo", "Malgun Gothic", "NanumGothic", "Noto Sans CJK KR"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for family in preferred:
        if family in available:
            plt.rcParams["font.family"] = family
            break
    plt.rcParams.update({"font.size": 11, "axes.unicode_minus": False,
                         "axes.spines.top": False, "axes.spines.right": False})


def draw_protocol_results(base, out):
    q = base / "protocol_validation"
    result = json.loads((q / "results.json").read_text())
    values = pd.read_csv(q / "summary.csv").set_index("candidate")
    names = ["S0_linear", result["primary"]]
    colors = ["#38678f", "#b36a42"]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8), layout="constrained")
    x = np.arange(3)
    columns = ["cv_mape_pct", "Valid (Batch 1 Hold-out)", "Test (Batch 2)"]
    for i, (name, label, color) in enumerate(zip(names, ["분산 1개 기준 모델", "프로토콜 CV 선택 모델"], colors)):
        bars = axes[0].bar(x + (i - .5) * .34, values.loc[name, columns].astype(float),
                           width=.34, color=color, label=label)
        axes[0].bar_label(bars, fmt="%.2f", padding=4, fontsize=10)
    axes[0].set(xticks=x, xticklabels=["Batch1 CV", "Batch1 Hold-out", "Batch2"],
                ylabel="MAPE (%) · 낮을수록 좋음", ylim=(0, 46),
                title="동일한 29셀로 학습한 모델 비교")
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    axes[0].grid(axis="y", alpha=.15)
    pred = pd.read_csv(q / "evaluation_predictions.csv")
    for name, label, color, marker in zip(names, ["분산 1개", "프로토콜 CV 선택"], colors, ["o", "x"]):
        g = pred.loc[pred.candidate.eq(name) & pred.batch.eq("Batch2")]
        axes[1].scatter(g.cycle_life, g.prediction, s=32, color=color, marker=marker,
                        label=label, alpha=.85)
    all_test = pred.loc[pred.candidate.isin(names) & pred.batch.eq("Batch2")]
    low = 300
    high = np.ceil(max(all_test.cycle_life.max(), all_test.prediction.max()) / 100) * 100 + 50
    axes[1].plot([low, high], [low, high], ls="--", color="gray", lw=1, label="실제 = 예측")
    axes[1].set(xlim=(low, high), ylim=(low, high), xlabel="실제 수명 (사이클)",
                ylabel="예측 수명 (사이클)", title="Batch2: 대각선 위는 수명 과대 예측")
    axes[1].legend(frameon=False, fontsize=9)
    axes[1].grid(alpha=.15)
    fig.savefig(out / "protocol_evaluation.png", dpi=170)
    plt.close(fig)


def draw_domain_shift(base, out):
    frame = pd.read_csv(base / "feature_experiments/cell_features.csv", dtype={"cell_id": str})
    split = pd.read_csv(base / "protocol_validation/split_manifest.csv", dtype={"cell_id": str})
    dev = split.loc[split.role.eq("development"), ["batch", "cell_id"]].merge(
        frame, on=["batch", "cell_id"], validate="one_to_one")
    test = frame.loc[frame.batch.eq("Batch2") & frame.label_eligible & frame.delta_status.eq("ok")]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8), layout="constrained")
    for ax, feature, title, xlabel in zip(axes,
            ["cycle_life", "qd_initial_median"],
            ["학습에는 없는 단수명 구간", "초기 용량도 학습 범위를 벗어남"],
            ["수명 (사이클)", "초기 방전 용량 중앙값 (Ah)"]):
        low, high = dev[feature].min(), dev[feature].max()
        ax.axvspan(low, high, color="#38678f", alpha=.08)
        for i, (g, color) in enumerate([(dev, "#38678f"), (test, "#b36a42")]):
            values = np.sort(g[feature].to_numpy())
            # Fixed offsets separate overlapping cells without introducing randomness.
            offsets = np.array([-1, 0, 1, 2, -2])[np.arange(len(values)) % 5] * .026
            ax.scatter(values, np.full(len(values), i) + offsets, color=color, alpha=.8, s=23)
        ax.set(yticks=[0, 1], yticklabels=["Batch1 학습 29셀", "Batch2 39셀"],
               ylim=(-.45, 1.45), xlabel=xlabel, title=title)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=.15)
    axes[0].axvline(500, ls="--", lw=1, color="gray")
    axes[0].text(500, 1.23, "500 이하: Batch2 28셀", ha="left", fontsize=9)
    axes[1].text(.5, .02, "음영: 학습 범위 / 상한 초과: Batch2 19셀",
                 transform=axes[1].transAxes, ha="center", fontsize=9)
    fig.savefig(out / "domain_shift.png", dpi=170)
    plt.close(fig)


def write_report(base, out, stage, summary, penalty):
    q = base / "protocol_validation"
    result = json.loads((q / "results.json").read_text())
    reporting = pd.read_csv(q / "performance_reporting.csv")
    old_reporting = pd.read_csv(base / "ablation_loss/performance_reporting.csv")
    grouped = pd.read_csv(q / "summary.csv").rename(columns={
        "candidate": "모델", "n_features": "입력 수", "cv_mape_pct": "프로토콜 CV (%)",
        "Valid (Batch 1 Hold-out)": "Hold-out (%)", "Test (Batch 2)": "Batch2 (%)"})
    grouped["모델"] = grouped["모델"].replace({"S0_linear": "분산 1개 선형회귀",
        result["primary"]: "CV 선택: 로그 ElasticNet", "control_five_log_ridge": "5개 특징 로그 Ridge",
        "control_five_mape": "5개 특징 MAPE 가중"})
    grouped = grouped[["모델", "입력 수", "프로토콜 CV (%)", "Hold-out (%)", "Batch2 (%)"]]
    stages = pd.read_csv(q / "stage_comparison.csv")
    order = ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "F1"]
    stages = stages.loc[stages.method.eq("protocol")].set_index("stage").loc[order]
    original_stages = stage.set_index("stage")
    f1 = pd.read_csv(base / "modeling_cell_split/model_comparison.csv")
    f1_score = f1.loc[f1.feature_set.eq("F1"), "cv_mean_mape_pct"].min()
    rows = []
    meanings = ["분산 하나·원 수명", "타깃을 로그 수명으로 변경", "+ ΔQ 최솟값",
                "+ 초기 용량·변화율·기울기", "+ 저항 변화율", "+ 온도 변화·온도 폭",
                "+ 충전시간 중앙값", "+ C1·전환 SOC·C2", "DAY1 제안: 분산+충전시간+평균 온도"]
    for key, meaning in zip(order, meanings):
        old = f1_score if key == "F1" else original_stages.loc[key, "cv_mape_pct"]
        rows.append([key, meaning, old, stages.loc[key, "cv_mape_pct"]])
    stages_table = pd.DataFrame(rows, columns=["단계", "누적 변경 내용", "기존 셀 CV (%)", "프로토콜 CV (%)"])
    report = f"""# DAY2 — 모델 개발 및 평가

초기 100사이클의 측정값으로 배터리 셀의 총 수명을 예측했다. **본문의 특징 추가와 규제로 Batch1 내부 검증은 개선됐지만, Batch2 일반화 개선은 얻지 못했다.** 본문 개발 범위에서 관측된 Batch2 최저 MAPE는 기존 분산 1개 선형회귀의 **26.19%**였다. 이후 비즈니스 목적을 고려한 추가 탐색은 [별도 부록](DAY2_BUSINESS_APPENDIX.md)에 정리했으며, 본문의 고정 모델과 지정 성능표는 유지했다.

기존 셀 분리에서 CV로 선택한 모델은 MAPE 가중 회귀(Batch2 **33.88%**)다. 이후 충전 프로토콜을 분리해 재검증했을 때 선택된 모델은 로그 ElasticNet(Batch2 **37.14%**)이다. 두 실험은 학습 셀 구성이 달라 별도로 보고한다. 프로토콜 분리는 새로운 충전 조건에 대한 검증이며, 성능 개선 방법으로 제시하지 않는다.

이 보고서에서 **내부 검증**은 Batch1 안에서의 검증, **외부 평가**는 다른 실험 배치인 Batch2에 적용한 결과를 뜻한다. **일반화**는 학습에 사용하지 않은 셀이나 다른 조건의 셀에도 예측 관계가 유지되는지를 말한다. 먼저 기존 개발 과정을 설명하고, 그 뒤 프로토콜 분리로 같은 결론이 유지되는지 확인한다.

## 1. 예측 목적과 데이터

- **문제:** 회귀. 100사이클 완료 시점에 제공된 총 수명 `cycle_life`를 예측한다. 남은 수명을 예측하는 문제와 구분한다.
- **분석 단위:** 셀 1개당 입력 1행. 배치와 `cell_id`를 함께 식별자로 사용한다.
- **평가:** 주 지표 MAPE, 보조 지표 MAE·RMSE·R². 모든 MAPE는 백분율이며 낮을수록 좋다.

예를 들어 100사이클까지 관측한 셀의 총 수명을 800사이클로 예측한다면, 모델의 출력은 800이다. 남은 700사이클을 직접 예측한 결과와는 구분한다. 입력은 초기 기록을 요약한 숫자들이고, 학습할 정답은 제공된 총 수명이다.

MAPE는 각 셀에서 `|예측 수명 − 실제 수명| / 실제 수명 × 100`을 계산한 뒤 평균한 값이다. 실제 수명 400을 500으로 예측하면 오차는 25%다. 따라서 MAPE 26.19%를 ‘정확도 73.81%’로 바꿔 해석하지 않는다. MAE는 사이클 단위의 평균 절대오차, RMSE는 큰 오차에 더 민감한 지표다.

| 데이터 | 원본 셀 | 사용 | 처리 근거 |
| --- | ---: | ---: | --- |
| Batch1 | 46 | 36 | 후속 기록 연결이 필요한 5셀, 수명 종료 전 실험이 끝난 5셀 제외 |
| Batch2 | 47 | 39 | 수명 결측 8셀 제외; 학습에 사용하지 않음 |
| Batch3 | 46 | EDA 44 | 수명 결측 2셀 제외; 선택 사항인 모델 추가 평가는 미실행 |

후속 기록이 필요한 Batch1 셀은 0~4번, 미완료 셀은 8·10·12·13·22번이다(0부터 시작하는 번호). 첫 집단은 같은 셀의 실험이 다른 파일에서 계속된 경우라 현재 파일만으로 전체 수명을 확인하기 어렵다. 두 번째 집단은 수명 종료까지 관측하지 못했으므로 마지막 기록 시점을 실제 수명으로 취급하면 수명을 과소평가할 수 있다. 필요한 후속 파일이 없어 연결 수명을 임의로 만들지 않았다. **DAY1의 Batch1 46셀은 수명값이 존재하는 탐색 집합, DAY2의 36셀은 레이블 점검 후 모델링 집합**이다.

남은 Batch1 셀도 마지막 관측 용량은 0.880057~0.883363 Ah로, 엄격한 0.88 Ah 미만 도달을 직접 관측한 레이블은 아니다. 이번 과제에서는 제공 수명을 유지하며, 모든 셀의 정확한 EOL을 새로 확정했다고 주장하지 않는다. 제외 근거는 [셀별 레이블 점검표](label_audit.csv)에 있다.

## 2. EDA에서 모델 설계로 연결

| 확인한 내용 | 도메인 관점의 가설 | 실제 설계·검증 |
| --- | --- | --- |
| 초기 ΔQ 로그 분산과 수명에 강한 음의 관계 | 방전 곡선의 초기 변화가 이후 수명의 단서일 수 있음 | 분산 하나 선형회귀를 기준 모델로 설정 |
| ΔQ 평균·최솟값·분산의 높은 상관 | 같은 곡선의 요약값이 중복될 수 있음 | 분산으로 시작, 최솟값 추가 및 제거를 비교 |
| 초기 총용량이 비슷해도 이후 열화는 달라짐 | 초기 상태와 미세한 변화량이 추가 정보를 줄 수 있음 | 초기 용량·변화율·기울기를 추가 |
| 후기 용량 감소가 가속되는 knee 후보 | 열화 속도가 일정하지 않음 | 현상 해석에만 사용; 미래 정보인 실제 knee는 입력에서 제외 |
| 온도·IR·충전시간 관계가 배치마다 다름 | 열화 반응과 운전·측정 조건이 함께 반영될 수 있음 | 보조 특징으로 비교하고 배치 공통 효과라고 단정하지 않음 |
| 충전 정책별 수명 차이, C1 단독 관계는 배치마다 다름 | 전류 크기뿐 아니라 전환 SOC·두 번째 전류도 관련될 수 있음 | C1·SOC·C2를 함께 후보로 추가; 별도로 프로토콜 분리 검증 |

이 가설들은 특정 손상 기전을 입증한 설명이 아니다. 예를 들어 ΔQ가 3 V 주변에서 크게 변했다고 해당 전압에서 손상이 발생했다고 확정할 수 없다. 초기 곡선 변화로 수명을 예측한 선행 연구를 출발점으로 삼았다. [Severson et al. (2019)](https://www.nature.com/articles/s41560-019-0356-8)

**모델 후보 선정 이유:** 적은 셀 수와 강한 단일 특징 관계에 맞춰 평균 예측·선형회귀부터 시작했다. 중복 특징의 계수 불안정성을 완화하기 위해 Ridge와 ElasticNet을 비교했다. 이후 상대오차에 맞춘 가중 회귀와 ΔQ 표현을 바꾸는 IQR·PLS를 제한적으로 비교했다.

선형회귀는 입력 특징에 가중치인 **계수**를 곱해 수명을 예측한다. Ridge는 계수가 지나치게 커지지 않도록 벌점을 주고, ElasticNet은 계수를 줄이면서 일부를 0으로 만들 수도 있다. 이를 **규제**라고 한다. 특징이 늘어나더라도 작은 학습 집합의 관계에 과하게 의존하지 않도록 하는 장치이며, 규제를 넣는 것만으로 다른 배치에서도 잘 맞는다고 보장되지는 않는다.

## 3. 데이터 처리와 검증 구조

**입력은 100사이클까지로 제한했다.** 수명 결측은 채우지 않았고, IR의 0 등 유효하지 않은 저항값은 결측으로 취급했다. IR이 없는 셀을 일괄 삭제하지 않았다. 충전시간은 약 3,933분인 극단적인 기록이 평균을 크게 올려 중앙값으로 요약했다. 정상적인 후기 용량 감소를 이상치로 삭제하지 않았다.

| 구분 | 기존 개발 실험 | 프로토콜 분리 추가 검증 |
| --- | --- | --- |
| Batch1 개발 / Hold-out | 28 / 8셀 | 29 / 7셀 |
| 분리 기준 | 서로 다른 셀 | C1·전환 SOC·C2가 같은 셀을 한쪽에 배치 |
| 개발 데이터 내부 CV | 셀 단위 5-fold | 프로토콜 단위 5-fold |
| 후보 선택 기준 | 개발 CV 평균 MAPE | 개발 프로토콜 CV 평균 MAPE |
| 평가 | 별도 Hold-out, 이후 Batch2 39셀 | 별도 Hold-out, 이후 Batch2 39셀 |

**CV**는 개발 데이터를 5부분으로 나누어 번갈아 검증하는 방법이다. 기존 개발 28셀을 예로 들면, 한 번에 약 22~23셀로 학습하고 나머지 5~6셀로 검증한다. 검증할 부분을 바꾸어 다섯 번 반복한 뒤 MAPE를 평균한다. **Hold-out**은 이 28셀과 별도로 보관한 8셀이다.

역할은 **CV로 후보 비교 → 선택한 설정을 개발 셀 전체에 학습 → 별도 Hold-out 확인 → 같은 모델로 Batch2 평가**다. CV의 다섯 모델을 그대로 이어서 학습하는 것이 아니라, 선택된 특징·설정으로 개발 셀 전체에서 모델 하나를 새로 학습한다.

교수님 형식의 Train 항목에는 CV 검증 평균을 넣는다. 실제 학습 셀을 다시 예측한 오차와 다르다. Hold-out을 포함한 전체 36셀로 재학습하지 않았으므로 각 실험의 Valid와 Test에는 같은 모델을 적용했다.

결측 대체·표준화·PLS는 매 CV 학습 폴드에서만 fit했다. 예를 들어 IR 결측을 채우는 중앙값과 표준화에 쓰는 평균·표준편차는 그때의 학습 셀에서 계산하고, 검증 셀에는 그 값을 그대로 적용했다. 검증 데이터까지 섞어 기준을 계산하면 검증 정보가 학습에 들어갈 수 있기 때문이다. 무작위 분할과 해당 모델 설정은 seed 42로 고정했고, 비무작위 GroupKFold는 결정론적이다. 초기 특징의 셀별 계산은 재사용하되, 분리 후 전처리와 모델 선택을 다시 수행했다.

**평가 이력:** DAY1에서 세 배치를 탐색했고, DAY2 후속 가설에도 이전 Hold-out·Batch2 결과를 참고했다. 따라서 이번 결과는 반복 개발에 따른 탐색 결과이며, 엄격한 의미의 독립적인 최종 테스트는 아니다. 각 실행에서 후보를 CV로 먼저 고정한 것과, 전체 개발 과정에서 외부 결과를 반복 확인한 것은 구분한다.

## 4. 피처 엔지니어링과 기존 개발 결과

### 4.1 특징을 추가한 이유와 계산

ΔQ는 공통 전압에서 `Q100(V) − Q10(V)`로 계산했다. 전압 지점 간 분산·최솟값을 요약한다. **피처 엔지니어링**은 이처럼 곡선이나 사이클별 기록을 모델에 넣을 숫자로 바꾸는 과정이다.

같은 셀에서 10사이클과 100사이클의 방전 곡선이 얼마나 달라졌는지를 비교하므로, 단순히 초기 용량이 큰지 작은지만 보는 것과는 다르다. 분산은 변화량이 전압에 따라 얼마나 다른지를, 최솟값은 가장 크게 음의 방향으로 변한 정도를 요약한다.

| 특징 | 계산·역할 | 가설과 주의점 |
| --- | --- | --- |
| 로그 ΔQ 분산 | 전압별 ΔQ 분산의 log10 | 곡선 변화의 불균일성을 요약; 물리적인 손상량 자체는 아님 |
| ΔQ 최솟값 | 가장 음수인 ΔQ | 큰 국소 변화가 분산을 보완할 가능성; 중복·잡음에 주의 |
| 초기 용량 | 10~20사이클 QD 중앙값 | 초기 상태 정보; 측정·보관 조건도 함께 반영할 수 있음 |
| 용량 변화율 | (90~100 중앙값 − 초기 용량) / 초기 용량 | 상대적인 변화; 초기 증가를 열화 회복으로 단정하지 않음 |
| 초기 감소 속도 | 10~100사이클 QD의 7점 이동 중앙값에 선형 적합, 기울기 × −100 | 초기 추세; 단위 Ah/100사이클, 양수는 감소 |
| IR·온도 | 두 초기 구간의 IR 변화율·평균 온도 변화·온도 폭 | 저항·발열 관련 보조 신호; 배치마다 관계가 다를 수 있음 |
| 충전시간·충전 조건 | 시간 중앙값, C1·전환 SOC·C2 | 운전 조건과 셀 반응을 보완할 후보 |

DAY1에서 제안한 분산+충전시간+평균 온도(F1)도 실제 비교했다. 기존 CV 최저는 **{f1_score:.2f}%**로 분산 하나의 **9.28%**를 개선하지 못했다. 이후 초기 용량 변화 정보를 추가했다.

{table(stages_table)}

S0→S7은 특징을 누적한 비교이고 F1은 별도 DAY1 제안이다. 각 단계에서 모델 설정도 함께 선택했으므로 차이를 특정 변수 하나의 효과로 볼 수 없다. 두 CV 열은 개발 셀 구성이 다르므로 분리 방식만의 효과로 비교하지 않는다. **C1은 검토하지 않은 변수가 아니라, 누적 추가했을 때 더 좋은 선택 모델을 만들지 못한 후보**다. C1만 따로 추가한 효과는 확인하지 않았다.

![기존 셀 분리에서 특징 추가와 대표 모델 비교](process_review/development.png)

기존 실험에서는 초기 용량 특징을 넣은 S3가 CV를 9.28%에서 7.70%로 낮췄지만 Batch2는 26.19%에서 38.20%로 높아졌다. 내부 개선이 외부 개선을 보장하지 않았다. 왼쪽은 단계별 CV, 오른쪽은 실제 외부 평가한 대표 모델만 표시했다.

여기서 ‘초기 용량이 Batch1에서 유용했다’는 말은 **다른 특징과 함께 사용했을 때 Batch1 검증 오차를 줄이는 데 도움이 됐다**는 뜻이다. 초기 용량이 높으면 언제나 오래간다는 규칙을 확인한 것은 아니다. Batch2에서 오차가 커진 결과를 보고, 추가한 정보 중 일부가 배치에 따라 의미가 달라지는지 점검할 필요가 생겼다.

### 4.2 특징 제거·학습 목표·규제·곡선 표현

| 시도 | 왜 했는가 | 관측 결과와 판단 |
| --- | --- | --- |
| 특징 하나씩 제거 | 중복 정보와 초기 상태 의존을 확인 | 초기 용량 제거 시 CV 7.70→9.81%; 나머지 3개 단독 제거의 CV 차이는 0.012%p 미만으로 작았음 |
| MAPE 가중 학습 | 같은 사이클 오차도 단수명에서 상대오차가 큼 | 같은 5개 특징·alpha 0.01에서 무가중 CV 8.42→가중 7.11%; Batch2 33.88%로 기준 모델보다 나쁨 |
| 초기 용량 규제 강화 | Batch1에서 유용한 초기 상태 관계가 다른 배치에서는 달라질 수 있음 | 내부 오차를 양보하자 Batch2 오차는 감소했으나 기준 모델을 넘지 못함 |
| 로그 IQR로 분산 대체 | ΔQ 곡선의 극단값에 덜 민감한 퍼짐 정보 | CV 9.12%, Batch2 30.56%; 외부 개선 없음 |
| PLS로 곡선 압축 | 통계값 하나가 잃는 전압별 변화 형태를 활용 | 1~3성분 중 CV로 1성분 선택; CV 10.28%, Batch2 29.43%; 외부 개선 없음 |

MAPE 학습은 학습 수명의 역수를 가중치로 둔 절대오차 회귀이며, 각 폴드의 학습 레이블만 사용했다. 같은 100사이클 오차라도 수명 400인 셀에서는 25%, 수명 1,000인 셀에서는 10%다. 따라서 단수명 셀의 같은 절대오차를 더 크게 반영하도록 학습 목표를 맞춘 것이다. 정답 수명을 입력 피처로 넣은 것이 아니다. IQR·PLS는 관련 후속 연구를 참고했으나, 이번 입력 처리·타깃·분할 조건은 동일 재현이 아니다. [Attia et al. (2021)](https://arxiv.org/abs/2101.01885)

**규제 실험의 질문은 ‘초기 용량에 덜 의존하면 다른 배치에서 더 잘 맞을까?’였다.** 초기 용량 입력을 숫자상 작게 바꾸거나 셀을 삭제한 것이 아니라, 표준화된 초기 용량의 계수에 더 큰 벌점이 가도록 했다. 마지막에는 해당 특징을 완전히 제거한 경우도 비교했다.

IQR은 ΔQ 값 가운데 50%의 폭을 요약한 값이다. PLS는 여러 전압 지점의 ΔQ를 수명과 관련된 소수의 조합으로 압축한다. 둘 다 더 많은 센서를 추가하는 대신, 기존 방전 곡선에서 정보를 꺼내는 방식을 바꾼 시도다.

![초기 용량 의존을 줄였을 때 내부·외부 오차 변화](process_review/capacity_tradeoff.png)

초기 용량 규제 1→10→100배→제거에서 Batch2 MAPE는 **38.20→35.43→29.83→27.92%**로 낮아졌다. 그러나 기존 분산 모델 **26.19%**보다 높다. 배치 의존 가능성을 지지하는 결과이며, 다른 계수도 함께 재추정되므로 초기 용량의 독립적 인과효과는 아니다.

### 4.3 기존 셀 분리 결과 요약

{table(summary)}

이 단계의 CV 선택 모델은 **5개 특징 MAPE 가중 회귀(alpha=0.01)**다. `alpha`는 규제 강도를 정하는 **하이퍼파라미터**, 즉 학습 전에 후보값을 정하고 CV로 비교하는 설정값이다. Batch2 점수를 보고 분산 모델로 선택을 소급 변경하지 않았다.

‘CV 선택 모델’은 개발 데이터의 검증 기준으로 고른 모델이고, ‘Batch2 최저 모델’은 실제 평가 결과에서 가장 낮은 오차를 보인 모델이다. 이번에는 둘이 달랐다. 이 구분을 유지해야 모델을 고르는 기준과 사후에 알게 된 결과가 섞이지 않는다. PLS는 100개 전압 입력을 1성분으로 압축한 모델이며, 학습 계수가 하나인 모델과 같지 않다.

## 5. 충전 프로토콜을 분리한 추가 검증

### 5.1 왜 다시 검증했는가

Batch2에는 학습에서 보지 못한 충전 조합이 많았다. 서로 다른 셀이 같은 정책을 공유한다고 무조건 누수는 아니지만, **새로운 정책에서도 적용되는지를 보려면 정책 자체를 학습·검증에서 분리할 이유가 있다.** 이 목적과 교수님 안내의 동일 프로토콜 중복 우려를 반영했다.

예를 들어 동일한 `4.8C(80%)-4.8C` 정책으로 실험한 여러 셀을 학습과 검증 양쪽에 나누면, 새로운 셀을 예측하더라도 그 충전 방식 자체는 학습에서 이미 본 상태다. 프로토콜 분리에서는 그 정책의 셀을 한쪽에 모두 배치하여, 검증에서 처음 보는 정책의 셀을 예측하도록 한다.

C1은 첫 단계 충전 전류, 전환 SOC는 두 번째 전류로 바꾸는 충전 상태 기준, C2는 두 번째 단계 충전 전류를 나타낸다. 이 세 값의 정확한 숫자 조합으로 묶어 개발 29셀·16개 프로토콜, Hold-out 7셀·4개 프로토콜로 나눴다. Hold-out과 프로토콜 CV 각 폴드의 학습·검증 프로토콜 교집합은 모두 0이다. `newstructure`는 별도 표시이며 Batch1에서 일정하다.

이 분할로 기존 특징 단계, 특징 제거, 규제와 손실 후보를 다시 비교했다. 프로토콜 CV 최저인 **로그 ElasticNet(alpha=0.001, l1_ratio=0.5)**을 선택했다. 로그 수명을 학습한 뒤 사이클 단위로 역변환하여 모든 오차를 계산했다. `alpha`는 전체 규제 강도, `l1_ratio`는 계수를 0으로 만들 수 있는 벌점과 계수를 전반적으로 줄이는 벌점의 비중을 조절한다. 로그 수명은 수명의 배율 차이를 표현하려는 선택이며, MAPE를 직접 최소화하는 것과 같은 방법은 아니다. 동일한 29셀에서 셀 CV로 비교해도 같은 후보가 선택됐고, 그 모델의 셀 CV는 5.77%, 프로토콜 CV는 5.68%였다. 프로토콜 중복이 기존 내부 성능의 주원인이었다고 확정할 수 없다.

입력 명세는 분산·최솟값·초기 용량·변화율·기울기 5개다. 개발 29셀로 적합한 모델에서 실제 0이 아닌 계수는 **최솟값·초기 용량·변화율의 3개**였다. 분산과 기울기는 규제로 0이 됐으며, CV 폴드마다 선택되는 계수는 달라질 수 있다. C1은 이 선택 모델의 예측 입력에는 들어가지 않았고 분리 기준으로 사용됐다. **데이터를 나눌 때 쓰는 정보와 예측할 때 쓰는 정보는 다를 수 있다.**

분산 계수가 0이라는 결과도 ‘ΔQ 정보가 필요 없다’는 뜻은 아니다. 같은 곡선에서 구한 최솟값과 분산은 서로 상관이 높으므로, 이 모델에서는 최솟값이 그 정보를 대신 전달했을 수 있다. 실제 학습에 들어간 입력은 5개이고, 해당 적합 결과에서 직접 예측에 기여한 계수가 3개라는 의미다.

### 5.2 같은 학습 셀에서의 비교

{table(grouped)}

![프로토콜 분리 모델의 내부·외부 성능과 Batch2 예측](process_review/protocol_evaluation.png)

왼쪽에서 CV 선택 모델은 기준 모델보다 Batch1 내부 오차가 작지만 Batch2 오차는 크다. 오른쪽에서 대각선 위 점은 과대 예측이며, 단수명 셀에서 이런 오차가 많이 보인다. **프로토콜 분리를 고려해도 배치 간 일반화 개선은 얻지 못했다.**

분산 모델의 Batch2는 기존 26.19%, 새 분할 28.68%다. 각 분할의 CV 선택 모델은 33.88%, 37.14%다. 분리 전 결과가 더 좋았지만 학습 셀과 일부 모델 설정이 바뀌었으므로, 차이를 분리 방식 하나의 인과효과로 해석하지 않는다.

## 6. 교수님 지정 형식의 성능 보고

**기존 셀 분리 — CV 선택 MAPE 가중 회귀**

{table(old_reporting)}

**프로토콜 분리 추가 검증 — CV 선택 로그 ElasticNet**

{table(reporting)}

표의 Gap은 이름에 적힌 앞뒤 순서를 단순히 빼지 않고, 교수님이 제시한 ‘양수이면 악화’에 맞춰 **Valid−CV, Test−Valid, Test−9.1**로 계산했다. MAPE는 %, 차이는 **%p**다. 양의 CV–Valid 차이는 과적합 의심 지표이지 그 자체로 과적합의 증명은 아니다.

기존 실험의 Gap(Train−Valid) −2.70%p는 Hold-out 오차가 CV보다 작았다는 뜻이다. 검증 셀이 8개뿐이어서 그 셀들의 구성이나 예측 난이도에 따른 차이도 있을 수 있다. 반면 Gap(Valid−Test) +29.47%p는 Batch2로 옮겼을 때 오차가 크게 증가했음을 보여준다. 이는 학습 셀을 외웠다는 증거 하나로 해석하기보다 배치 차이와 함께 살펴볼 결과다.

원논문 참고값 9.1%를 달성하지 못했다. 과제의 Batch1 학습→Batch2 평가 조건과 논문의 데이터·학습/테스트 구성이 다르므로 이 차이를 동일 조건의 재현 오차로 해석하지 않는다. 특히 로컬 Batch2 파일 날짜는 2018-02-20, 공식 처리 코드의 두 번째 파일은 2017-06-30이다. [공식 처리 코드](https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation/blob/master/LoadData.m)

회귀를 선택했으므로 분류의 Accuracy·F1은 산출하지 않았다. Batch3 모델 평가는 선택 사항으로 수행하지 않았다.

## 7. 왜 Batch2에서 오차가 남았는가

최종 ElasticNet의 Batch2 상대오차 상위 3셀을 확인하면 단수명을 길게 예측하는 경향이 드러난다. 셀 번호는 0부터 시작하며, 예측은 반올림했다.

| Batch2 셀 | 충전 정책 | 실제 수명 | 예측 수명 | 상대오차 (%) |
| --- | --- | ---: | ---: | ---: |
| 6 | 3.6C(9%)-5C | 393 | 704 | 79.06 |
| 29 | 5.2C(58%)-4C | 452 | 734 | 62.29 |
| 18 | 5.2C(50%)-4.25C | 449 | 717 | 59.64 |

단수명 28셀의 MAPE는 **38.92%**였다. 짧은 수명을 과대예측하는 문제가 두드러졌지만, 나머지 11셀도 **32.61%**로 배치 전체의 차이를 함께 살펴볼 필요가 있다.

![프로토콜 분리 개발 데이터와 Batch2의 수명·초기 용량 범위](process_review/domain_shift.png)

음영은 프로토콜 분리 개발 29셀의 범위다. **Batch2 28셀의 수명은 500 이하로 학습 범위(534~1,054) 밖이며, 19셀의 초기 용량도 학습 상한보다 높다.** 이 차이는 단순히 같은 프로토콜을 양쪽에 배치했는지와 다른 문제다.

| 확인된 사실 | 가능한 설명 | 다음 개선 방향 |
| --- | --- | --- |
| Batch1에 500 미만 수명 셀이 없음 | 단수명 사례 부족이 과대예측의 주요 원인으로 보임 | 단수명 사례를 확보해 학습 범위를 보완 |
| 초기 용량·변화 추세가 학습 범위를 벗어남 | 상태·측정 조건과 열화 신호가 함께 학습됐을 가능성 | 다양한 초기 상태에서 특징의 일관성 확인 |
| 같은 배치의 미관측 프로토콜에서는 개선, Batch2에서는 악화 | 충전 조합뿐 아니라 수집 배치의 차이도 영향을 준 것으로 보임 | 같은 정책을 여러 배치에서 반복 측정해 비교 |
| 여러 특징을 추가해도 외부 개선이 없음 | Batch1에서 유용한 관계가 다른 배치에서 약해짐 | 단순 모델을 기준으로 유지하고 새 데이터에서 재검증 |

여기서 **학습 범위 밖 예측(외삽)**이란, 학습 중 확인하지 못한 값의 구간까지 배운 관계를 적용하는 것이다. 선형 모델은 범위 밖의 수명을 출력할 수 있지만, 그 구간에서도 같은 관계가 맞는지는 학습 데이터로 확인하지 못했다. 초기 용량도 마찬가지로, Batch1에서 유용했던 용량 차이가 Batch2에서는 같은 수명 차이를 뜻하지 않을 수 있다.

`newstructure`는 Batch1에 없고 Batch2 9셀에 있다. 이 표시의 물리적 의미는 확정하지 않았다. 또한 새 개발 집합의 C1은 5.4~8C이고 Batch2 28셀은 5.4C 미만이다. 따라서 새 검증에는 학습보다 낮은 첫 단계 전류로 충전한 셀을 예측하는 조건도 포함된다. C1을 직접 입력하지 않았더라도 충전 조건의 차이가 ΔQ나 용량 특징에 간접적으로 나타날 수 있다. 수명 범위·초기 상태·충전 조건의 차이가 함께 외부 예측을 어렵게 만든 것으로 예상한다.

## 8. 결론과 평가 한계

**로그 ΔQ 분산은 적은 특징으로 시작할 타당한 기준이었다. 초기 용량 특징과 MAPE 학습은 Batch1 내부 성능을 개선했지만, Batch2 일반화에는 실패했다. 초기 용량 의존을 줄이거나 곡선 표현을 바꿔도 기존 단일 분산 모델의 관측 최저 26.19%를 개선하지 못했다.**

프로토콜을 분리한 검증에서도 이 결론은 유지됐다. 검증 목적을 더 구체화하는 것과 모델의 외부 성능을 높이는 것은 별개다. 이번 결과는 더 많은 특징보다 **학습 범위와 배치 간 관계의 안정성**을 함께 확인해야 함을 보여준다.

- 수명과 상관이 있다는 사실을 충전 조건의 인과효과나 특정 열화 기전으로 표현하지 않는다.
- 작은 표본에서 여러 후보를 비교했으므로 선택된 CV 최저값은 낙관적일 수 있다. 새로운 독립 표본에서 검증한 성능으로 일반화하지 않는다.
- Hold-out·Batch2를 후속 개발에서 반복 참고했다. ‘Test를 마지막에 한 번만 사용’한 완전한 독립 평가 요건은 충족하지 못했으며, 이 한계를 숨기지 않는다.
- Knee는 육안 탐색, 충전 패턴은 정책의 C1·SOC·C2 요약 분석이다. 정밀 knee 검출이나 실제 전류 파형 분석을 수행했다고 주장하지 않는다.

## 9. 재현 및 근거 파일

실행 환경·노트북 순서는 [실행 안내](../../docs/PROJECT_GUIDE.md)에 있다. 상세 결과는 [프로토콜 분리 검증 부록](DAY2_PROTOCOL_VALIDATION.md), [기존 결과 요약](process_review/model_summary.csv), [프로토콜 분리 성능표](protocol_validation/performance_reporting.csv)에서 확인할 수 있다. 개발 종료 후의 [비즈니스 목적 추가 실험 부록](DAY2_BUSINESS_APPENDIX.md)은 본문 모델을 교체하지 않은 후속 탐색 기록이다.

`src/day2_process_review.py`는 저장된 결과를 읽어 이 문서와 그림만 생성한다. 이번 정리는 새 모델 학습·후보 선택·평가를 수행하지 않았다. 입력 특징, 분할표, 모델, 예측, 설정·환경 기록은 각 실험 폴더에 보존했다.
"""
    if (base / "batch3_test/results.json").is_file():
        scores = pd.read_csv(base / "batch3_test/evaluation_metrics.csv")
        primary = scores.loc[scores.candidate.eq("S3_elasticnet_0.001_0.5")
                             & scores.evaluation.eq("Test (Batch 3)")].iloc[0]
        baseline = scores.loc[scores.candidate.eq("S0_linear")
                              & scores.evaluation.eq("Test (Batch 3)")].iloc[0]
        report = report.replace("| Batch3 | 46 | EDA 44 | 수명 결측 2셀 제외; 선택 사항인 모델 추가 평가는 미실행 |",
                                "| Batch3 | 46 | 추가 평가 44 | 수명 결측 2셀 제외; 기존 고정 모델 2개로 평가 |")
        report = report.replace("Batch3 모델 평가는 선택 사항으로 수행하지 않았다.",
                                "Batch3 추가 평가도 수행했으며 결과는 아래 §8에 정리했다.")
        report = report.replace("## 8. 결론과 평가 한계", "## 9. 결론과 평가 한계")
        report = report.replace("## 9. 재현 및 근거 파일", "## 10. 재현 및 근거 파일")
        extra = f"""## 8. Batch3 추가 평가

현재 프로토콜 CV 선택 ElasticNet과 같은 개발 29셀로 학습한 분산 기준 모델을 **재학습 없이** 평가했다. Batch3 원본 46셀 중 수명 결측 2셀을 제외한 44셀을 사용했다. 원본 초기 특징과 저장 입력의 일치, 동일 전압 격자, 초기 100사이클 이내 입력을 확인했고, 큰 예측 오차를 이유로 셀을 삭제하지 않았다.

| 같은 학습 29셀 모델 | Batch2 MAPE (%) | Batch3 MAPE (%) | Batch2 MAE | Batch3 MAE |
| --- | ---: | ---: | ---: | ---: |
| CV 선택 ElasticNet | 37.14 | {primary.mape_pct:.2f} | 197.70 | {primary.mae:.2f} |
| 분산 1개 선형회귀 | 28.68 | {baseline.mape_pct:.2f} | 142.76 | {baseline.mae:.2f} |

![Batch3 실제 수명과 고정 모델 예측](batch3_test/predictions.png)

**Batch3에서도 분산 기준 모델의 전체 MAPE가 낮았다.** 다만 ElasticNet은 학습 수명 범위 안의 27셀에서는 7.57%로 분산 모델의 8.71%보다 좋았고, 학습 최대 수명 1,054사이클을 넘는 17셀에서는 27.06%로 분산 모델의 17.46%보다 나빴다. 이 17셀을 ElasticNet이 전부 과소예측했으며 평균 약 395사이클 부족했다. 추가 특징의 내부 검증 이점이 장수명 외삽으로 이어지지 않았다.

Batch3의 MAPE는 Batch2보다 낮지만 사이클 단위 MAE는 거의 비슷하거나 조금 커졌고, RMSE는 두 모델 모두 증가했다. 더 긴 실제 수명은 같은 사이클 오차의 상대오차를 작게 만들 수 있으므로 MAPE 개선만으로 모든 예측이 좋아졌다고 해석하지 않는다.

이번 실행이 Batch3의 첫 모델 평가지만, Batch3는 이미 EDA에 사용했다. 완전히 보지 않은 데이터라고 주장하지 않고, 추가 점수로 모델을 재선정·튜닝하지 않았다. Batch3 전체 지정 성능표·품질 점검·비교 그래프는 [Batch3 추가 테스트 보고서](DAY2_BATCH3_TEST.md)에 있다.

"""
        original_path = base / "batch3_original_test/results.json"
        if original_path.is_file():
            original = json.loads(original_path.read_text(encoding="utf-8"))
            extra += f"기존 셀 분리 **28셀** 모델(Batch2 26.19%)도 이후 사용자의 요청으로 같은 Batch3 44셀에서 평가했다. MAPE는 **{original['batch3_metrics']['mape_pct']:.2f}%**로, 프로토콜 분리 29셀 분산 모델의 {baseline.mape_pct:.2f}%와 가까웠다. 이미 첫 Batch3 결과를 본 뒤의 추가 확인이며 재학습·튜닝·모델 재선정은 없었다. 학습 셀 구성이 다르므로 작은 차이를 분리 방식의 인과적 효과로 해석하지 않는다.\n\n"
        report = report.replace("## 9. 결론과 평가 한계", extra + "## 9. 결론과 평가 한계")
    (base / "DAY2_REPORT.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
