# ESS 배터리 수명 예측

**초기 100사이클의 측정값으로 배터리 셀의 총 수명(Cycle Life)을 예측**하고, 다른 실험 배치에서도 그 관계가 유지되는지 평가한다. ESS의 조기 품질 선별과 교체·보증 계획에 쓸 수 있는 신호를 탐색하는 것이 목적이다.

[DAY1: EDA·모델 전략](outputs/final/DAY1_REPORT.md) · [DAY2: 개발·평가](outputs/day2/DAY2_REPORT.md) · [Batch3 추가 평가](outputs/day2/DAY2_BATCH3_TEST.md) · [비즈니스 부록](outputs/day2/DAY2_BUSINESS_APPENDIX.md) · [실행 안내](docs/PROJECT_GUIDE.md)

---

## 프로젝트 개요

- **데이터셋:** MIT–Stanford Battery Dataset (Severson et al., Nature Energy 2019), 수업 제공 파일
- **학습 데이터:** Batch1 (2017-05-12). 원본 46셀 중 레이블 점검 후 36셀 사용
- **평가 데이터:** Batch2 (2018-02-20) 39셀, 추가 Batch3 (2018-04-12) 44셀. 수명 결측 셀 제외
- **Task:** Regression (Cycle Life 예측)
- **Target / Input:** `cycle_life` / 초기 100사이클 이내 정보
- **Metric:** 주 지표 MAPE(%), 보조 지표 MAE·RMSE·R²

### 검증 구조

```text
Batch1 (36셀)
 ├─ Development 29셀 ── 프로토콜 단위 5-fold CV (모델 선택)
 └─ Hold-out 7셀     ── 별도 검증

Batch2 39셀 ── 외부 평가
Batch3 44셀 ── 추가 외부 평가
```

같은 `C1 + 전환 SOC + C2` 충전 조합이 학습과 검증 양쪽에 들어가지 않도록 분리했다.

### 한눈에 보기

- Batch1 **프로토콜 CV에서는 ElasticNet**(5개 특징)이 가장 낮은 오차를 보였다.
- Batch2·3 외부 배치에서는 **`log10_delta_Q_var` 하나를 쓰는 선형회귀**의 오차가 ElasticNet보다 낮았다.
- 최종 추천 모델(선형회귀)의 MAPE: Batch1 CV **7.93%** · Hold-out **10.73%** · Batch2 **28.68%** · Batch3 **12.09%**
- 이 추천에는 Batch2·3 결과가 반영되어 **독립적인 최종 테스트가 아니다.**
- 핵심 한계 중 하나는 배치 간 분포 차이다. Batch1에는 500사이클 미만 셀이 없지만 Batch2는 71.79%가 단수명 셀이다. 실제로 Batch2 단수명 셀의 수명을 길게 예측하는 경향이 확인됐다.

**핵심 메시지**

1. 초기 `ΔQ(V)` 변화는 수명과 강한 상관을 보였다.
2. Batch1 **프로토콜 CV에서는 여러 특징을 쓰는 ElasticNet이 가장 낮은 오차를 보였다.**
3. 외부 Batch2·3에서는 특징 하나의 단순 선형회귀 오차가 ElasticNet보다 낮았다.
4. Batch1에 단수명 사례가 없는 점은 Batch2 성능 저하의 유력한 요인 중 하나다. 초기 상태·충전 조건 등 다른 배치 차이와의 기여도는 분리하지 못했다.
5. 다양한 수명·운영 조건의 학습 데이터를 보완하면 개선될 가능성이 있으나, 별도 검증이 필요하다.

---

## 파일 구조

```text
├── data/README.md                 # 원본 MAT 배치 안내
├── notebooks/
│   ├── 30-ESSHealth-DAY1-EDA.ipynb
│   ├── 40~44-*.ipynb              # 기준 모델·특징·규제·곡선 실험
│   ├── 45-*.ipynb                 # 프로토콜 분리 검증
│   ├── 46-*.ipynb                 # Batch3 평가
│   └── 47-*.ipynb                 # 비즈니스 부록
├── src/                           # 특징 추출·학습·평가·검산
├── outputs/
│   ├── final/                     # DAY1 보고서·그림
│   └── day2/                      # DAY2 보고서·모델·예측·성능표
├── docs/PROJECT_GUIDE.md          # 실행 순서·평가 기준 점검
├── requirements.txt
└── README.md
```

## 환경 설정

Python **3.11.15** (macOS·Linux 기준). 라이브러리 버전은 `requirements.txt`에 고정했다.

```bash
git clone https://github.com/piw0814-create/data_project.git
cd data_project
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user --name data-project --display-name "Python (data-project)"
.venv/bin/python -m jupyterlab
```

보고서·노트북에는 그래프와 실행 결과가 저장되어 있어 열람에는 설치가 필요 없다. 원본 EDA·특징 추출에는 [MAT 파일](data/README.md)이 필요하다. 저장 결과 검산: `.venv/bin/python -m src.verify_results`

---

## EDA

| 질문 | 핵심 발견 | 모델링 반영 |
| --- | --- | --- |
| Cycle Life 분포 | 500 미만 비율: Batch1 0%, Batch2 71.79%, Batch3 0%. 1,000 초과: 21.74%, 7.69%, 52.27% | 단수명·장수명 구간 오차를 따로 확인 |
| 열화 곡선·knee | 후기에 감소가 가속되는 셀이 많고 시작 시점은 셀마다 다름 (knee는 육안 탐색) | 초기 추세만 후보로 사용. 전체 수명을 본 뒤 아는 knee는 입력에서 제외 |
| ΔQ(V) | 대표 단수명 셀에서 100−10사이클 곡선 변화가 큼. 로그 분산–수명 상관 −0.886 / −0.902 / −0.702 | 로그 ΔQ 분산을 기준 특징으로 사용 |
| 충전 조건 | C1 단독으로 수명 순서를 설명하기 어려움. 전환 SOC·C2·실험 집단에 따라 차이 | 충전 조합을 후보 특징과 검증 그룹으로 고려 |
| 상관·데이터 품질 | ΔQ 평균·최솟값·분산의 정보 중복, 충전시간 극단값, IR=0 | 대표 특징부터 시작, 시간은 중앙값, IR=0은 결측 처리 |

**추가 확인:** Batch2 일부 셀의 `newstructure` 표시(Batch1에는 없음), 학습 범위 밖의 초기 용량, 충전 프로토콜 분리 재검증.

분포 통계는 수명값이 있는 셀 기준이다. DAY2에서는 후속 기록 연결이 필요한 5셀과 실험 미완료 5셀을 Batch1에서 추가로 제외했다.

---

## Modeling

### 피처 엔지니어링 전략

공통 전압에서 `ΔQ(V) = Q100(V) − Q10(V)`를 계산했다.

```text
초기 100사이클
│
├─ Q10(V), Q100(V)
│    └─ ΔQ(V) = Q100(V) − Q10(V)
│         ├─ 로그 분산 (기준)
│         └─ 최솟값
│
└─ 사이클별 방전 용량
     ├─ 초기 용량
     ├─ 용량 변화율
     └─ 초기 감소 속도
```

곡선의 변화와 초기 상태가 서로 보완적인 정보를 줄 것이라는 가설로 특징을 확장했다. IR·온도·충전시간·C-rate도 비교했으나 더 나은 CV 후보를 만들지 못했다. 결측 대체·표준화는 CV 학습 폴드 안에서만 학습했다.

### 모델 선택 및 근거

- **후보 모델:** 평균 예측, 선형회귀, Ridge, ElasticNet. 이후 특징 제거, 규제 강화, MAPE 가중 학습, IQR·PLS 표현을 추가 비교
- **최종 추천 모델:** `log10_delta_Q_var` 1개 → `cycle_life`를 직접 예측하는 **선형회귀** (`S0_linear`)
- **선택 이유**
  1. 같은 29셀로 학습한 ElasticNet보다 Batch2·3 오차가 낮았다.
  2. 입력이 하나라 구조가 단순하다.
  3. EDA에서 확인한 ΔQ–수명 관계와 직접 연결된다.
- **이전 CV 선택 모델:** 로그 수명 ElasticNet. Batch1 CV는 가장 좋았지만 외부 배치에서는 오차가 더 컸다. 당시 선택 기록은 유지했다.

> **주의:** 최종 추천에는 Batch2·3 결과가 반영됐다. 아래 성능은 추천 이후 독립적으로 처음 확인한 테스트가 아니며, 다음 독립 배치에서 추가 검증할 모델로 추천한다.

---

## 성능 결과

### 모델 비교 (같은 Batch1 개발 29셀·Hold-out 7셀, MAPE)

| 모델 | 입력 수 | Batch1 CV | Hold-out | Batch2 | Batch3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 분산 1개 선형회귀 | 1 | 7.93% | 10.73% | **28.68%** | **12.09%** |
| 로그 ElasticNet | 5 | **5.68%** | **8.92%** | 37.14% | 15.10% |

![같은 학습 셀에서의 모델 성능과 Batch2 예측 비교](outputs/day2/process_review/protocol_evaluation.png)

- 왼쪽은 내부 검증 개선이 Batch2 개선으로 이어지지 않은 결과다.
- 오른쪽에서 대각선 위 점은 실제보다 수명을 길게 예측한 셀이다.

```text
Batch1 프로토콜 CV: ElasticNet 우세
        ↓
Batch2 / Batch3: 단순 선형회귀의 오차가 더 낮음
        ↓
내부 검증 개선 ≠ 배치 간 예측 개선
```

### 교수님 지정 형식 (최종 추천 모델)

Train은 재예측 오차가 아니라 **CV 검증 평균**이다.

| 구분 | MAPE (%) | 비고 |
| --- | ---: | --- |
| Train (Batch 1 CV) | 7.93 | 개발 29셀, 프로토콜 단위 5-fold |
| Valid (Batch 1 Hold-out) | 10.73 | 별도 7셀 |
| Test (Batch 2) | 28.68 | 39셀; 추천에 참고한 기존 평가 |
| Gap (Train-Valid) | +2.80 | Valid − CV, %p |
| Gap (Valid-Test) | +17.96 | Test − Valid, %p |
| Gap (Target-Test) | +19.58 | Test − 논문 참고값 9.1%, %p |
| Test (Batch 3) | 12.09 | 44셀; 추천에 참고한 기존 평가 |
| Gap (Batch2-Batch3) | −16.60 | Batch3 − Batch2, %p |
| Gap (Target-Test, Batch3) | +2.99 | Batch3 − 과제 공통 참고값 9.1%, %p |

Train→Valid(+2.80%p)보다 Valid→Batch2(+17.96%p) 차이가 훨씬 크다. 단순 모델에서도 배치 차이의 영향이 남았다. 단, Hold-out이 7셀뿐이라 과적합 여부를 단정하지 않는다.

**참고**

- 9.1%는 논문 참고값이며 사용 파일·학습/테스트 구성이 달라 동일 조건 재현이 아니다.
- Batch3는 수명이 길어 MAPE가 작아지는 영향이 있다. MAE는 Batch2 142.76 → Batch3 150.22사이클로 오히려 커졌다.
- 처음 셀 분리의 28셀 모델(Batch2 26.19%·Batch3 11.94%)은 학습 셀이 달라 별도 실험이며, 현재 모델의 점수로 쓰지 않았다. 상세는 [DAY2 보고서](outputs/day2/DAY2_REPORT.md)를 참고한다.
- 개발 중 Hold-out·Batch2를 반복 확인했으므로 완전히 독립적인 최종 테스트 조건은 충족하지 못했다.

### 비즈니스 목적별 추가 비교 (요약)

과대예측에 더 큰 벌점을 주는 **분위수 0.3 회귀**는 Batch2 MAPE를 28.68% → 25.44%로 낮췄지만 Batch3는 12.09% → 14.59%로 나빠졌다. **분위수 0.5**는 CV와 Batch2에서 더 좋아 유효한 대안이며, 현재 추천이 유일한 최적 모델이라는 뜻은 아니다. 배치와 검증 기준을 함께 만족하는 일관된 개선을 확립하지 못해 추가 모델은 채택하지 않았다. 분위수 0.3은 검증된 수명 보증 하한이 아니다. 자세한 내용은 [비즈니스 부록](outputs/day2/DAY2_BUSINESS_APPENDIX.md)에 있다.

---

## 오류 분석

- **Batch2:** 상대오차 상위 3셀(6·18·15번)은 실제 393·449·396, 예측 662·746·656사이클로 모두 **단수명 과대예측**이었다. 단수명 28셀 MAPE는 32.84%.
- **Batch3:** 학습 최대 수명 1,054를 넘는 17셀 중 16셀을 **과소예측**했다. 이 구간 MAPE는 17.46%.
- **원인 가설:** Batch1에 단수명 사례가 없고 수명 범위가 좁은 점은 유력한 요인 중 하나다. 다만 초기 상태·충전 조건·특징–수명 관계의 배치 차이와 각각의 기여도는 분리하지 못했다.
- **개선 방향:** 단수명과 다양한 운영 조건의 사례를 보완하면 개선될 가능성이 있다. 이는 가설이며 별도 검증이 필요하다.

---

## ESS 도메인 해석

- **활용:** 초기 시험 결과로 정밀 검사가 필요한 셀을 선별하고, 교체·보증 계획의 참고 수명으로 활용할 수 있다.
- **오류 비용:**
  ```text
  수명 과대예측 → 교체 지연 위험
  수명 과소예측 → 정상 셀 조기 폐기, 잔존 가치 손실
  ```
  어느 오류의 비용이 큰지 정한 뒤 새 데이터에서 검증해야 한다.
- **한계와 추가 필요 사항:** 실험실 셀 데이터이므로 ESS의 온도·SOC 범위·부하 변화·달력 노화·팩 내 편차를 반영한 검증이 필요하다. 운영에서는 배치별 입력 분포와 실제 오차를 추적하고, 새 조건의 데이터가 쌓이면 재학습하는 방향이 적절하다.

---

## 참고문헌

- [Severson et al. (2019)](https://www.nature.com/articles/s41560-019-0356-8). Data-driven prediction of battery cycle life before capacity degradation. *Nature Energy*, 4, 383–391.
- [원논문 공식 코드](https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation)
- [Attia, Severson & Witmer (2021)](https://arxiv.org/abs/2101.01885)

## 팀 구성

- 박인우: EDA, 피처 엔지니어링, 모델 개발, 성능 평가(Batch2·3)
