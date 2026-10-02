# ESS 배터리 수명 예측

**초기 100사이클의 측정값으로 배터리 셀의 총 수명을 예측**하고, 다른 실험 배치에서도 예측이 유지되는지 평가했다. ESS의 조기 품질 선별과 교체 계획에 활용할 수 있는 신호를 탐색하는 것이 목적이다.

[DAY1: EDA·모델 전략](outputs/final/DAY1_REPORT.md) · [DAY2: 개발·평가](outputs/day2/DAY2_REPORT.md) · [Batch3 추가 평가](outputs/day2/DAY2_BATCH3_TEST.md) · [실행 안내·과제 기준 점검](docs/PROJECT_GUIDE.md)

## 프로젝트 개요

- **데이터:** MIT–Stanford Battery Dataset 계열의 수업 제공 파일 (Severson et al., 2019).
- **학습:** Batch1 (2017-05-12). 원본 46셀 중 레이블 점검 후 36셀 사용.
- **평가:** Batch2 (2018-02-20) 39셀, 추가 Batch3 (2018-04-12) 44셀. 수명 결측 셀 제외.
- **태스크:** Regression. 정답은 제공된 총 수명 `cycle_life`; 주 지표는 MAPE(%), 보조 지표는 MAE·RMSE·R².
- **최종 검증 구조:** Batch1 개발 29셀에서 프로토콜 단위 5-fold CV → 별도 Hold-out 7셀 → 고정 모델로 Batch2·3 평가. 같은 C1·전환 SOC·C2 조합을 학습/검증 양쪽에 나누지 않았다.

## 파일 구조

```text
├── data/README.md                 # 원본 MAT 배치 안내
├── notebooks/
│   ├── 30-ESSHealth-DAY1-EDA.ipynb
│   ├── 40~44-*.ipynb             # 기준 모델·특징·규제·곡선 실험
│   ├── 45-*.ipynb                # 프로토콜 분리 최종 검증
│   ├── 46-*.ipynb                # Batch3 평가
│   └── 47-*.ipynb                # 비즈니스 부록: 저장 결과 분석
├── src/                          # 특징 추출·학습·평가·검산
├── outputs/
│   ├── final/                    # DAY1 보고서·그림
│   └── day2/                     # DAY2 보고서·실험별 모델·예측·성능표
├── docs/PROJECT_GUIDE.md          # 실행 순서·평가 기준 점검
├── requirements.txt
└── README.md
```

## 환경 설정

Python **3.11.15**, 라이브러리 버전은 `requirements.txt`에 고정했다. 아래는 macOS·Linux 기준이다.

```bash
git clone https://github.com/piw0814-create/data_project.git
cd data_project
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user --name data-project --display-name "Python (data-project)"
.venv/bin/python -m jupyterlab
```

보고서·노트북에는 그래프와 실행 결과가 저장되어 있어 **열람에는 설치가 필요 없다.** 원본 EDA·특징 추출에는 [안내된 MAT 파일](data/README.md)이 필요하다. 저장 결과의 수치·분할·모델 무결성은 `.venv/bin/python -m src.verify_results`로 확인한다. 전체 실행 순서는 [실행 안내](docs/PROJECT_GUIDE.md)를 따른다.

## EDA: 발견과 모델링 시사점

| 질문 | 핵심 발견 | 모델 설계에 반영한 점 |
| --- | --- | --- |
| 수명 분포 | 500사이클 미만 비율: Batch1 0%, Batch2 71.79%, Batch3 0%. 1,000 초과 비율: 21.74%, 7.69%, 52.27% | 짧은 수명·긴 수명 구간의 오차를 따로 확인 |
| 열화 곡선·knee | 후기에 감소가 가속되는 셀이 많고, 급격한 감소 시작 시점은 셀마다 다름 | 초기 추세를 후보로 사용. 전체 수명을 본 뒤 알 수 있는 knee는 입력에서 제외 |
| ΔQ(V) | 대표 단수명 셀에서 100−10사이클 곡선 변화가 큼. 로그 분산과 수명의 상관은 배치별 −0.886, −0.902, −0.702 | 로그 ΔQ 분산 하나로 기준 모델 구성 |
| 충전 조건 | C1 단독으로 수명 순서를 설명하기 어려움. 전환 SOC·C2·실험 집단에 따라 차이가 남음 | 충전 조합을 후보 특징과 검증 그룹으로 고려 |
| 상관·데이터 품질 | ΔQ 평균·최솟값·분산의 정보 중복, 충전시간 극단값, IR=0 확인 | 대표 특징부터 시작해 추가·제거 비교. 시간은 중앙값, IR=0은 결측 처리 |

분포 통계는 DAY1의 수명값이 있는 셀 기준이다. DAY2에서는 Batch1의 후속 기록 연결 필요 5셀·실험 미완료 5셀을 추가 제외했다. 그래프와 해석은 [DAY1 보고서](outputs/final/DAY1_REPORT.md)에 함께 제시했다.

## Modeling

### 피처 엔지니어링 전략

공통 전압에서 `ΔQ(V) = Q100(V) − Q10(V)`를 계산했다. 로그 분산으로 시작해 최솟값, 초기 용량, 용량 변화율, 초기 감소 속도를 추가했다. **곡선의 변화와 초기 상태가 서로 보완적인 수명 정보를 줄 것**이라는 가설이었다. IR·온도·충전시간·C-rate도 추가 비교했으나 더 나은 최종 CV 후보를 만들지 못했다.

### 모델 선택 및 근거

- **후보:** 평균 예측·선형회귀를 기준으로, 작은 표본과 중복 특징을 고려한 Ridge·ElasticNet을 비교했다. 이후 특징 제거, 규제 강화, MAPE 가중 학습, IQR·PLS 표현을 비교했다.
- **최종 선택:** 로그 수명 ElasticNet (`alpha=0.001`, `l1_ratio=0.5`). Batch1 프로토콜 CV 평균 MAPE가 가장 낮았다.
- **입력 5개:** 로그 ΔQ 분산, ΔQ 최솟값, 10~20사이클 용량 중앙값, 초기 대비 90~100사이클 용량 변화율, 10~100사이클 감소 속도. 최종 적합에서 분산·감소 속도의 계수는 0이 되어 실제로 3개가 기여했다.
- **전처리:** CV 학습 폴드 안에서만 결측 대체·표준화를 학습했다. 예측을 사이클 단위로 되돌려 평가하며, 후기 정보는 사용하지 않았다.

## 성능 결과

**최종 CV 선택 모델: 프로토콜 분리 ElasticNet.** MAPE는 낮을수록 좋다. Train은 학습 데이터 재예측 오차가 아니라 **CV 검증 평균**이다.

| 구분 | MAPE (%) | 비고 |
| --- | ---: | --- |
| Train (Batch 1 CV) | 5.68 | 개발 29셀, 프로토콜 단위 5-fold |
| Valid (Batch 1 Hold-out) | 8.92 | 별도 7셀 |
| Test (Batch 2) | 37.14 | 39셀 |
| Gap (Train-Valid) | +3.24 | Valid − CV, %p |
| Gap (Valid-Test) | +28.22 | Test − Valid, %p |
| Gap (Target-Test) | +28.04 | Test − 논문 참고값 9.1%, %p |
| Test (Batch 3) | 15.10 | 44셀, 같은 고정 모델 |
| Gap (Batch2-Batch3) | −22.04 | Batch3 − Batch2, %p |
| Gap (Target-Test, Batch3) | +6.00 | Batch3 − 과제 공통 참고값 9.1%, %p |

비교 기준도 함께 남겼다. **내부 검증 최저 모델이 다른 배치에서 가장 좋은 모델은 아니었다.**

| 모델 | 학습 셀 | Batch2 MAPE | Batch3 MAPE |
| --- | ---: | ---: | ---: |
| 최종 CV 선택 ElasticNet | 29 | 37.14% | 15.10% |
| 같은 학습 셀의 분산 1개 선형회귀 | 29 | 28.68% | 12.09% |
| 초기 셀 분리의 분산 1개 선형회귀 | 28 | 26.19% | 11.94% |

![같은 학습 셀에서의 모델 성능과 Batch2 예측 비교](outputs/day2/process_review/protocol_evaluation.png)

왼쪽은 내부 검증 개선이 Batch2 개선으로 이어지지 않은 결과다. 오른쪽에서 대각선 위 점은 실제보다 수명을 길게 예측한 셀이다.

학습 셀이 다른 점수 차이를 분리 방식만의 효과로 해석하지 않는다. Batch3는 수명이 길어 상대오차가 작아지는 영향도 있다. 최종 모델의 MAE는 Batch2 **197.70**, Batch3 **194.63사이클**로 비슷했다.

9.1%는 논문 참고값이며, 사용 파일·학습/테스트 구성이 달라 동일 조건 재현은 아니다. 또한 개발 중 Hold-out·Batch2 결과를 반복 확인했으므로 완전히 독립적인 최종 테스트라는 조건은 충족하지 못했다. 후속 실험은 [비즈니스 부록](outputs/day2/DAY2_BUSINESS_APPENDIX.md)에 분리했다.

## 오류 분석

최종 모델의 Batch2 상대오차 상위 3셀은 **6·29·18번**이다. 실제 수명은 각각 **393·452·449**, 예측은 **704·734·717사이클**로 모두 단수명 과대예측이었다. Batch2의 단수명 28셀 MAPE는 **38.92%**였다. Batch3에서는 학습 최대 수명 1,054를 넘는 17셀을 모두 과소예측했다.

**Batch1에 단수명 사례가 없고 수명 범위가 좁은 점이 주요 원인으로 보인다.** 초기 용량·충전 조건의 배치 차이도 영향을 주었을 가능성이 있다. 특징을 더 늘리는 것보다, 단수명과 다양한 운전 조건의 사례를 학습에 보완하는 방향에서 개선을 기대할 수 있다.

## ESS 도메인 해석

- **활용:** 초기 시험 결과로 정밀 검사가 필요한 셀을 선별하고, 교체·보증 계획을 위한 참고 수명을 제공할 수 있다.
- **오류 비용:** 단수명 과대예측은 교체 지연, 장수명 과소예측은 조기 폐기·활용 손실로 이어질 수 있다. 부록의 보수적 예측은 앞의 오류를 줄였지만 뒤의 오류를 키웠다.
- **실제 적용:** 실험실 셀 데이터를 사용했으므로 ESS의 온도·SOC 범위·부하 변화·달력 노화·팩 내 편차를 반영한 추가 검증이 필요하다. 운영에서는 배치별 입력 분포와 실제 오차를 추적하고, 새 조건의 데이터가 쌓이면 재학습하는 방향이 적절하다.

## 참고문헌

- [Severson et al. (2019)](https://www.nature.com/articles/s41560-019-0356-8). Data-driven prediction of battery cycle life before capacity degradation. *Nature Energy*, 4, 383–391.
- [원논문 공식 코드](https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation): 데이터 처리·분할 비교.
- [Attia, Severson & Witmer (2021)](https://arxiv.org/abs/2101.01885): 초기 곡선 특징·모델 표현 참고.
