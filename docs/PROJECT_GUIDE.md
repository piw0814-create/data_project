# 실행 안내와 과제 기준 점검

[프로젝트 요약](../README.md)에서 결과를 먼저 확인한 뒤 필요한 노트북만 열면 된다.

## 실행 파일

| 순서 | 노트북 | 내용 |
| --- | --- | --- |
| 1 | [30 — DAY1 EDA](../notebooks/30-ESSHealth-DAY1-EDA.ipynb) | Batch1 단독 탐색에서 세 배치 비교까지 |
| 2 | [40 — 기준 모델](../notebooks/40-ESSHealth-DAY2.ipynb) | 기존 셀 분리, CV, Hold-out, Batch2 평가 |
| 3 | [41 — 특징 추가](../notebooks/41-ESSHealth-DAY2-feature-experiments.ipynb) | 누적 특징·모델 설정 비교 |
| 4 | [42 — 특징 제거·손실](../notebooks/42-ESSHealth-DAY2-ablation-loss.ipynb) | 중복 특징과 MAPE 가중 학습 |
| 5 | [43 — 규제](../notebooks/43-ESSHealth-DAY2-capacity-penalty.ipynb) | 초기 용량 의존 억제 |
| 6 | [44 — 곡선 표현](../notebooks/44-ESSHealth-DAY2-curve-models.ipynb) | 로그 IQR·PLS 비교 |
| 7 | [45 — 프로토콜 검증](../notebooks/45-ESSHealth-DAY2-protocol-validation.ipynb) | 충전 조합을 분리한 재검증 |
| 8 | [46 — Batch3 추가 테스트](../notebooks/46-ESSHealth-DAY2-batch3-test.ipynb) | 현재 고정 모델과 분산 기준 모델의 추가 평가 |
| 부록 | [47 — 비즈니스 목적 추가 실험](../notebooks/47-ESSHealth-DAY2-business-appendix.ipynb) | 저장 예측에서 지표·그래프를 재계산하고 단수명 과대예측과 장수명 예측의 상충 관계 해석 |

노트북에는 실행 결과가 남아 있다. 모델 구현은 `src/`, 근거 데이터와 모델은 `outputs/day2/`에 있다. 기존 셀 분리 후속 실험은 `feature_experiments/split_manifest.csv`를 공유한다. 프로토콜 실험은 별도 분할표를 사용한다. `label_audit.csv`는 셀 제외 기준을 담은 필수 입력이다. 부록 노트북 47은 원본 MAT나 폐기한 실험 학습 코드 없이 저장 결과를 분석하며, 새 학습·예측을 수행하지 않는다.

## 환경과 데이터 준비

Python **3.11.15**. 라이브러리 버전은 `requirements.txt`에 고정했다. 아래 명령은 저장소를 복제하거나 압축을 푼 프로젝트 루트에서 실행한다. macOS·Linux 기준이다.

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user --name data-project --display-name "Python (data-project)"
.venv/bin/python -m jupyterlab
```

학습·원본 EDA를 실행하려면 제공받은 다음 파일을 `data/`에 둔다. 대용량 원본은 GitHub와 제출 압축에 포함하지 않았다. 파일 배치 안내는 [data/README.md](../data/README.md)에 있다.

- `2017-05-12_batchdata_updated_struct_errorcorrect.mat` — Batch1
- `2018-02-20_batchdata_updated_struct_errorcorrect.mat` — Batch2
- `2018-04-12_batchdata_updated_struct_errorcorrect.mat` — Batch3

설치 후 저장된 예측값·평가표·분할·모델 해시를 검사하려면 다음을 실행한다. 재학습하거나 파일을 수정하지 않는다.

```bash
.venv/bin/python -m src.verify_results
```

저장 결과만으로 DAY2 문서와 그림을 재생성하려면 다음을 실행한다. 재학습하지 않으며 원본 MAT가 필요하지 않다.

```bash
.venv/bin/python -m src.day2_process_review
```

실험은 위 노트북 순서로 확인한다. 노트북 40은 저장 결과가 있으면 특징 추출·학습·평가를 반복하지 않는다. 새 학습 시에는 노트북 40의 마지막 Batch2 평가까지 완료한 뒤 41로 넘어간다. `src.run_day2`만 실행하면 새 실험에서는 CV와 Hold-out까지만 수행하므로, 후속 실험 전체를 실행한 것과 다르다.

Batch3 추가 테스트는 `.venv/bin/python -m src.day2_batch3`로 실행한다. 원본이 필요한 최초 평가에서는 고정 모델 2개만 사용하며, 완료 결과가 있으면 원본 접근·재학습·재평가 없이 결과를 읽는다. 기존 26.19% 모델의 추가 확인은 `src.day2_batch3_original`이며, 최초 두 모델 평가 후 요청에 따른 별도 기록이다.

후속 구현은 `src.day2_experiments`, `src.day2_ablation`, `src.day2_capacity_penalty`, `src.day2_curve_models`, `src.day2_protocol_validation`에 있다. 완료된 결과를 재사용하고, 중단된 결과가 있으면 덮어쓰기 전에 확인하도록 한다. 새 실험은 별도 프로젝트 사본에서 해당 실험 결과 폴더를 비운 뒤 수행한다. `label_audit.csv`와 이전 단계의 입력·분할표는 유지한다. 기록된 코드 해시는 당시 학습 버전의 기록이며 이번 공유용 정리 이후 코드와 다를 수 있다.


## 과제 기준 점검

| 평가 항목 | 반영 내용·확인 위치 | 상태 |
| --- | --- | --- |
| EDA: 분포·통계·해석 | [DAY1](../outputs/final/DAY1_REPORT.md) §2–6: 세 배치 비교, 장단수명 비율, 대표 셀, 상관·중복 특징 | 반영 |
| EDA → 모델 전략 | DAY1 §7–9, [DAY2](../outputs/day2/DAY2_REPORT.md) §2·4: 발견 → 가설 → 특징·모델 → 결과 | 반영 |
| 특징 및 모델 구현 | 노트북 40–45, `src/`: 분산 기준 모델, 특징 추가·제거, 규제·손실·곡선 표현 비교 | 반영 |
| Batch1 학습 / Batch2 평가 | 최종 프로토콜 분할: Batch1 개발 29셀 / Hold-out 7셀, Batch2 39셀 | 반영 |
| CV와 Hold-out 구분 | 개발 29셀에서 5-fold GroupKFold; 같은 충전 조합은 같은 그룹. 별도 Hold-out은 후보 선택에 사용하지 않음 | 현재 실행에 반영 |
| 지정 성능표·논문 Gap | DAY2 §6: CV·Hold-out·Batch2 MAPE, 3개 Gap; 9.1%와 비교 | 반영 |
| Batch3 추가 평가 | [Batch3 보고서](../outputs/day2/DAY2_BATCH3_TEST.md): 44셀, 고정 모델 평가·Gap | 반영 |
| 도메인 해석·한계 | README ESS 해석, DAY2 §7–9, [비즈니스 부록](../outputs/day2/DAY2_BUSINESS_APPENDIX.md) | 반영 |
| Knee 탐색 | 전체 열화 곡선에서 급격한 감소 구간을 육안으로 탐색. 정밀 시점 추정은 하지 않음 | 정성 분석 |
| 충전 패턴과 열화 | C1·전환 SOC·C2와 수명·열화 속도 분석. 실제 전류 파형의 직접 분석은 하지 않음 | 정책 요약값으로 분석 |

## Final Reminders 점검

- **재현성:** 무작위 분할·해당 모델의 seed는 42. GroupKFold는 섞지 않는 결정론적 분할이다. Python·라이브러리 버전, 셀 분할표, 입력·모델 해시를 보존했다.
- **분할 후 전처리:** 각 셀의 초기 특징을 계산한 뒤 셀·프로토콜을 분리한다. 셀별 특징 계산에는 다른 셀의 통계가 필요하지 않으며, 결측 대체·표준화·PLS 등의 학습은 각 CV 학습 폴드 안에서만 수행한다.
- **미래 정보 차단:** 한 셀의 100사이클까지를 입력으로, 총 수명을 정답으로 사용한다. 실제 knee·후기 감소 속도는 입력에 넣지 않는다. 셀마다 한 행인 회귀이므로 한 셀의 시간 기록을 무작위로 나누는 방식과 다르다.
- **정직한 평가:** 각 실행의 후보 선택은 Batch1 CV를 기준으로 했지만, 전체 개발 과정에서는 Hold-out·Batch2 결과를 여러 번 확인했다. 최종 추천은 외부 배치의 관측 성능과 단순성을 종합해 `S0_linear`로 정한 사후 판단이다. 모델 경로와 성능표는 `outputs/day2/final_recommendation/`에 있으며 당시 CV 선택 기록은 유지했다. 따라서 **‘테스트를 마지막에 한 번만 사용’ 조건은 충족하지 못했다.** 이 이력은 보고서에도 적었다. 새 독립 배치를 확보하면 최종 확인용으로 분리하는 것이 좋다.
- **분포와 과적합:** 큰 오차를 이유로 테스트 셀을 삭제하지 않았다. MAPE와 MAE·RMSE·R², 수명 구간별 오차를 함께 확인했다. 회귀 과제이므로 분류의 Accuracy·F1·클래스 재표본추출은 적용하지 않았다.
- **운영 모니터링:** 실 BESS의 Live Test는 수행하지 않았다. 운영 시에는 초기 입력 분포와 수명 확인 후의 오차를 배치별로 확인하고, 새 조건·단수명 사례가 누적되면 재학습하는 방향을 제안한다.

## 파일 보존과 제출

보고서는 Markdown이 최신본이다. 실험별 입력·분할·설정·모델·예측은 수치 검증에 필요하므로 보존한다. 추가 탐색은 노트북 47과 비즈니스 부록에서 핵심만 확인할 수 있다. 이 부록은 저장 결과 재계산용이며, 폐기한 추가 실험의 학습 코드는 포함하지 않는다.

```bash
.venv/bin/python -m src.package_submission
```

`submission.zip`에 보고서·그림·노트북·코드·환경 설정·근거 기록과 `CONTENTS.sha256`을 묶는다. 대용량 MAT 원본, 가상환경, 캐시, 읽기 전용 강의 자료, Git 이력은 제외한다. 원본 데이터 안내는 [data/README.md](../data/README.md)에 있다.
