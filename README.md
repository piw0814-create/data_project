# 배터리 수명 예측 — DAY1·DAY2

초기 100사이클의 측정값으로 셀의 총 수명 `cycle_life`를 예측하는 회귀 과제다. Batch1으로 학습하고 Batch2에서 평가했다. Batch3는 EDA에 사용했으며 선택 사항인 모델 추가 평가는 수행하지 않았다.

GitHub에서는 아래 보고서 링크부터 읽으면 된다. 그래프·평가표·실행 결과를 저장해 두었으므로 **내용 확인에는 원본 데이터나 설치가 필요 없다.**

## 읽는 순서

1. [DAY1 — EDA와 모델 설계](outputs/final/DAY1_REPORT.md): 다섯 질문에 대한 그래프·해석·모델링 시사점.
2. [DAY2 — 모델 개발 및 평가](outputs/day2/DAY2_REPORT.md): 도메인 가설, 특징 개발, 결과 비교, 프로토콜 분리 검증, 지정 성능표와 결론.
3. [DAY2 부록 — 프로토콜 분리 상세 결과](outputs/day2/DAY2_PROTOCOL_VALIDATION.md): 후보·설정·집단별 오차 확인용.

**주요 결과:** 기존 셀 분리에서 관측된 Batch2 최저 MAPE는 분산 1개 선형회귀의 **26.19%**다. 해당 개발 과정의 CV 선택 모델은 MAPE 가중 회귀로 Batch2 **33.88%**였다. 프로토콜 분리 추가 검증에서는 CV 선택 ElasticNet이 **37.14%**, 같은 학습 셀의 분산 기준 모델이 **28.68%**였다. 프로토콜 분리로 성능이 개선됐다고 해석하지 않는다.

Batch2를 후속 개발에서 반복 확인했으므로 새 독립 테스트라고 주장하지 않는다. 9.1%는 과제의 논문 참고값이며 데이터·분할 조건이 달라 동일 조건 재현이 아니다.

## 과제 요구사항과 위치

| 요구사항 | 확인 위치 |
| --- | --- |
| 세 배치 수명 분포·장단수명 비율·짧은 셀 비교 | DAY1 §2 |
| 열화 곡선·가속·knee 탐색 | DAY1 §3 — knee는 육안 탐색 |
| ΔQ100−10 비교·통계 특징 추출 | DAY1 §4 |
| 충전 조건·수명·열화 속도의 관계 | DAY1 §5 — C1·SOC·C2 기반, 실제 전류 파형 분석은 미실행 |
| 상관관계·다중공선성 | DAY1 §6 |
| Feature Engineering·회귀 선택·모델 전략 | DAY1 §7–9, DAY2 §2·4 |
| Batch1 학습·CV·Hold-out, Batch2 평가 | DAY2 §3–6 |
| MAPE·세 가지 Gap·논문 기준 비교 | DAY2 §6 — 두 분리 방식 모두 지정 형식으로 보고 |
| 도메인 근거·결과 분석·한계 | DAY2 §2·4·7·8 |
| 재현성·전처리 누수 방지 | DAY2 §3, 각 실험의 설정·분할·환경 기록 |

## 실행 파일

| 순서 | 노트북 | 내용 |
| --- | --- | --- |
| 1 | [30 — DAY1 EDA](notebooks/30-ESSHealth-DAY1-EDA.ipynb) | Batch1 단독 탐색에서 세 배치 비교까지 |
| 2 | [40 — 기준 모델](notebooks/40-ESSHealth-DAY2.ipynb) | 기존 셀 분리, CV, Hold-out, Batch2 평가 |
| 3 | [41 — 특징 추가](notebooks/41-ESSHealth-DAY2-feature-experiments.ipynb) | 누적 특징·모델 설정 비교 |
| 4 | [42 — 특징 제거·손실](notebooks/42-ESSHealth-DAY2-ablation-loss.ipynb) | 중복 특징과 MAPE 가중 학습 |
| 5 | [43 — 규제](notebooks/43-ESSHealth-DAY2-capacity-penalty.ipynb) | 초기 용량 의존 억제 |
| 6 | [44 — 곡선 표현](notebooks/44-ESSHealth-DAY2-curve-models.ipynb) | 로그 IQR·PLS 비교 |
| 7 | [45 — 프로토콜 검증](notebooks/45-ESSHealth-DAY2-protocol-validation.ipynb) | 충전 조합을 분리한 재검증 |

노트북에는 실행 결과가 남아 있다. 모델 구현은 `src/`, 근거 데이터와 모델은 `outputs/day2/`에 있다. 기존 셀 분리 후속 실험은 `feature_experiments/split_manifest.csv`를 공유한다. 프로토콜 실험은 별도 분할표를 사용한다. `label_audit.csv`는 셀 제외 기준을 담은 필수 입력이다.

## 환경과 데이터 준비

Python **3.11.15**. 라이브러리 버전은 `requirements.txt`에 고정했다. 아래 명령은 저장소를 복제하거나 압축을 푼 프로젝트 루트에서 실행한다. macOS·Linux 기준이다.

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user --name data-project --display-name "Python (data-project)"
.venv/bin/python -m jupyterlab
```

학습·원본 EDA를 실행하려면 제공받은 다음 파일을 `data/`에 둔다. 대용량 원본은 GitHub와 제출 압축에 포함하지 않았다. 파일 배치 안내는 [data/README.md](data/README.md)에 있다.

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

후속 구현은 `src.day2_experiments`, `src.day2_ablation`, `src.day2_capacity_penalty`, `src.day2_curve_models`, `src.day2_protocol_validation`에 있다. 완료된 결과를 재사용하고, 중단된 결과가 있으면 덮어쓰기 전에 확인하도록 한다. 새 실험은 별도 프로젝트 사본에서 해당 실험 결과 폴더를 비운 뒤 수행한다. `label_audit.csv`와 이전 단계의 입력·분할표는 유지한다. 기록된 코드 해시는 당시 학습 버전의 기록이며 이번 공유용 정리 이후 코드와 다를 수 있다.

## 폴더 구성

| 폴더 | 내용 |
| --- | --- |
| `notebooks/` | 단계별 설명과 실행 결과를 포함한 7개 노트북 |
| `src/` | 특징 추출, 학습·평가, 보고서 생성, 결과 검산 코드 |
| `outputs/final/` | DAY1 보고서와 주요 그래프 |
| `outputs/day2/` | DAY2 보고서, 실험별 설정·분할·예측·평가표·학습 모델 |
| `data/` | 원본 MAT를 둘 위치와 안내; 원본은 별도 제공 |

개인 경로, 가상환경, 캐시, 강의 참고자료(`sources/`), 제출 ZIP은 GitHub 공유 대상에서 제외한다. 모델 파일은 이 저장소의 저장 결과 재현용이다.

## 제출 구성

`submission.zip`에는 이 README, DAY1·DAY2 보고서와 그림, 노트북, 구현 코드, 환경 설정, 셀별 입력·분할·예측·모델·평가 기록이 포함된다. 압축 안의 `CONTENTS.sha256`은 포함 파일의 무결성 확인용 목록이다.

원본 데이터, `.venv`, `.git`, `sources`, 캐시, 이전 DAY1 PDF는 제외했다. **제출 문서의 최신본은 Markdown 파일**이며 기존 PDF는 이번 개정 내용이 반영되지 않은 이전 파일이다. PDF로 내보낼 경우 최신 Markdown을 사용한다.

```bash
.venv/bin/python -m src.package_submission
```

이 명령은 제출 파일만 묶으며, 모델이나 평가 결과를 변경하지 않는다.
