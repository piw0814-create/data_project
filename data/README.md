# 원본 데이터

저장소에는 대용량 MAT 원본을 포함하지 않는다. 수업에서 제공받은 아래 파일을 이 폴더에 두면 원본 EDA와 특징 추출을 실행할 수 있다.

| 배치 | 파일 |
| --- | --- |
| Batch1 | `2017-05-12_batchdata_updated_struct_errorcorrect.mat` |
| Batch2 | `2018-02-20_batchdata_updated_struct_errorcorrect.mat` |
| Batch3 | `2018-04-12_batchdata_updated_struct_errorcorrect.mat` |

파일 날짜와 내용이 다른 배치를 임의로 대체하지 않는다. 로컬 Batch2와 원논문 공개 처리 코드의 두 번째 파일은 날짜가 다르므로, 이번 결과는 논문의 동일 조건 재현이 아니다.

저장된 결과의 수치 검산과 보고서 확인에는 MAT 파일이 필요하지 않다. 사용한 셀의 레이블 점검·제외 기준은 [label_audit.csv](../outputs/day2/label_audit.csv)에 있다.
