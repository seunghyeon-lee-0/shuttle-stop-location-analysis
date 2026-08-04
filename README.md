# 원주댄싱카니발 셔틀버스 입지분석 — End-to-End 파이프라인 (백업용 재구성)

원본 raw 데이터(od_YYYYMMDD_1.csv, KIKcd_H.xlsx, 실제 정류장 좌표 등)는 저장소에 포함하지 않는다
(`.gitignore`에서 `data/*.csv`, `data/*.xlsx` 제외). `data/make_sample_data.py`,
`data/generate_candidate_stops.py` 로 동일 스키마의 더미 샘플을 생성해 파이프라인 전체가
실제로 동작하는지 검증했다 (아래 "실행 결과" 참고).

MCLP/PMP/Greedy 코드는 `claude_location_optimization_prompts.md` 에 정의된 스펙(공통 지시사항,
CONFIG dataclass, 함수 구조, 산출물 스펙, 검증 항목)을 따라 작성했다.

## 진행 상태
- [x] (1) 데이터 전처리 — `src/00_preprocessing.py`
- [x] (2) 데이터 csv 본 — `data/make_sample_data.py`, `data/generate_candidate_stops.py` (생성 스크립트만 포함, 산출 csv는 미포함)
- [x] (3) MCLP 코드 — `src/01_mclp_location_selection.py`
- [x] (4) P-Median(PMP) 코드 — `src/02_pmedian_location_selection.py`
- [x] (5) Greedy 중복제거/최종 선정 코드 — `src/04_greedy_candidate_reconciliation.py` (병합: `src/03_merge_location_candidates.py`)
- [x] (6) 최종 후보 데이터셋 csv — `outputs/greedy/greedy_final_selected_stops.csv`

## 폴더 구조
```
shuttle-stop-location-analysis/
├── README.md
├── requirements.txt
├── .gitignore
├── config/location_pipeline.example.yaml
├── data/
│   ├── make_sample_data.py            # OD 원본 스키마 더미 샘플 생성기
│   ├── generate_candidate_stops.py    # 후보정류장(=수요지점) 44개 합성 데이터 생성기 (AHP 4요인 포함)
│   └── *.csv                          # 위 스크립트 실행 결과물 (gitignore, 로컬에만 존재)
├── src/
│   ├── 00_preprocessing.py
│   ├── distance_utils.py              # Haversine/Euclidean 공통 거리 계산
│   ├── validation.py                  # 공통 입력 검증
│   ├── 01_mclp_location_selection.py
│   ├── 02_pmedian_location_selection.py
│   ├── 03_merge_location_candidates.py
│   ├── 04_greedy_candidate_reconciliation.py
│   └── run_location_pipeline.py       # 전체 파이프라인 오케스트레이터
├── tests/                             # pytest 21개, 전부 통과 확인
└── outputs/{mclp,pmedian,merged,greedy,logs}/
```

## 실행 방법

### 전체 파이프라인 (권장)
```bash
cd data && python3 generate_candidate_stops.py   # 합성 후보/수요 데이터 생성 (로컬 전용)
cd ../src
python3 run_location_pipeline.py --config ../config/location_pipeline.example.yaml
```
`--resume` (완료된 단계 재사용), `--start-from {mclp|pmedian|merge|greedy}`,
`--stop-after {...}`, `--dry-run` 옵션 지원.

### 단계별 개별 실행
```bash
cd src
python3 01_mclp_location_selection.py --demand-csv ../data/candidate_stops_sample.csv \
  --candidate-csv ../data/candidate_stops_sample.csv --p 22 --coverage-radius-m 400 \
  --output-dir ../outputs/mclp

python3 02_pmedian_location_selection.py --demand-csv ../data/candidate_stops_sample.csv \
  --candidate-csv ../data/candidate_stops_sample.csv --p 22 --output-dir ../outputs/pmedian

python3 03_merge_location_candidates.py --mclp-csv ../outputs/mclp/mclp_selected_stops.csv \
  --pmedian-csv ../outputs/pmedian/pmedian_selected_stops.csv \
  --original-candidate-csv ../data/candidate_stops_sample.csv --output-dir ../outputs/merged

python3 04_greedy_candidate_reconciliation.py --combined-csv ../outputs/merged/combined_candidate_stops.csv \
  --target-stop-count 22 --min-stop-spacing-m 400 --output-dir ../outputs/greedy
```

### 테스트
```bash
cd tests && python3 -m pytest -q   # 21 passed
```

## 실행 결과 (합성 데이터 기준, 실제 검증 완료)
- 후보/수요 44개(7개 읍면동) → MCLP 22개 선택(coverage_rate=1.0) → P-Median 22개 선택
  (weighted_mean_distance≈77m) → 병합 결과 고유 44개(공통 선정 14~15개) → Greedy 최종
  **22개** 선정, 최종 선택 정류장 간 최소 거리 **415m**(≥400m 요건 충족).
- `pipeline_summary.json` 의 4단계 모두 `SUCCESS`, `final_selected_stop_count: 22` 확인.

## 원본 대비 재구성 범위
- `src/00_preprocessing.py`: `data_진행과정/전처리코드제출.ipynb`, `0923 데이터 _ 1차.ipynb` 등
  원본 노트북에서 확인된 로직을 함수 단위로 재정리. 단, 속도 이상치 제거/요일·주말 변수/외부인 변수/
  추석 제외 로직은 원본 코드가 유실된 상태라 결과보고서(p.13)의 서술을 근거로 복원했다 — 실제 재현 시
  원본 팀이 사용한 정확한 파라미터·컬럼과 대조 검증 필요.
- `src/01~04`(MCLP/PMP/Merge/Greedy): 저장소에 원본 코드가 전혀 남아있지 않아, 결과보고서(p.22~37)의
  수식·제약조건 서술과 `claude_location_optimization_prompts.md` 스펙만을 근거로 새로 구현했다.
- 코드 내 주석은 최소화했다. 위 배경 설명과 가정은 이 README에서 관리한다.

## 실제 데이터 적용 시 확정해야 할 항목
`claude_location_optimization_prompts.md` 10절과 동일. 특히:
- 수요지점 = 후보 정류장인지 (본 프로젝트는 보고서 p.25에 따라 "예"로 가정하고 구현)
- Greedy `priority_weights`, `tie_break_columns` 의 정확한 규칙 (현재 값은 예시)
- MCLP `coverage_radius_m` 400m와 Greedy `min_stop_spacing_m` 400m가 실제로 같은 의도인지
- 읍면동 최소/최대 제약 유무 (지정면 57개 등 원본 후보 수가 매우 편중되어 있었음 — 보고서 p.25)
