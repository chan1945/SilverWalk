# SilverWalk Streamlit 프로젝트 구조

이 문서는 Streamlit 기반으로 정리한 SilverWalk 프로젝트 구조를 설명한다.
SilverWalk는 노인보행사고 위험도로 예측 및 시각화 AI 시스템으로, 데이터 처리, 공간 분석, 모델 학습, 예측 결과 확인, 지도 시각화를 하나의 Streamlit 앱에서 제공한다.

## 최상위 구조

```text
SilverWalk/
├── .streamlit/
├── app/
├── artifacts/
├── configs/
├── data/
├── docs/
├── notebooks/
├── pages/
├── scripts/
├── src/
├── tests/
├── streamlit_app.py
├── requirements.txt
└── pyproject.toml
```

| 경로 | 역할 |
| --- | --- |
| `streamlit_app.py` | Streamlit 앱의 홈 화면이자 실행 진입점이다. |
| `pages/` | Streamlit 멀티페이지 화면을 관리한다. 데이터 현황, 지도 미리보기, 예측 결과 화면을 둔다. |
| `app/` | Streamlit 실행에 필요한 보조 코드와 로컬 `src` 패키지 로딩 설정을 둔다. |
| `.streamlit/` | Streamlit 서버와 테마 설정을 저장한다. |
| `src/silverwalk_ai/` | 데이터 처리, 공간 연산, 피처 생성, 모델링, 예측, 시각화의 핵심 Python 패키지다. |
| `data/` | 원본 데이터, 중간 처리 데이터, 최종 처리 데이터, 사용자 산출 데이터를 저장한다. |
| `artifacts/` | 모델, 전처리 객체, 예측 결과, 지도 HTML, 리포트 등 재생성 가능한 산출물을 저장한다. |
| `configs/` | 데이터 처리와 모델 학습에 필요한 설정 파일을 저장한다. |
| `scripts/` | 반복 실행하는 데이터 처리, 학습, 산출물 생성 작업을 저장한다. |
| `notebooks/` | 데이터 탐색, 공간 분석 검증, 모델 실험용 노트북을 둔다. |
| `tests/` | Python 패키지와 Streamlit 보조 로직을 검증하는 테스트를 둔다. |
| `docs/` | 요구사항, 기술스택, 프로젝트 구조 문서를 관리한다. |

## Streamlit 화면

| 파일 | 역할 |
| --- | --- |
| `streamlit_app.py` | 프로젝트 개요, 주요 경로, 실행 방법을 보여준다. |
| `pages/01_data_overview.py` | 원본/처리/산출 데이터 파일 현황을 보여주고 테이블 파일을 미리본다. |
| `pages/02_map_preview.py` | SHP, GeoJSON 공간 데이터를 Folium 지도에서 확인한다. |
| `pages/03_prediction_results.py` | 예측 결과 CSV, Parquet, GeoJSON 파일을 조회하고 위험도 컬럼 요약을 표시한다. |

## 핵심 패키지

| 폴더 | 역할 |
| --- | --- |
| `src/silverwalk_ai/data/` | 경로 정의, 파일 탐색, CSV/XLSX/Parquet 로딩 등 데이터 접근 로직을 둔다. |
| `src/silverwalk_ai/spatial/` | GeoPandas, Shapely 기반 공간 연산을 구현한다. |
| `src/silverwalk_ai/features/` | 도로 링크별 사고 라벨, 주변 시설, 지역 특성 변수를 생성한다. |
| `src/silverwalk_ai/modeling/` | Keras MLP 모델 정의, 학습, 평가 로직을 구현한다. |
| `src/silverwalk_ai/prediction/` | 학습 모델과 전처리 객체를 사용해 도로 링크별 위험도를 예측한다. |
| `src/silverwalk_ai/visualization/` | Folium 지도 생성, 위험도 색상 규칙, 분석 시각화 로직을 둔다. |
| `src/silverwalk_ai/utils/` | 공통 유틸리티, 로깅, 설정 로딩 등을 관리한다. |

## 데이터와 산출물

| 경로 | 역할 |
| --- | --- |
| `data/NODELINKDATA/` | 도로 노드·링크 원본 SHP 데이터를 보관한다. |
| `data/SIGUNGU/` | 시군구 행정구역 원본 데이터를 보관한다. |
| `data/raw/` | 새로 수집한 원본 CSV, SHP, GeoJSON 파일을 저장한다. |
| `data/external/` | 외부 기관에서 받은 참조 데이터나 보조 데이터를 저장한다. |
| `data/interim/` | 좌표계 변환, 서울시 필터링 등 중간 처리 결과를 저장한다. |
| `data/processed/` | 모델 학습과 예측에 바로 사용할 수 있는 최종 전처리 데이터를 저장한다. |
| `data/outputs/` | 사용자에게 제공할 분석 결과 CSV, 예측 결과 GeoJSON 등을 저장한다. |
| `artifacts/models/` | 학습된 Keras 모델 파일을 저장한다. |
| `artifacts/preprocessors/` | scaler, encoder 등 전처리 객체를 저장한다. |
| `artifacts/predictions/` | 도로 링크별 위험도 예측 결과를 저장한다. |
| `artifacts/maps/` | Folium으로 생성한 지도 HTML 파일을 저장한다. |
| `artifacts/reports/` | 모델 평가 지표와 실험 리포트를 저장한다. |

## 관리 규칙

- 사용자 화면은 Streamlit 멀티페이지 구조인 `streamlit_app.py`와 `pages/`에서 관리한다.
- 공통 분석 로직은 화면 파일에 길게 두지 않고 `src/silverwalk_ai/` 패키지로 분리한다.
- 원본 데이터는 `data/NODELINKDATA/`, `data/SIGUNGU/`, `data/raw/`에 보관하고 직접 수정하지 않는다.
- 재생성 가능한 모델, 지도, 예측 결과는 `artifacts/` 또는 `data/outputs/`에 저장한다.
- 일회성 실행 흐름은 `scripts/`에 두고, 재사용 가능한 로직은 Python 모듈로 옮긴다.
