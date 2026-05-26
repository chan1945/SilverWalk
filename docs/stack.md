# SilverWalk 기술스택

Streamlit 기반 단일 앱 구조에서 사용할 기술 스택이다.

| 영역 | 기술스택 | 사용 목적 |
| --- | --- | --- |
| 앱 프레임워크 | Streamlit | 데이터 현황, 지도, 예측 결과를 하나의 Python 앱에서 제공 |
| 앱 지도 표시 | streamlit-folium, Folium | Streamlit 화면 안에서 위험도로 지도 시각화 |
| 개발 언어 | Python | 데이터 전처리, 공간분석, 모델 개발, 앱 구현 |
| 공간 데이터 처리 | GeoPandas | 행정구역 경계, 도로 링크, 사고지점, 시설물 위치 데이터 처리 |
| 공간 연산 | Shapely | 도로 링크 버퍼 생성, 교차 여부 판단, 거리 계산 |
| SHP/GeoJSON 입출력 | Pyogrio / Fiona | 공간 데이터 파일 읽기·저장 |
| 데이터 처리 | Pandas | 학습용 테이블 생성, 결측값 처리, CSV/XLSX/Parquet 조회 |
| 수치 연산 | NumPy | 배열 연산, 모델 입력 데이터 처리 |
| 딥러닝 프레임워크 | TensorFlow + Keras | MLP 기반 노인보행사고 위험도 예측 모델 구현 |
| 모델 구조 | MLP | 도로 링크별 정형 데이터를 입력받아 위험도 0~1 출력 |
| 모델 손실 함수 | Binary Crossentropy | 사고 발생 여부를 예측하는 이진 분류 학습 |
| 모델 출력 함수 | Sigmoid | 위험도를 0~1 사이 확률값으로 출력 |
| 모델 저장 | `.keras` / `.h5` | 학습된 Keras 모델 저장 및 재사용 |
| 머신러닝 보조 | scikit-learn | train/test 분리, 정규화, 인코딩, 성능 평가 |
| 모델 평가 지표 | Recall, Precision, F1-score | 위험도로 예측 성능 평가 |
| 지도 데이터 포맷 | GeoJSON | 예측 결과를 지도에 시각화하기 위한 공간 데이터 포맷 |
| 데이터 저장 1차 | CSV, GeoJSON, Parquet | MVP용 파일 기반 저장 |
| 데이터 저장 고도화 | PostgreSQL + PostGIS | 공간 데이터 DB 저장 및 공간 쿼리 처리 |
| 전처리 객체 저장 | `.pkl` | scaler, encoder 등 전처리 객체 저장 |
| 시각화 보조 | Matplotlib | 데이터 확인용 도로 링크 및 분석 결과 시각화 |
| 개발 환경 | Python venv | 파이썬 가상환경 기반 개발 |
| 버전 관리 | Git, GitHub | 코드 관리 및 협업 |
| 배포 선택 | Streamlit Community Cloud / Docker | 앱 배포와 실행 환경 통일 |
| 프로젝트 문서화 | README.md, Markdown | 설치 방법, 데이터 처리 절차, 구조 설명 |
