# SilverWalk

SilverWalk는 서울시 도로 주변 포인트를 기준으로 노인보행사고 위험도를 지도에서 확인하는 Streamlit 앱 프로젝트입니다.

현재 앱은 다음 데이터를 사용합니다.

- 서울시 경계: 온라인 공개 GeoJSON을 앱 실행 시 로딩합니다. 로컬 경계 파일은 필요하지 않습니다.
- 위험도 포인트: `data/original_train_data/seoul_road_points.csv` 로컬 CSV를 사용합니다.

`data/`는 GitHub 저장소에 포함하지 않습니다. 팀원은 저장소를 clone한 뒤 필요한 CSV 파일만 별도로 전달받아 같은 경로에 배치하면 됩니다. Git LFS는 필요하지 않습니다.

## Mac 실행 방법

### 1. 필수 프로그램 설치

Mac에 Git과 Python 3.10 이상이 필요합니다.

Homebrew를 사용하는 경우:

```bash
brew install git python
```

Homebrew가 없다면 아래에서 설치해도 됩니다.

- Git: https://git-scm.com/download/mac
- Python: https://www.python.org/downloads/macos/

설치 확인:

```bash
git --version
python3 --version
```

### 2. 저장소 clone

```bash
git clone https://github.com/chan1945/SilverWalk.git
cd SilverWalk
```

### 3. 가상환경 생성 및 활성화

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

터미널 프롬프트 앞에 `(.venv)`가 보이면 가상환경이 활성화된 상태입니다.

### 4. 패키지 설치

```bash
pip install -r requirements.txt
```

Mac에서 TensorFlow 또는 CUDA 관련 패키지 설치가 실패하고 Streamlit 앱만 실행하면 되는 경우에는 아래 명령으로 앱 실행에 필요한 패키지만 설치할 수 있습니다.

```bash
pip install streamlit streamlit-folium folium geopandas shapely pyogrio fiona pandas numpy openpyxl scikit-learn matplotlib
```

이 경우 MLP 학습 코드는 실행하지 못할 수 있지만, 현재 지도 앱 실행에는 충분합니다.

### 5. 데이터 파일 배치

팀원에게 공유한 `seoul_road_points.csv` 파일을 아래 경로에 둡니다.

```text
data/original_train_data/seoul_road_points.csv
```

폴더가 없다면 먼저 생성합니다.

```bash
mkdir -p data/original_train_data
```

확인:

```bash
ls -lh data/original_train_data/seoul_road_points.csv
```

경계 시각화용 `data/SIGUNGU` 폴더나 Shapefile은 필요하지 않습니다.

### 6. VWorld API Key 설정

VWorld API Key는 선택 사항입니다.

- 키가 있으면 VWorld 배경지도를 사용합니다.
- 키가 없으면 앱이 기본 배경지도로 실행됩니다.

키를 사용할 경우 `.streamlit/secrets.toml` 파일을 만들고 아래처럼 입력합니다.

```bash
mkdir -p .streamlit
nano .streamlit/secrets.toml
```

```toml
[vworld]
api_key = "발급받은_API_KEY"
```

환경변수로 설정해도 됩니다.

```bash
export VWORLD_API_KEY="발급받은_API_KEY"
```

### 7. 앱 실행

지도는 `artifacts/predictions/two_stage_zero_risk_predictions.csv`의 `위험도_actual`과 `최종위험도점수_percent`를 사용합니다. 예측 스크립트는 원본 `위험도 = 0`인 포인트만 대상으로 모델 1의 `사고발생확률_p`와 모델 2의 `조건부위험도_r`을 예측한 뒤 `최종위험도점수 = p * r`을 계산합니다.

`최종위험도점수_percent`는 예측 대상 안에서 최대 `최종위험도점수`를 100%로 두고 환산한 상대 위험도입니다.

지도에는 아래 조건을 모두 만족하는 포인트만 표시합니다.

```text
위험도_actual = 0
최종위험도점수_percent > 10
```

이 파일이 없거나 `최종위험도점수_percent` 컬럼이 없으면 모델 1, 모델 2, 전처리 파일을 준비한 뒤 먼저 두 모델 결합 예측을 실행합니다.

```bash
python scripts/predict/predict_two_stage_zero_risk.py
```

지도 툴팁에 포인트별 개선우선순위를 함께 표시하려면 최종 예측 파일을 만든 뒤 SHAP 기반 추천 파일을 생성합니다.

```bash
python scripts/explain/recommend_improvements.py
```

이 명령은 `위험도_actual = 0 AND 최종위험도점수_percent > 10` 포인트를 대상으로 두 모델에서 공통으로 위험 기여가 큰 개선 가능 feature를 찾고, 포인트별 개선우선순위 1~3개를 `artifacts/recommendations/point_improvement_recommendations.csv`에 저장합니다.

노트북에서 학습한 모델 파일로 최종 예측을 만들었다면 추천 생성도 같은 모델 파일을 지정합니다.

```bash
python scripts/explain/recommend_improvements.py \
  --classifier-model-path artifacts/models/mlp_accident_classifier_notebook.keras \
  --regressor-model-path artifacts/models/mlp_positive_risk_regressor_notebook.keras
```

기본 경로는 아래 파일들을 사용합니다.

- 모델 1: `artifacts/models/mlp_accident_classifier.keras`
- 모델 2: `artifacts/models/mlp_positive_risk_regressor.keras`
- 전처리 객체: `artifacts/preprocessors/original_train_preprocessor.joblib`
- 입력 데이터: `data/original_train_data/seoul_road_points.csv`
- 예측 결과: `artifacts/predictions/two_stage_zero_risk_predictions.csv`
- 개선우선순위 추천 결과: `artifacts/recommendations/point_improvement_recommendations.csv`
- 지도 표시 기준: `위험도_actual = 0 AND 최종위험도점수_percent > 10`

지도 등급은 아래 구간으로 표시합니다.

| 구간 | 등급 |
|---|---|
| `10% 초과 ~ 30% 미만` | 주의 |
| `30% 이상 ~ 50% 미만` | 경고 |
| `50% 이상` | 위험 |

```bash
streamlit run streamlit_app.py
```

브라우저에서 아래 주소로 접속합니다.

```text
http://localhost:8501
```

## 자주 나는 문제

### `streamlit: command not found`

가상환경이 활성화되지 않았거나 패키지 설치가 되지 않은 상태입니다.

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### `FileNotFoundError: seoul_road_points.csv`

위험도 포인트 CSV가 없는 상태입니다. 아래 경로에 파일이 있는지 확인합니다.

```text
data/original_train_data/seoul_road_points.csv
```

### `두 모델 결합 최종 결과 파일이 없습니다`

지도에서 사용할 `최종위험도점수_percent` 파일이 없는 상태입니다. 모델 1, 모델 2, 전처리 객체를 준비한 뒤 아래 명령을 실행합니다.

```bash
python scripts/predict/predict_two_stage_zero_risk.py
```

### 지도가 뜨지 않거나 경계가 표시되지 않음

서울시 경계는 온라인 GeoJSON을 받아옵니다. 인터넷 연결이 필요합니다.

### VWorld 배경지도가 뜨지 않음

VWorld API Key가 없거나 잘못된 경우입니다. 앱은 기본 배경지도로도 실행됩니다.

## Ubuntu 실행 방법

```bash
sudo apt update
sudo apt install -y git python3 python3-venv python3-pip

git clone https://github.com/chan1945/SilverWalk.git
cd SilverWalk

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

mkdir -p data/original_train_data
# 공유받은 seoul_road_points.csv를 data/original_train_data/ 아래에 배치

streamlit run streamlit_app.py
```

## Windows 실행 방법

```powershell
git clone https://github.com/chan1945/SilverWalk.git
cd SilverWalk

py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

New-Item -ItemType Directory -Force data\original_train_data
# 공유받은 seoul_road_points.csv를 data\original_train_data\ 아래에 배치

streamlit run streamlit_app.py
```

PowerShell 실행 정책 때문에 가상환경 활성화가 막히면 아래 명령을 실행한 뒤 다시 활성화합니다.

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\.venv\Scripts\Activate.ps1
```

## 주요 구조

```text
SilverWalk/
├── streamlit_app.py
├── app/
├── src/silverwalk_ai/
├── data/
├── artifacts/
├── configs/
├── scripts/
├── notebooks/
├── tests/
└── docs/
```

- `streamlit_app.py`: Streamlit 앱 진입점
- `app/`: Streamlit 화면 구성 및 실행 보조 코드
- `src/silverwalk_ai/`: 데이터 처리, 모델링, 시각화 공통 패키지
- `data/`: 로컬 데이터 위치. GitHub에는 포함하지 않음
- `artifacts/`: 모델, 전처리 객체, 예측 결과, 리포트
