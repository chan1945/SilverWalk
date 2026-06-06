# SilverWalk

SilverWalk는 서울시 도로 주변 포인트를 기준으로 노인보행사고 위험도를 지도에서 확인하는 Streamlit 앱 프로젝트입니다.

현재 앱은 다음 데이터를 사용합니다.

- 서울시 경계: 온라인 공개 GeoJSON을 앱 실행 시 로딩합니다. 로컬 경계 파일은 필요하지 않습니다.
- 위험도 포인트: `artifacts/predictions/two_stage_zero_risk_predictions.csv`를 사용합니다.
- 개선우선순위: `artifacts/recommendations/point_improvement_recommendations.csv`를 사용합니다.

`data/`의 원본 학습 데이터, 모델 파일, 전처리 객체는 GitHub 저장소에 포함하지 않습니다. 배포 앱은 이미 생성된 최종 예측 CSV와 개선우선순위 CSV만 읽어서 시각화합니다. Git LFS는 필요하지 않습니다.

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

앱 실행만 할 때는 배포용 의존성을 설치합니다.

```bash
pip install -r requirements.txt
```

모델 학습, 최종 예측 생성, SHAP 개선우선순위 생성을 실행할 때는 학습용 의존성을 추가로 설치합니다.

```bash
pip install -r requirements-train.txt
```

Streamlit Cloud 배포 환경에서는 `requirements.txt`만 사용합니다.

### 5. 결과 파일 확인

앱 실행에는 아래 두 결과 파일이 필요합니다. 이 두 파일은 배포용 결과물이므로 GitHub에 포함합니다.

```text
artifacts/predictions/two_stage_zero_risk_predictions.csv
artifacts/recommendations/point_improvement_recommendations.csv
```

확인:

```bash
ls -lh artifacts/predictions/two_stage_zero_risk_predictions.csv
ls -lh artifacts/recommendations/point_improvement_recommendations.csv
```

원본 학습 데이터는 모델을 다시 학습하거나 예측 결과를 다시 만들 때만 필요합니다.

```text
data/original_train_data/seoul_road_points.csv
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

앱은 아래 파일들을 읽어서 표시합니다.

- 예측 결과: `artifacts/predictions/two_stage_zero_risk_predictions.csv`
- 개선우선순위 추천 결과: `artifacts/recommendations/point_improvement_recommendations.csv`
- 지도 표시 기준: `위험도_actual = 0 AND 최종위험도점수_percent > 10`

지도 등급은 아래 구간으로 표시합니다.

| 구간 | 등급 |
|---|---|
| `10% 초과 ~ 30% 미만` | 주의 |
| `30% 이상 ~ 50% 미만` | 경고 |
| `50% 이상` | 위험 |

지도에서 포인트를 선택하면 왼쪽 사이드바에 선택 포인트의 `POINT_ID`, 위험도 정보, 개선우선순위 1~3개를 표시합니다.

```bash
streamlit run streamlit_app.py
```

브라우저에서 아래 주소로 접속합니다.

```text
http://localhost:8501
```

## Streamlit Cloud 배포 방법

### 1. 배포 전 확인

아래 파일들이 GitHub에 포함되어 있어야 합니다.

```text
streamlit_app.py
requirements.txt
runtime.txt
artifacts/predictions/two_stage_zero_risk_predictions.csv
artifacts/recommendations/point_improvement_recommendations.csv
```

아래 파일들은 GitHub에 올리지 않습니다.

```text
.streamlit/secrets.toml
data/
artifacts/models/
artifacts/preprocessors/
artifacts/reports/
```

### 2. GitHub에 push

```bash
git add streamlit_app.py app src .streamlit/config.toml .streamlit/secrets.toml.example
git add requirements.txt requirements-train.txt runtime.txt README.md .gitignore
git add artifacts/predictions/two_stage_zero_risk_predictions.csv
git add artifacts/recommendations/point_improvement_recommendations.csv
git commit -m "Prepare Streamlit deployment"
git push
```

### 3. Streamlit Cloud에서 앱 생성

Streamlit Community Cloud에서 새 앱을 만들고 아래처럼 설정합니다.

```text
Repository: GitHub의 SilverWalk 저장소
Branch: main
Main file path: streamlit_app.py
```

### 4. Secrets 설정

VWorld 배경지도를 사용하려면 Streamlit Cloud의 Secrets에 아래 내용을 입력합니다.

```toml
[vworld]
api_key = "발급받은_VWORLD_API_KEY"
```

VWorld 키가 없어도 앱은 기본 배경지도로 실행됩니다.

## 결과 파일 재생성 방법

모델을 다시 학습하고 최종 결과 파일을 다시 만들 때만 아래 절차를 실행합니다.

```bash
pip install -r requirements-train.txt

python scripts/data/preprocess_original_train_data.py
python scripts/train/train_accident_classifier.py
python scripts/train/train_positive_risk_regressor.py
python scripts/predict/predict_two_stage_zero_risk.py
python scripts/explain/recommend_improvements.py
```

노트북에서 학습한 모델 파일로 최종 예측을 만들었다면 예측과 추천 생성에 같은 모델 파일을 지정합니다.

```bash
python scripts/predict/predict_two_stage_zero_risk.py \
  --classifier-model-path artifacts/models/mlp_accident_classifier_notebook.keras \
  --regressor-model-path artifacts/models/mlp_positive_risk_regressor_notebook.keras

python scripts/explain/recommend_improvements.py \
  --classifier-model-path artifacts/models/mlp_accident_classifier_notebook.keras \
  --regressor-model-path artifacts/models/mlp_positive_risk_regressor_notebook.keras
```

## 자주 나는 문제

### `streamlit: command not found`

가상환경이 활성화되지 않았거나 패키지 설치가 되지 않은 상태입니다.

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### `두 모델 결합 최종 결과 파일이 없습니다`

지도에서 사용할 최종 예측 CSV가 없는 상태입니다. 아래 경로에 파일이 있는지 확인합니다.

```text
artifacts/predictions/two_stage_zero_risk_predictions.csv
```

### 개선우선순위가 표시되지 않음

추천 CSV가 없는 상태입니다. 아래 경로에 파일이 있는지 확인합니다.

```text
artifacts/recommendations/point_improvement_recommendations.csv
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
- `artifacts/`: 모델, 전처리 객체, 예측 결과, 리포트. GitHub에는 배포용 최종 예측 CSV와 개선우선순위 CSV만 포함
