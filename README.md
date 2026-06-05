# SilverWalk

SilverWalk는 도로를 25m 간격으로 나눈 각 포인트 기준으로 노인보행사고 위험도를 예측하고 지도에서 확인하는 Streamlit 앱 프로젝트입니다.

## 사전 준비

- Python 3.10 이상
- Git
- VWorld API Key

`data/`의 원본 공간 데이터와 학습 데이터는 GitHub 저장소에 포함하지 않습니다. 팀원은 별도로 공유받은 데이터를 프로젝트 루트의 `data/` 아래에 배치해야 합니다.

## Ubuntu 실행 방법

### 1. 필수 패키지 설치

```bash
sudo apt update
sudo apt install -y git python3 python3-venv python3-pip
```

### 2. 저장소 clone

```bash
git clone https://github.com/chan1945/SilverWalk.git
cd SilverWalk
```

### 3. 가상환경 생성 및 의존성 설치

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. VWorld API Key 설정

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
nano .streamlit/secrets.toml
```

`.streamlit/secrets.toml`에 발급받은 키를 입력합니다.

```toml
[vworld]
api_key = "발급받은_API_KEY"
```

### 5. 앱 실행

```bash
streamlit run streamlit_app.py
```

브라우저에서 아래 주소로 접속합니다.

```text
http://localhost:8501
```

## Windows 실행 방법

### 1. 필수 프로그램 설치

- Python 3.10 이상: https://www.python.org/downloads/
- Git for Windows: https://git-scm.com/download/win

### 2. 저장소 clone

```powershell
git clone https://github.com/chan1945/SilverWalk.git
cd SilverWalk
```

### 3. 가상환경 생성 및 의존성 설치

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

PowerShell 실행 정책 때문에 가상환경 활성화가 막히면 아래 명령을 실행한 뒤 다시 활성화합니다.

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\.venv\Scripts\Activate.ps1
```

### 4. VWorld API Key 설정

```powershell
Copy-Item .streamlit\secrets.toml.example .streamlit\secrets.toml
notepad .streamlit\secrets.toml
```

`.streamlit\secrets.toml`에 발급받은 키를 입력합니다.

```toml
[vworld]
api_key = "발급받은_API_KEY"
```

### 5. 앱 실행

```powershell
streamlit run streamlit_app.py
```

브라우저에서 아래 주소로 접속합니다.

```text
http://localhost:8501
```

## 참고: VWorld API Key를 환경변수로 설정하기

`secrets.toml` 대신 환경변수로도 설정할 수 있습니다.

Ubuntu:

```bash
export VWORLD_API_KEY="발급받은_API_KEY"
streamlit run streamlit_app.py
```

Windows PowerShell:

```powershell
$env:VWORLD_API_KEY="발급받은_API_KEY"
streamlit run streamlit_app.py
```

## 주요 구조

```text
SilverWalk/
├── streamlit_app.py
├── pages/
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
- `pages/`: 데이터 현황, 지도 미리보기, 예측 결과 화면
- `app/`: Streamlit 화면 구성 및 실행 보조 코드
- `src/silverwalk_ai/`: 데이터 처리, 공간 분석, 모델링, 예측, 시각화 공통 패키지
- `data/`: 원본/중간/최종 데이터
- `artifacts/`: 모델, 전처리 객체, 예측 결과, 지도 HTML, 리포트
