# SilverWalk

SilverWalk는 경상남도 진주시 도로 링크를 기준으로 노인보행사고 위험도를 예측하고 지도에서 확인하는 Streamlit 앱 프로젝트입니다.

## 실행

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## VWorld WMTS 설정

배경지도로 브이월드 WMTS를 사용하려면 API key를 환경변수나 Streamlit secrets에 설정한다.

```bash
export VWORLD_API_KEY="발급받은_API_KEY"
streamlit run streamlit_app.py
```

또는 `.streamlit/secrets.toml.example`을 참고해 `.streamlit/secrets.toml`에 저장한다.

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

- `streamlit_app.py`: Streamlit 홈 화면
- `pages/`: 데이터 현황, 지도 미리보기, 예측 결과 화면
- `app/`: Streamlit 실행 보조 코드
- `src/silverwalk_ai/`: 데이터 처리, 공간 분석, 모델링, 예측, 시각화 공통 패키지
- `data/`: 원본/중간/최종 데이터
- `artifacts/`: 모델, 전처리 객체, 예측 결과, 지도 HTML, 리포트
