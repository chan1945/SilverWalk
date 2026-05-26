from __future__ import annotations

import streamlit as st

from app.bootstrap import add_src_to_path

add_src_to_path()

from silverwalk_ai.data.loaders import GEO_EXTENSIONS, TABLE_EXTENSIONS, list_files, read_table
from silverwalk_ai.data.paths import PATHS


st.set_page_config(page_title="SilverWalk 예측", page_icon="SW", layout="wide")
st.title("예측 결과")

result_files = list_files(PATHS.predictions, TABLE_EXTENSIONS | GEO_EXTENSIONS)
result_files += list_files(PATHS.outputs, TABLE_EXTENSIONS | GEO_EXTENSIONS)

if not result_files:
    st.info("아직 예측 결과 파일이 없습니다. `artifacts/predictions/` 또는 `data/outputs/`에 CSV/GeoJSON을 저장하면 여기에서 확인할 수 있습니다.")
    st.stop()

selected = st.selectbox(
    "결과 파일",
    result_files,
    format_func=lambda path: str(path.relative_to(PATHS.root)),
)

suffix = selected.suffix.lower()

if suffix in TABLE_EXTENSIONS:
    try:
        frame = read_table(selected)
    except Exception as exc:
        st.error(f"결과 파일을 읽지 못했습니다: {exc}")
        st.stop()

    st.write(f"{len(frame):,}행, {len(frame.columns):,}열")

    numeric_columns = frame.select_dtypes("number").columns.tolist()
    risk_candidates = [column for column in numeric_columns if "risk" in column.lower() or "score" in column.lower()]
    selected_risk = st.selectbox("위험도 컬럼", ["선택 안 함", *risk_candidates, *[c for c in numeric_columns if c not in risk_candidates]])

    if selected_risk != "선택 안 함":
        high_risk = frame[frame[selected_risk] >= 0.5]
        metric_a, metric_b, metric_c = st.columns(3)
        metric_a.metric("전체 링크", f"{len(frame):,}")
        metric_b.metric("위험도 0.5 이상", f"{len(high_risk):,}")
        metric_c.metric("평균 위험도", f"{frame[selected_risk].mean():.3f}")
        st.bar_chart(frame[selected_risk])

    st.dataframe(frame.head(500), width="stretch")
else:
    st.write("공간 예측 결과는 지도 미리보기 페이지에서 확인할 수 있습니다.")
    st.code(f"선택 파일: {selected.relative_to(PATHS.root)}", language="text")
