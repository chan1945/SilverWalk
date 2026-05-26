from __future__ import annotations

import streamlit as st

from app.bootstrap import add_src_to_path

add_src_to_path()

from silverwalk_ai.data.loaders import GEO_EXTENSIONS, TABLE_EXTENSIONS, list_files, read_table
from silverwalk_ai.data.paths import PATHS


st.set_page_config(page_title="SilverWalk 데이터", page_icon="SW", layout="wide")
st.title("데이터 현황")

targets = {
    "도로 노드/링크 원본": PATHS.nodelink,
    "시군구 원본": PATHS.sigungu,
    "처리 데이터": PATHS.processed,
    "분석 산출물": PATHS.outputs,
    "예측 산출물": PATHS.predictions,
}

summary = []
for label, directory in targets.items():
    files = list_files(directory)
    summary.append(
        {
            "구분": label,
            "경로": str(directory.relative_to(PATHS.root)),
            "파일 수": len(files),
            "공간 파일": len([path for path in files if path.suffix.lower() in GEO_EXTENSIONS]),
            "테이블 파일": len([path for path in files if path.suffix.lower() in TABLE_EXTENSIONS]),
        }
    )

st.dataframe(summary, width="stretch", hide_index=True)

st.subheader("파일 미리보기")
all_preview_files = []
for directory in targets.values():
    all_preview_files.extend(list_files(directory, TABLE_EXTENSIONS))

if not all_preview_files:
    st.info("미리볼 수 있는 CSV/XLSX/Parquet 파일이 아직 없습니다.")
else:
    selected = st.selectbox(
        "파일",
        all_preview_files,
        format_func=lambda path: str(path.relative_to(PATHS.root)),
    )
    try:
        frame = read_table(selected)
    except Exception as exc:
        st.error(f"파일을 읽지 못했습니다: {exc}")
    else:
        st.write(f"{len(frame):,}행, {len(frame.columns):,}열")
        st.dataframe(frame.head(100), width="stretch")
