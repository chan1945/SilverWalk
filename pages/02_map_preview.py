from __future__ import annotations

import streamlit as st

from app.map_view import render_map_app


st.set_page_config(page_title="SilverWalk 지도", page_icon="SW", layout="wide")
st.title("지도 미리보기")
render_map_app()
