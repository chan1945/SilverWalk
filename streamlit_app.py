from __future__ import annotations

import streamlit as st

from app.map_view import render_map_app
from app.bootstrap import add_src_to_path

add_src_to_path()

from silverwalk_ai.data.paths import ensure_project_dirs


st.set_page_config(
    page_title="SilverWalk",
    page_icon="SW",
    layout="wide",
    initial_sidebar_state="expanded",
)

ensure_project_dirs()

st.title("SilverWalk")
render_map_app()
