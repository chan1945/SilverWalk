from __future__ import annotations

import os
from dataclasses import dataclass

import streamlit as st

from app.bootstrap import add_src_to_path

add_src_to_path()

from silverwalk_ai.data.paths import PATHS
from silverwalk_ai.visualization.maps import VWORLD_LAYER_TYPES, make_base_map


DEFAULT_SIDO = "서울특별시"
DEFAULT_SIGUNGU = "전체"
ROAD_COLUMNS = ["LINK_ID", "ROAD_NAME", "ROAD_RANK", "MAX_SPD", "LANES", "LENGTH", "geometry"]


@dataclass(frozen=True)
class RegionSelection:
    sido_code: str
    sido_name: str
    sigungu_name: str
    sigungu_code: str | None

    @property
    def display_name(self) -> str:
        if self.sigungu_code is None:
            return self.sido_name
        return f"{self.sido_name} {self.sigungu_name}"


def _get_default_vworld_key() -> str:
    try:
        return st.secrets["vworld"]["api_key"]
    except Exception:
        return os.getenv("VWORLD_API_KEY", "")


@st.cache_data(show_spinner=False)
def _load_region_codes():
    import pandas as pd

    frame = pd.read_excel(PATHS.sigungu / "센서스 공간정보 지역 코드.xlsx", header=1)
    frame = frame.rename(
        columns={
            "시도코드": "sido_code",
            "시도명칭": "sido_name",
            "시군구코드": "sigungu_code_part",
            "시군구명칭": "sigungu_name",
        }
    )
    frame = frame[["sido_code", "sido_name", "sigungu_code_part", "sigungu_name"]].dropna()
    frame["sido_code"] = frame["sido_code"].astype(str).str.zfill(2)
    frame["sigungu_code_part"] = frame["sigungu_code_part"].astype(str).str.zfill(3)
    frame["sigungu_code"] = frame["sido_code"] + frame["sigungu_code_part"]
    return frame.drop_duplicates(["sido_name", "sigungu_name", "sigungu_code"]).sort_values(
        ["sido_name", "sigungu_name"]
    )


@st.cache_data(show_spinner=False)
def _load_sigungu_boundary_projected(sido_code: str, sigungu_code: str | None):
    import geopandas as gpd

    path = PATHS.sigungu / "BND_SIGUNGU_PG" / "BND_SIGUNGU_PG.shp"
    if sigungu_code is not None:
        return gpd.read_file(path, where=f"SIGUNGU_CD = '{sigungu_code}'")

    gdf = gpd.read_file(path)
    return gdf[gdf["SIGUNGU_CD"].astype(str).str.startswith(sido_code)]


@st.cache_data(show_spinner=False)
def _load_sigungu_boundary(sido_code: str, sigungu_code: str | None):
    gdf = _load_sigungu_boundary_projected(sido_code, sigungu_code)
    if not gdf.empty and gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    return gdf


@st.cache_data(show_spinner=False)
def _load_roads_for_region(sido_code: str, sigungu_code: str | None, simplify_tolerance: float):
    import geopandas as gpd

    boundary_gdf = _load_sigungu_boundary_projected(sido_code, sigungu_code)
    if boundary_gdf.empty:
        return gpd.GeoDataFrame(columns=ROAD_COLUMNS, geometry="geometry", crs="EPSG:4326")

    path = PATHS.nodelink / "MOCT_LINK.shp"
    boundary_geometry = boundary_gdf.geometry.union_all()

    # 시군구 경계 bbox로 먼저 읽기 범위를 줄이고, 실제 경계 교차 여부로 최종 필터링한다.
    gdf = gpd.read_file(path, bbox=tuple(boundary_gdf.total_bounds))
    gdf = gdf[gdf.geometry.intersects(boundary_geometry)]
    if gdf.empty:
        return gpd.GeoDataFrame(columns=ROAD_COLUMNS, geometry="geometry", crs="EPSG:4326")

    gdf = gdf[[column for column in ROAD_COLUMNS if column in gdf.columns]]
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)

    if simplify_tolerance > 0:
        # 성능 모드에서만 화면 표시용 선형 geometry를 단순화한다.
        gdf["geometry"] = gdf.geometry.simplify(simplify_tolerance, preserve_topology=True)
    return gdf


def _select_region() -> RegionSelection:
    regions = _load_region_codes()
    sido_names = regions["sido_name"].drop_duplicates().tolist()

    default_sido_index = sido_names.index(DEFAULT_SIDO) if DEFAULT_SIDO in sido_names else 0
    sido_name = st.sidebar.selectbox("광역시-도", sido_names, index=default_sido_index)

    sigungu_rows = regions[regions["sido_name"] == sido_name]
    sigungu_names = sigungu_rows["sigungu_name"].drop_duplicates().tolist()
    sigungu_options = ["전체", *sigungu_names]
    default_sigungu_index = (
        sigungu_options.index(DEFAULT_SIGUNGU) if DEFAULT_SIGUNGU in sigungu_options else 0
    )
    sigungu_name = st.sidebar.selectbox("시군구", sigungu_options, index=default_sigungu_index)

    selected_sido = sigungu_rows.iloc[0]
    if sigungu_name == "전체":
        return RegionSelection(
            sido_code=str(selected_sido["sido_code"]),
            sido_name=sido_name,
            sigungu_name=sigungu_name,
            sigungu_code=None,
        )

    selected = sigungu_rows[sigungu_rows["sigungu_name"] == sigungu_name].iloc[0]
    return RegionSelection(
        sido_code=str(selected["sido_code"]),
        sido_name=sido_name,
        sigungu_name=sigungu_name,
        sigungu_code=str(selected["sigungu_code"]),
    )


def _map_center(boundary_gdf) -> tuple[float, float]:
    if boundary_gdf.empty:
        return 35.1803, 128.1076

    centroid = boundary_gdf.geometry.union_all().centroid
    return centroid.y, centroid.x


def _ranked_road_style(feature):
    road_rank = feature["properties"].get("ROAD_RANK")
    color_by_rank = {
        "101": "#ef4444",
        "102": "#f97316",
        "103": "#eab308",
        "104": "#22c55e",
        "105": "#14b8a6",
        "106": "#2563eb",
        "107": "#334155",
    }
    return {
        "color": color_by_rank.get(str(road_rank), "#475569"),
        "weight": 2.0,
        "opacity": 0.9,
    }


def _clear_road_style(_feature):
    return {
        "color": "#0f172a",
        "weight": 1.6,
        "opacity": 0.94,
    }


def render_map_app() -> None:
    selection = _select_region()

    with st.sidebar.expander("도로 표시", expanded=True):
        show_roads = st.checkbox("도로 링크 표시", value=True)
        road_style_mode = st.radio("도로 스타일", ["전체 도로 선명", "도로등급별 색상"], horizontal=False)
        simplify_roads = st.checkbox("성능 모드", value=False)

    with st.sidebar.expander("배경지도", expanded=False):
        vworld_layer = st.selectbox("VWorld 레이어", list(VWORLD_LAYER_TYPES), index=0)
        vworld_api_key = st.text_input("VWorld API Key", value=_get_default_vworld_key(), type="password")
        if not vworld_api_key:
            st.caption("API key가 없으면 임시 기본 배경지도를 표시합니다.")

    boundary_gdf = _load_sigungu_boundary(selection.sido_code, selection.sigungu_code)
    roads_gdf = None
    if show_roads:
        simplify_tolerance = 0.00001 if simplify_roads else 0.0
        roads_gdf = _load_roads_for_region(selection.sido_code, selection.sigungu_code, simplify_tolerance)
    center = _map_center(boundary_gdf)

    import folium
    from streamlit_folium import st_folium

    map_obj = make_base_map(
        center=center,
        zoom_start=12,
        vworld_api_key=vworld_api_key.strip() or None,
        vworld_layer=vworld_layer,
    )

    if not boundary_gdf.empty:
        folium.GeoJson(
            boundary_gdf.to_json(),
            name=selection.display_name,
            tooltip=folium.GeoJsonTooltip(fields=["SIGUNGU_NM", "SIGUNGU_CD"]),
            style_function=lambda _: {
                "color": "#1d4ed8",
                "weight": 3,
                "fillColor": "#60a5fa",
                "fillOpacity": 0.08,
            },
        ).add_to(map_obj)
    else:
        st.warning("선택한 시군구 경계 데이터를 찾지 못했습니다.")

    if show_roads and roads_gdf is not None and not roads_gdf.empty:
        tooltip_fields = [
            column
            for column in ["ROAD_NAME", "LINK_ID", "ROAD_RANK", "MAX_SPD", "LANES", "LENGTH"]
            if column in roads_gdf
        ]
        folium.GeoJson(
            roads_gdf.to_json(),
            name="도로 링크",
            tooltip=folium.GeoJsonTooltip(fields=tooltip_fields),
            style_function=_clear_road_style if road_style_mode == "전체 도로 선명" else _ranked_road_style,
        ).add_to(map_obj)
        st.sidebar.caption(f"도로 링크 {len(roads_gdf):,}개")
    elif show_roads:
        st.sidebar.caption("도로 링크 0개")

    folium.LayerControl(collapsed=False).add_to(map_obj)
    st_folium(map_obj, height=760, width="100%")
