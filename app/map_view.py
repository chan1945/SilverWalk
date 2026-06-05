from __future__ import annotations

import json
import os
from dataclasses import dataclass
from urllib.request import urlopen

import streamlit as st

from app.bootstrap import add_src_to_path

add_src_to_path()

from silverwalk_ai.data.paths import PATHS
from silverwalk_ai.visualization.maps import VWORLD_LAYER_TYPES, make_base_map


DEFAULT_SIDO = "서울특별시"
DEFAULT_SIGUNGU = "전체"
HIGH_RISK_THRESHOLD = 50.0
ONLINE_SEOUL_SIGUNGU_GEOJSON_URL = (
    "https://raw.githubusercontent.com/southkorea/seoul-maps/master/"
    "kostat/2013/json/seoul_municipalities_geo_simple.json"
)


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
    return _load_online_seoul_region_codes()


@st.cache_data(show_spinner=False)
def _load_online_seoul_geojson() -> dict:
    with urlopen(ONLINE_SEOUL_SIGUNGU_GEOJSON_URL, timeout=20) as response:
        return json.load(response)


@st.cache_data(show_spinner=False)
def _load_online_seoul_region_codes():
    import pandas as pd

    try:
        features = _load_online_seoul_geojson().get("features", [])
    except Exception:
        return pd.DataFrame(
            columns=["sido_code", "sido_name", "sigungu_code_part", "sigungu_name", "sigungu_code"]
        )

    rows = []
    for feature in features:
        properties = feature.get("properties", {})
        sigungu_code = str(properties.get("code", ""))
        sigungu_name = properties.get("name")
        if not sigungu_code or not sigungu_name:
            continue
        rows.append(
            {
                "sido_code": "11",
                "sido_name": DEFAULT_SIDO,
                "sigungu_code_part": sigungu_code[2:],
                "sigungu_name": sigungu_name,
                "sigungu_code": sigungu_code,
            }
        )

    return pd.DataFrame(rows).sort_values(["sido_name", "sigungu_name"])


@st.cache_data(show_spinner=False)
def _load_sigungu_boundary_projected(sido_code: str, sigungu_code: str | None):
    return _load_online_seoul_boundary(sido_code, sigungu_code)


@st.cache_data(show_spinner=False)
def _load_online_seoul_boundary(sido_code: str, sigungu_code: str | None):
    import geopandas as gpd

    if sido_code != "11":
        return gpd.GeoDataFrame(columns=["SIGUNGU_CD", "SIGUNGU_NM", "geometry"], geometry="geometry", crs=4326)

    try:
        geojson = _load_online_seoul_geojson()
    except Exception:
        return gpd.GeoDataFrame(columns=["SIGUNGU_CD", "SIGUNGU_NM", "geometry"], geometry="geometry", crs=4326)

    gdf = gpd.GeoDataFrame.from_features(geojson["features"], crs=4326)
    gdf = gdf.rename(columns={"code": "SIGUNGU_CD", "name": "SIGUNGU_NM"})
    gdf["SIGUNGU_CD"] = gdf["SIGUNGU_CD"].astype(str)

    if sigungu_code is not None:
        gdf = gdf[gdf["SIGUNGU_CD"] == sigungu_code]

    return gdf[["SIGUNGU_CD", "SIGUNGU_NM", "geometry"]]


@st.cache_data(show_spinner=False)
def _load_sigungu_boundary(sido_code: str, sigungu_code: str | None):
    gdf = _load_sigungu_boundary_projected(sido_code, sigungu_code)
    if not gdf.empty and gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    return gdf


@st.cache_data(show_spinner=False)
def _load_high_risk_points(threshold: float):
    import pandas as pd

    path = PATHS.data / "original_train_data" / "seoul_road_points.csv"
    frame = pd.read_csv(path, usecols=["POINT_ID", "위도", "경도", "위험도"])
    frame = frame.dropna(subset=["위도", "경도", "위험도"])
    frame = frame[frame["위험도"] > threshold].copy()
    return frame.sort_values("위험도", ascending=False)


def _select_region() -> RegionSelection:
    regions = _load_region_codes()
    if regions.empty:
        st.sidebar.caption("행정구 경계 데이터를 불러오지 못해 서울특별시 전체로 표시합니다.")
        return RegionSelection(
            sido_code="11",
            sido_name=DEFAULT_SIDO,
            sigungu_name=DEFAULT_SIGUNGU,
            sigungu_code=None,
        )

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
        return 37.5665, 126.9780

    centroid = boundary_gdf.geometry.union_all().centroid
    return centroid.y, centroid.x


def _high_risk_point_color(risk: float) -> str:
    if risk >= 150:
        return "#7f1d1d"
    if risk >= 100:
        return "#dc2626"
    if risk >= 75:
        return "#f97316"
    return "#facc15"


def render_map_app() -> None:
    selection = _select_region()

    with st.sidebar.expander("배경지도", expanded=False):
        vworld_layer = st.selectbox("VWorld 레이어", list(VWORLD_LAYER_TYPES), index=0)
        vworld_api_key = st.text_input("VWorld API Key", value=_get_default_vworld_key(), type="password")
        if not vworld_api_key:
            st.caption("API key가 없으면 임시 기본 배경지도를 표시합니다.")

    boundary_gdf = _load_sigungu_boundary(selection.sido_code, selection.sigungu_code)
    high_risk_points = _load_high_risk_points(HIGH_RISK_THRESHOLD)
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

    high_risk_layer = folium.FeatureGroup(
        name=f"위험도 {HIGH_RISK_THRESHOLD:g} 초과 포인트",
        show=True,
    )
    for row in high_risk_points.itertuples(index=False):
        risk = float(getattr(row, "위험도"))
        color = _high_risk_point_color(risk)
        folium.CircleMarker(
            location=(float(getattr(row, "위도")), float(getattr(row, "경도"))),
            radius=5,
            color=color,
            weight=1,
            fill=True,
            fill_color=color,
            fill_opacity=0.82,
            tooltip=(
                f"POINT_ID: {int(getattr(row, 'POINT_ID'))}<br>"
                f"위험도: {risk:.3f}"
            ),
        ).add_to(high_risk_layer)
    high_risk_layer.add_to(map_obj)
    st.sidebar.caption(f"위험도 {HIGH_RISK_THRESHOLD:g} 초과 포인트 {len(high_risk_points):,}개")

    folium.LayerControl(collapsed=False).add_to(map_obj)
    st_folium(map_obj, height=760, width="100%")
