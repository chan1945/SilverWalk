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
HIGH_RISK_PERCENT_THRESHOLD = 10.0
RECOMMENDATION_COLUMNS = [f"개선우선순위{rank}" for rank in range(1, 4)]
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


def _file_mtime_ns(path) -> int:
    return path.stat().st_mtime_ns if path.exists() else 0


@st.cache_data(show_spinner=False)
def _load_improvement_recommendations(file_mtime_ns: int):
    import pandas as pd

    path = PATHS.recommendations / "point_improvement_recommendations.csv"
    columns = ["POINT_ID", *RECOMMENDATION_COLUMNS]
    if not path.exists():
        return pd.DataFrame(columns=columns)

    try:
        return pd.read_csv(path, usecols=columns)
    except ValueError:
        return pd.DataFrame(columns=columns)


@st.cache_data(show_spinner=False)
def _load_high_risk_points(percent_threshold: float, prediction_mtime_ns: int, recommendation_mtime_ns: int):
    import pandas as pd

    path = PATHS.predictions / "two_stage_zero_risk_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(
            "두 모델 결합 최종 결과 파일이 없습니다. "
            "`python scripts/predict/predict_two_stage_zero_risk.py`를 먼저 실행하십시오."
        )

    try:
        frame = pd.read_csv(
            path,
            usecols=[
                "POINT_ID",
                "위도",
                "경도",
                "위험도_actual",
                "사고발생확률_p",
                "조건부위험도_r",
                "최종위험도점수",
                "최종위험도점수_percent",
            ],
        )
    except ValueError as error:
        raise ValueError(
            "두 모델 결합 최종 결과 파일에 필요한 컬럼이 없습니다. "
            "`python scripts/predict/predict_two_stage_zero_risk.py`를 다시 실행하십시오."
        ) from error
    frame = frame.dropna(subset=["위도", "경도", "위험도_actual", "최종위험도점수_percent"])
    frame = frame[
        (frame["위험도_actual"] == 0)
        & (frame["최종위험도점수_percent"] > percent_threshold)
    ].copy()
    recommendations = _load_improvement_recommendations(recommendation_mtime_ns)
    if not recommendations.empty:
        frame = frame.merge(recommendations, on="POINT_ID", how="left")

    for column in RECOMMENDATION_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
        frame[column] = frame[column].fillna("")

    return frame.sort_values("최종위험도점수_percent", ascending=False)


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
    if risk >= 50:
        return "#dc2626"
    if risk >= 30:
        return "#f97316"
    return "#facc15"


def _high_risk_level(risk: float) -> str:
    if risk >= 50:
        return "위험"
    if risk >= 30:
        return "경고"
    return "주의"


def _improvement_recommendations(row) -> list[str]:
    recommendations = []
    for rank, column in enumerate(RECOMMENDATION_COLUMNS, start=1):
        value = getattr(row, column, "")
        if value:
            recommendations.append(str(value))
    return recommendations


def _find_clicked_point(high_risk_points, clicked_object):
    if high_risk_points is None or high_risk_points.empty or not clicked_object:
        return None
    if "lat" not in clicked_object or "lng" not in clicked_object:
        return None

    import pandas as pd

    clicked_lat = float(clicked_object["lat"])
    clicked_lng = float(clicked_object["lng"])
    distances = (
        (high_risk_points["위도"] - clicked_lat) ** 2
        + (high_risk_points["경도"] - clicked_lng) ** 2
    )
    closest_index = distances.idxmin()
    if pd.isna(closest_index) or distances.loc[closest_index] > 1e-8:
        return None
    return high_risk_points.loc[closest_index]


def _render_selected_point_sidebar(selected_point) -> None:
    st.sidebar.divider()
    st.sidebar.subheader("선택 포인트")

    if selected_point is None:
        st.sidebar.caption("지도에서 위험 포인트를 선택하면 개선우선순위가 표시됩니다.")
        return

    risk_percent = float(selected_point["최종위험도점수_percent"])
    recommendations = _improvement_recommendations(selected_point)

    st.sidebar.metric("POINT_ID", f"{int(selected_point['POINT_ID'])}")
    st.sidebar.write(f"등급: {_high_risk_level(risk_percent)}")
    st.sidebar.write(f"최종위험도: {risk_percent:.2f}%")
    st.sidebar.write(f"사고발생확률 p: {float(selected_point['사고발생확률_p']):.4f}")
    st.sidebar.write(f"위험도 r: {float(selected_point['조건부위험도_r']):.3f}")

    st.sidebar.markdown("**개선우선순위**")
    if recommendations:
        for rank, recommendation in enumerate(recommendations, start=1):
            st.sidebar.write(f"{rank}. {recommendation}")
    else:
        st.sidebar.caption("두 모델에서 공통으로 양수 기여한 개선 추천 항목이 없습니다.")


def _selected_point_from_session(high_risk_points):
    if high_risk_points is None or high_risk_points.empty:
        return None
    selected_point_id = st.session_state.get("selected_point_id")
    if selected_point_id is None:
        return None

    rows = high_risk_points[high_risk_points["POINT_ID"] == selected_point_id]
    if rows.empty:
        return None
    return rows.iloc[0]


def render_map_app() -> None:
    selection = _select_region()

    with st.sidebar.expander("배경지도", expanded=False):
        vworld_layer = st.selectbox("VWorld 레이어", list(VWORLD_LAYER_TYPES), index=0)
        vworld_api_key = st.text_input("VWorld API Key", value=_get_default_vworld_key(), type="password")
        if not vworld_api_key:
            st.caption("API key가 없으면 임시 기본 배경지도를 표시합니다.")

    boundary_gdf = _load_sigungu_boundary(selection.sido_code, selection.sigungu_code)
    try:
        prediction_path = PATHS.predictions / "two_stage_zero_risk_predictions.csv"
        recommendation_path = PATHS.recommendations / "point_improvement_recommendations.csv"
        high_risk_points = _load_high_risk_points(
            HIGH_RISK_PERCENT_THRESHOLD,
            _file_mtime_ns(prediction_path),
            _file_mtime_ns(recommendation_path),
        )
    except (FileNotFoundError, ValueError) as error:
        st.error(str(error))
        high_risk_points = None
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

    if high_risk_points is not None:
        high_risk_layer = folium.FeatureGroup(
            name=f"최종 위험도 {HIGH_RISK_PERCENT_THRESHOLD:g}% 초과 포인트",
            show=True,
        )
        for row in high_risk_points.itertuples(index=False):
            accident_probability = float(getattr(row, "사고발생확률_p"))
            conditional_risk = float(getattr(row, "조건부위험도_r"))
            final_score = float(getattr(row, "최종위험도점수"))
            risk_percent = float(getattr(row, "최종위험도점수_percent"))
            color = _high_risk_point_color(risk_percent)
            folium.CircleMarker(
                location=(float(getattr(row, "위도")), float(getattr(row, "경도"))),
                radius=5,
                color=color,
                weight=1,
                fill=True,
                fill_color=color,
                fill_opacity=0.82,
                tooltip=(
                    f"등급: {_high_risk_level(risk_percent)}<br>"
                    f"사고발생확률 p: {accident_probability:.4f}<br>"
                    f"위험도 r: {conditional_risk:.3f}<br>"
                    f"최종위험도 점수: {final_score:.3f}<br>"
                    f"최종위험도(%): {risk_percent:.2f}%"
                ),
            ).add_to(high_risk_layer)
        high_risk_layer.add_to(map_obj)
        st.sidebar.caption(
            f"최종 위험도 {HIGH_RISK_PERCENT_THRESHOLD:g}% 초과 포인트 {len(high_risk_points):,}개"
        )

    folium.LayerControl(collapsed=False).add_to(map_obj)
    map_state = st_folium(
        map_obj,
        height=760,
        width="100%",
        returned_objects=["last_object_clicked"],
    )
    clicked_object = map_state.get("last_object_clicked") if map_state else None
    clicked_point = _find_clicked_point(high_risk_points, clicked_object)
    if clicked_point is not None:
        st.session_state["selected_point_id"] = int(clicked_point["POINT_ID"])

    selected_point = clicked_point if clicked_point is not None else _selected_point_from_session(high_risk_points)
    _render_selected_point_sidebar(selected_point)
