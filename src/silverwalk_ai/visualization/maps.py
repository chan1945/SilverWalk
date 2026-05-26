"""Folium map helpers for road-risk visualization."""

from __future__ import annotations

from pathlib import Path


JINJU_CENTER = (35.1803, 128.1076)
VWORLD_ATTRIBUTION = "VWorld"
VWORLD_LAYER_TYPES = {
    "Base": "png",
    "white": "png",
    "midnight": "png",
    "Hybrid": "png",
    "Satellite": "jpeg",
}


def risk_color(value: float | int | None) -> str:
    if value is None:
        return "#64748b"
    if value >= 0.75:
        return "#dc2626"
    if value >= 0.5:
        return "#f97316"
    if value >= 0.25:
        return "#facc15"
    return "#22c55e"


def vworld_wmts_url(api_key: str, layer: str = "Base") -> str:
    tile_type = VWORLD_LAYER_TYPES[layer]
    return f"https://api.vworld.kr/req/wmts/1.0.0/{api_key}/{layer}/{{z}}/{{y}}/{{x}}.{tile_type}"


def add_vworld_wmts_layer(map_obj, api_key: str, layer: str = "Base", name: str | None = None):
    import folium

    folium.TileLayer(
        tiles=vworld_wmts_url(api_key, layer),
        attr=VWORLD_ATTRIBUTION,
        name=name or f"VWorld {layer}",
        min_zoom=6,
        max_zoom=19,
        max_native_zoom=19,
        overlay=False,
        control=True,
    ).add_to(map_obj)
    return map_obj


def make_base_map(
    center: tuple[float, float] = JINJU_CENTER,
    zoom_start: int = 12,
    vworld_api_key: str | None = None,
    vworld_layer: str = "Base",
):
    import folium

    map_obj = folium.Map(
        location=center,
        zoom_start=zoom_start,
        tiles=None,
        control_scale=True,
    )

    if vworld_api_key:
        add_vworld_wmts_layer(map_obj, vworld_api_key, vworld_layer)
    else:
        folium.TileLayer(
            tiles="CartoDB positron",
            name="CartoDB positron",
            control=True,
        ).add_to(map_obj)

    return map_obj


def add_geojson_layer(map_obj, geojson_path: Path, name: str = "Layer"):
    import folium

    folium.GeoJson(
        str(geojson_path),
        name=name,
        style_function=lambda _: {
            "color": "#2563eb",
            "weight": 2,
            "fillOpacity": 0.12,
        },
    ).add_to(map_obj)
    folium.LayerControl(collapsed=False).add_to(map_obj)
    return map_obj
