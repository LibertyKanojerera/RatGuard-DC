"""Small, dependency-light client for DC's ArcGIS REST services (maps2.dcgis.dc.gov)."""
from __future__ import annotations

import time

import pandas as pd
import requests

HEADERS = {"User-Agent": "RatGuard-DC/0.1 (AI for Good student project)"}
SESSION = requests.Session()
SESSION.headers.update(HEADERS)


class ArcGISError(RuntimeError):
    pass


def request_json(url: str, params: dict, method: str = "POST", retries: int = 5, timeout: int = 120) -> dict:
    """Call an ArcGIS REST endpoint, retrying on network errors and server hiccups."""
    last = None
    for attempt in range(retries):
        try:
            if method == "GET":
                r = SESSION.get(url, params=params, timeout=timeout)
            else:
                r = SESSION.post(url, data=params, timeout=timeout)
            r.raise_for_status()
            js = r.json()
            if isinstance(js, dict) and "error" in js:
                raise ArcGISError(f"{url}: {js['error']}")
            return js
        except (requests.RequestException, ValueError, ArcGISError) as e:  # noqa: PERF203
            last = e
            wait = 2 ** attempt
            print(f"    retry {attempt + 1}/{retries} in {wait}s ({type(e).__name__}: {str(e)[:120]})")
            time.sleep(wait)
    raise ArcGISError(f"Failed after {retries} attempts: {last}")


def list_layers(service_url: str) -> list[dict]:
    js = request_json(service_url, {"f": "json"}, method="GET")
    return js.get("layers", [])


def query_all(
    layer_url: str,
    where: str = "1=1",
    out_fields: str = "*",
    geometry: bool = False,
    page_size: int = 1000,
    label: str = "",
) -> list[dict]:
    """Page through every matching record. Returns one dict per feature.

    For point layers with geometry=True, adds `lon`/`lat` (WGS84).
    For polygons, use `query_geojson` instead.
    """
    rows: list[dict] = []
    offset = 0
    while True:
        params = {
            "where": where,
            "outFields": out_fields,
            "returnGeometry": "true" if geometry else "false",
            "outSR": 4326,
            "orderByFields": "OBJECTID ASC",
            "resultOffset": offset,
            "resultRecordCount": page_size,
            "f": "json",
        }
        js = request_json(f"{layer_url}/query", params)
        feats = js.get("features", [])
        for f in feats:
            rec = dict(f.get("attributes", {}))
            g = f.get("geometry")
            if geometry and g and "x" in g:
                rec["lon"], rec["lat"] = g["x"], g["y"]
            rows.append(rec)
        offset += len(feats)
        if label and offset and offset % (page_size * 20) == 0:
            print(f"    {label}: {offset:,} rows...")
        if not feats or (not js.get("exceededTransferLimit") and len(feats) < page_size):
            break
    return rows


def query_geojson(layer_url: str, out_fields: str, where: str = "1=1", page_size: int = 1000) -> dict:
    """Download a polygon layer as one GeoJSON FeatureCollection in WGS84."""
    features: list[dict] = []
    offset = 0
    while True:
        params = {
            "where": where,
            "outFields": out_fields,
            "returnGeometry": "true",
            "outSR": 4326,
            "orderByFields": "OBJECTID ASC",
            "resultOffset": offset,
            "resultRecordCount": page_size,
            "f": "geojson",
        }
        js = request_json(f"{layer_url}/query", params)
        feats = js.get("features", [])
        features.extend(feats)
        offset += len(feats)
        exceeded = js.get("exceededTransferLimit") or js.get("properties", {}).get("exceededTransferLimit")
        if not feats or (not exceeded and len(feats) < page_size):
            break
    return {"type": "FeatureCollection", "features": features}


def type_counts(layer_url: str, field: str = "SERVICECODEDESCRIPTION") -> pd.DataFrame:
    """Count records per service type using a server-side statistics query."""
    stats = '[{"statisticType":"count","onStatisticField":"OBJECTID","outStatisticFieldName":"n"}]'
    js = request_json(
        f"{layer_url}/query",
        {"where": "1=1", "groupByFieldsForStatistics": field, "outStatistics": stats, "f": "json"},
    )
    rows = [f["attributes"] for f in js.get("features", [])]
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=[field, "n"])
    cols = {c: ("n" if c.lower() == "n" else field) for c in df.columns}
    df = df.rename(columns=cols)
    return df.sort_values("n", ascending=False).reset_index(drop=True)


def sql_in(field: str, values: list[str]) -> str:
    quoted = ",".join("'" + v.replace("'", "''") + "'" for v in values)
    return f"{field} IN ({quoted})"
