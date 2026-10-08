"""Offline test of the downloader: a fake ArcGIS server built from the synthetic files.

Checks paging (1,000-row pages), statistics queries, GeoJSON layers, field fallbacks,
and that `build` can read what `download` writes.
Run:  python run.py synthetic && python -m tests.test_download_mock
"""
from __future__ import annotations

import json
import re
import shutil
import tempfile
from pathlib import Path

import pandas as pd

from ratguard import arcgis, build, download
from ratguard.config import RAW_SYNTHETIC

SYN = RAW_SYNTHETIC
YEARS = sorted(int(p.stem.split("_")[1]) for p in SYN.glob("sr_*.parquet"))
LAYER_IDS = {y: 100 + i for i, y in enumerate(YEARS)}
SR = {y: pd.read_parquet(SYN / f"sr_{y}.parquet") for y in YEARS}
CALLS = {"query": 0, "stats": 0}


def fake_request_json(url, params, method="POST", **kw):
    if url.endswith("ServiceRequests/FeatureServer"):
        return {"layers": [{"id": LAYER_IDS[y], "name": f"All Service Requests - {y}"} for y in YEARS]
                + [{"id": 13, "name": "All Service Requests - Last 90 Days"}]}
    m = re.search(r"/(\d+)/query$", url)
    lid = int(m.group(1)) if m else None
    if "outStatistics" in params:
        CALLS["stats"] += 1
        y = [k for k, v in LAYER_IDS.items() if v == lid][0]
        c = SR[y].groupby("SERVICECODEDESCRIPTION").size()
        return {"features": [{"attributes": {"SERVICECODEDESCRIPTION": k, "N": int(v)}} for k, v in c.items()]}
    CALLS["query"] += 1
    off, n = int(params.get("resultOffset", 0)), int(params.get("resultRecordCount", 1000))
    if lid in LAYER_IDS.values():
        y = [k for k, v in LAYER_IDS.items() if v == lid][0]
        labels = re.findall(r"'((?:[^']|'')*)'", params["where"])
        df = SR[y][SR[y]["SERVICECODEDESCRIPTION"].isin([x.replace("''", "'") for x in labels])].reset_index(drop=True)
        fields = params["outFields"].split(",")
        if "DETAILS" in fields and y == YEARS[0]:
            raise arcgis.ArcGISError("Invalid field: DETAILS")  # exercise the fallback path
        page = df.iloc[off: off + n].copy()
        page["ADDDATE"] = pd.to_datetime(page["ADDDATE"]).astype("datetime64[ms]").astype("int64")
        page["RESOLUTIONDATE"] = pd.to_datetime(page["RESOLUTIONDATE"]).astype("datetime64[ms]").astype("int64")
        keep = [f for f in fields if f in page.columns]
        feats = [{"attributes": r} for r in page[keep].to_dict("records")]
        return {"features": feats, "exceededTransferLimit": off + n < len(df)}
    name = {2: "block_groups", 17: "clusters", 53: "wards"}.get(lid)
    if name and params.get("f") == "geojson":
        gj = json.load(open(SYN / f"{name}.geojson"))
        return {"type": "FeatureCollection", "features": gj["features"][off: off + n],
                "properties": {"exceededTransferLimit": off + n < len(gj["features"])}}
    pts = {82: "vacant", 5: "liquor", 4: "grocery"}.get(lid)
    if pts:
        df = pd.read_parquet(SYN / f"{pts}.parquet").iloc[off: off + n]
        if "PRESENT26" in params["outFields"]:
            raise arcgis.ArcGISError("Invalid field: PRESENT26")
        return {"features": [{"attributes": {k: v for k, v in r.items() if k not in ("lon", "lat")},
                              "geometry": {"x": r["lon"], "y": r["lat"]}} for r in df.to_dict("records")]}
    if lid == 41:
        inc = pd.read_csv(SYN / "tract_income.csv", dtype={"GEOID": str}).iloc[off: off + n]
        inc = inc.rename(columns={"median_hh_income": "DP03_0062E", "per_capita_income": "DP03_0088E"})
        return {"features": [{"attributes": r} for r in inc.to_dict("records")]}
    raise AssertionError(f"unexpected call {url} {params}")


def main():
    tmp = Path(tempfile.mkdtemp())
    download.RAW = tmp
    arcgis.request_json = fake_request_json
    download.arcgis.request_json = fake_request_json
    cfg = download.load_config()
    cfg["years"] = YEARS
    log311 = download.download_311(cfg, years=YEARS, force=True)
    logctx = download.download_context(cfg, force=True)
    shutil.copy(SYN / "weather_daily.csv", tmp / "weather_daily.csv")
    print("311:", log311)
    print("context:", logctx)
    print("calls:", CALLS)
    for y in YEARS:
        got = pd.read_parquet(tmp / f"sr_{y}.parquet")
        assert len(got) == len(SR[y]), (y, len(got), len(SR[y]))
        assert got["ADDDATE"].dt.year.isin([y, y + 1, y - 1]).all()
    assert all(not str(v).startswith("FAILED") for v in logctx.values()), logctx
    build.raw_dir = lambda synthetic: tmp
    out = tmp / "out"
    out.mkdir()
    build.PROCESSED = build.APP_DATA = build.REPORTS = out  # never touch the real outputs
    meta = build.run(synthetic=False)
    assert meta["block_groups"] > 0 and meta["requests_used"]["rodent"] > 0
    print("OK - downloader paging, fallbacks and build all work on the mock server.")
    shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
