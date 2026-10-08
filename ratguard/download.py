"""Download everything RatGuard needs from DC Open Data (ArcGIS REST) and Open-Meteo.

Outputs land in data/raw/ (git-ignored). Re-running skips files that already exist,
except the current year's 311 file, which is refreshed so the app stays current.
"""
from __future__ import annotations

import datetime as dt
import json
import re

import pandas as pd
import requests

from . import arcgis
from .config import RAW, load_config, norm_type, type_to_group

SR_FIELDS = "OBJECTID,SERVICEREQUESTID,SERVICECODEDESCRIPTION,ADDDATE,RESOLUTIONDATE,SERVICEORDERSTATUS,WARD,LATITUDE,LONGITUDE"
RODENT_EXTRA = ",DETAILS,INSPECTIONFLAG,STATUS_CODE"


def _ms_to_local(s: pd.Series) -> pd.Series:
    t = pd.to_datetime(pd.to_numeric(s, errors="coerce"), unit="ms", utc=True)
    return t.dt.tz_convert("America/New_York").dt.tz_localize(None)


def year_layers(cfg: dict) -> dict[int, int]:
    url = f"{cfg['arcgis']['base']}/{cfg['arcgis']['service_requests']}"
    out = {}
    for lyr in arcgis.list_layers(url):
        m = re.search(r"All Service Requests - (\d{4})$", lyr.get("name", ""))
        if m:
            out[int(m.group(1))] = lyr["id"]
    return out


def download_311(cfg: dict, years: list[int] | None = None, force: bool = False) -> dict:
    base = f"{cfg['arcgis']['base']}/{cfg['arcgis']['service_requests']}"
    layers = year_layers(cfg)
    mapping = type_to_group(cfg)
    this_year = dt.date.today().year
    log = {}
    for year in years or cfg["years"]:
        out = RAW / f"sr_{year}.parquet"
        if year not in layers:
            print(f"[311 {year}] no layer named 'All Service Requests - {year}' - skipped")
            log[year] = "missing layer"
            continue
        if out.exists() and not force and year != this_year:
            print(f"[311 {year}] cached ({out.name})")
            log[year] = "cached"
            continue
        url = f"{base}/{layers[year]}"
        print(f"[311 {year}] layer {layers[year]}: counting service types...")
        counts = arcgis.type_counts(url)
        counts.to_csv(RAW / f"service_types_{year}.csv", index=False)
        labels = counts["SERVICECODEDESCRIPTION"].dropna().tolist()
        chosen = {g: [lab for lab in labels if mapping.get(norm_type(lab)) == g] for g in cfg["service_groups"]}
        for g, names in cfg["service_groups"].items():
            found = {norm_type(x) for x in chosen[g]}
            missing = [n for n in names if norm_type(n) not in found]
            if missing:
                print(f"    note: {g} types not present in {year}: {missing}")
        frames = []
        for g, labs in chosen.items():
            if not labs:
                continue
            fields = SR_FIELDS + (RODENT_EXTRA if g == "rodent" else "")
            print(f"    downloading {g} ({len(labs)} types)...")
            try:
                rows = arcgis.query_all(url, where=arcgis.sql_in("SERVICECODEDESCRIPTION", labs), out_fields=fields, label=f"{year} {g}")
            except arcgis.ArcGISError:
                # Older layers may lack an optional field; fall back to the core set.
                rows = arcgis.query_all(url, where=arcgis.sql_in("SERVICECODEDESCRIPTION", labs), out_fields=SR_FIELDS, label=f"{year} {g}")
            df = pd.DataFrame(rows)
            if df.empty:
                continue
            df["group"] = g
            frames.append(df)
            print(f"    {g}: {len(df):,} requests")
        if not frames:
            log[year] = "no rows"
            continue
        df = pd.concat(frames, ignore_index=True)
        for c in ("ADDDATE", "RESOLUTIONDATE"):
            if c in df:
                df[c] = _ms_to_local(df[c])
        for c in ("LATITUDE", "LONGITUDE"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["year_layer"] = year
        df.to_parquet(out, index=False)
        log[year] = f"{len(df):,} rows"
        print(f"[311 {year}] saved {len(df):,} rows -> {out.name}")
    return log


def _save_points(cfg: dict, key: str, fields: str, fallback: str = "OBJECTID") -> str:
    url = f"{cfg['arcgis']['base']}/{cfg['arcgis'][key]}"
    try:
        rows = arcgis.query_all(url, out_fields=fields, geometry=True, page_size=1000)
    except arcgis.ArcGISError:
        rows = arcgis.query_all(url, out_fields=fallback, geometry=True, page_size=1000)
    df = pd.DataFrame(rows)
    df.to_parquet(RAW / f"{key}.parquet", index=False)
    return f"{len(df):,} points"


def _save_polygons(cfg: dict, key: str, fields: str) -> str:
    url = f"{cfg['arcgis']['base']}/{cfg['arcgis'][key]}"
    gj = arcgis.query_geojson(url, fields)
    with open(RAW / f"{key}.geojson", "w", encoding="utf-8") as f:
        json.dump(gj, f)
    return f"{len(gj['features']):,} polygons"


def download_context(cfg: dict, force: bool = False) -> dict:
    """Block groups, income, vacant buildings, food businesses, neighborhoods, wards."""
    jobs = [
        ("block_groups", lambda: _save_polygons(cfg, "block_groups", "GEOCODE,TRACT,BLKGRP,POP100,HU100,ALAND"), "block_groups.geojson"),
        ("clusters", lambda: _save_polygons(cfg, "clusters", "NAME,NBH_NAMES"), "clusters.geojson"),
        ("wards", lambda: _save_polygons(cfg, "wards", "WARD,NAME"), "wards.geojson"),
        ("vacant", lambda: _save_points(cfg, "vacant", "OBJECTID,STATUS,FIRST_KNOWN_DATE"), "vacant.parquet"),
        ("liquor", lambda: _save_points(cfg, "liquor", "OBJECTID,TRADE_NAME,CLASS,TYPE,STATUS"), "liquor.parquet"),
        ("grocery", lambda: _save_points(cfg, "grocery", "OBJECTID,STORENAME,PRESENT25,PRESENT26", "OBJECTID,STORENAME"), "grocery.parquet"),
        ("tract_income", lambda: _save_income(cfg), "tract_income.csv"),
    ]
    log = {}
    for name, fn, fname in jobs:
        if (RAW / fname).exists() and not force:
            print(f"[{name}] cached")
            log[name] = "cached"
            continue
        try:
            print(f"[{name}] downloading...")
            log[name] = fn()
            print(f"[{name}] {log[name]}")
        except Exception as e:  # optional sources must not stop the run
            log[name] = f"FAILED: {e}"
            print(f"[{name}] FAILED - continuing without it ({str(e)[:150]})")
    return log


def _save_income(cfg: dict) -> str:
    url = f"{cfg['arcgis']['base']}/{cfg['arcgis']['tract_income']}"
    rows = arcgis.query_all(url, out_fields="GEOID,NAMELSAD,DP03_0062E,DP03_0088E")
    df = pd.DataFrame(rows).rename(columns={"DP03_0062E": "median_hh_income", "DP03_0088E": "per_capita_income"})
    for c in ("median_hh_income", "per_capita_income"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df.loc[df[c] < 0, c] = None  # Census uses negative sentinels for "no estimate"
    df.to_csv(RAW / "tract_income.csv", index=False)
    return f"{len(df):,} tracts"


def download_weather(cfg: dict) -> str:
    w = cfg["weather"]
    start = f"{min(cfg['years'])}-01-01"
    end = (dt.date.today() - dt.timedelta(days=7)).isoformat()
    params = {
        "latitude": w["latitude"],
        "longitude": w["longitude"],
        "start_date": start,
        "end_date": end,
        "daily": "temperature_2m_mean,precipitation_sum",
        "timezone": "America/New_York",
    }
    r = requests.get(w["url"], params=params, timeout=120)
    r.raise_for_status()
    d = r.json()["daily"]
    df = pd.DataFrame({"date": d["time"], "temp_c": d["temperature_2m_mean"], "precip_mm": d["precipitation_sum"]})
    df.to_csv(RAW / "weather_daily.csv", index=False)
    return f"{len(df):,} days"


def run(years: list[int] | None = None, force: bool = False, skip_311: bool = False) -> None:
    cfg = load_config()
    log = {"started": dt.datetime.now().isoformat(timespec="seconds"), "mode": "real"}
    log["context"] = download_context(cfg, force=force)
    try:
        log["weather"] = download_weather(cfg)
        print(f"[weather] {log['weather']}")
    except Exception as e:
        log["weather"] = f"FAILED: {e}"
        print(f"[weather] FAILED - continuing without it ({e})")
    if not skip_311:
        log["311"] = download_311(cfg, years=years, force=force)
    log["finished"] = dt.datetime.now().isoformat(timespec="seconds")
    with open(RAW / "download_log.json", "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, default=str)
    print("Download complete. Next: python run.py build")


def show_types() -> None:
    """Print every 311 service type with counts per year, marking the ones RatGuard uses."""
    cfg = load_config()
    mapping = type_to_group(cfg)
    files = sorted(RAW.glob("service_types_*.csv"))
    if not files:
        print("No service-type inventories yet. Run: python run.py download")
        return
    frames = []
    for f in files:
        d = pd.read_csv(f)
        d["year"] = int(f.stem.split("_")[-1])
        frames.append(d)
    df = pd.concat(frames)
    wide = df.pivot_table(index="SERVICECODEDESCRIPTION", columns="year", values="n", aggfunc="sum").fillna(0).astype(int)
    wide["ratguard_group"] = [mapping.get(norm_type(i), "") for i in wide.index]
    wide = wide.sort_values(list(wide.columns[:-1])[-1], ascending=False)
    out = RAW.parent / "service_types_by_year.csv"
    wide.to_csv(out)
    with pd.option_context("display.max_rows", 400, "display.width", 200):
        print(wide)
    print(f"\nSaved {out}. Edit service_groups in config.yaml to add or drop types.")
