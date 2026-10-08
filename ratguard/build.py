"""Turn raw downloads into analysis-ready tables.

Outputs (data/processed/):
  panel.parquet    one row per census block group x month, with counts, features and labels
  weekly.parquet   block group x week counts of rat and sanitation requests (for the lag analysis)
  bg_static.parquet / bg.geojson   block group attributes and simplified shapes
"""
from __future__ import annotations

import json
import re

import geopandas as gpd
import numpy as np
import pandas as pd

from .config import APP_DATA, PROCESSED, REPORTS, load_config, norm_type, raw_dir, type_to_group

GROUPS = ["rodent", "food", "shelter", "containers"]
DC_BBOX = (-77.125, 38.79, -76.905, 39.0)  # lon0, lat0, lon1, lat1
PLANAR = 26985  # NAD83 / Maryland (metres) - used for areas and interior points


def _ward_num(x) -> float:
    m = re.search(r"(\d)", str(x))
    return float(m.group(1)) if m else np.nan


def _points(df: pd.DataFrame, lon: str = "lon", lat: str = "lat") -> gpd.GeoDataFrame:
    df = df.copy()
    df[lon] = pd.to_numeric(df[lon], errors="coerce")
    df[lat] = pd.to_numeric(df[lat], errors="coerce")
    df = df.dropna(subset=[lon, lat])
    df = df[df[lon].between(DC_BBOX[0], DC_BBOX[2]) & df[lat].between(DC_BBOX[1], DC_BBOX[3])]
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon], df[lat]), crs=4326)


def _count_in(bg: gpd.GeoDataFrame, pts: gpd.GeoDataFrame, name: str) -> pd.Series:
    if pts is None or pts.empty:
        return pd.Series(0, index=bg["GEOCODE"], name=name)
    j = gpd.sjoin(pts, bg[["GEOCODE", "geometry"]], predicate="within", how="inner")
    return j.groupby("GEOCODE").size().reindex(bg["GEOCODE"]).fillna(0).astype(int).rename(name)


def _truthy(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin(["y", "yes", "1", "true", "t", "x"])


def load_block_groups(raw) -> gpd.GeoDataFrame:
    bg = gpd.read_file(raw / "block_groups.geojson").to_crs(4326)
    bg["GEOCODE"] = bg["GEOCODE"].astype(str)
    bg = bg.drop_duplicates("GEOCODE").reset_index(drop=True)
    bg["POP100"] = pd.to_numeric(bg.get("POP100"), errors="coerce").fillna(0)
    bg["HU100"] = pd.to_numeric(bg.get("HU100"), errors="coerce").fillna(0)
    area_km2 = bg.to_crs(PLANAR).area / 1e6
    aland = pd.to_numeric(bg.get("ALAND"), errors="coerce") / 1e6
    bg["area_km2"] = np.where(aland > 0, aland, area_km2)
    rep = bg.to_crs(PLANAR).representative_point().to_crs(4326)
    bg["rep_lon"], bg["rep_lat"] = rep.x, rep.y
    return bg


def attach_context(bg: gpd.GeoDataFrame, raw) -> gpd.GeoDataFrame:
    rep = gpd.GeoDataFrame(bg[["GEOCODE"]], geometry=gpd.points_from_xy(bg.rep_lon, bg.rep_lat), crs=4326)
    bg["ward"] = np.nan
    if (raw / "wards.geojson").exists():
        w = gpd.read_file(raw / "wards.geojson").to_crs(4326)
        j = gpd.sjoin(rep, w[["WARD", "geometry"]], predicate="within", how="left").drop_duplicates("GEOCODE")
        bg["ward"] = j.set_index("GEOCODE").reindex(bg.GEOCODE)["WARD"].map(_ward_num).values
    bg["neighborhood"] = ""
    if (raw / "clusters.geojson").exists():
        c = gpd.read_file(raw / "clusters.geojson").to_crs(4326)
        col = "NBH_NAMES" if "NBH_NAMES" in c else "NAME"
        j = gpd.sjoin(rep, c[[col, "geometry"]], predicate="within", how="left").drop_duplicates("GEOCODE")
        bg["neighborhood"] = j.set_index("GEOCODE").reindex(bg.GEOCODE)[col].fillna("").astype(str).values
    bg["tract"] = bg["GEOCODE"].str[:11]
    bg["median_hh_income"] = np.nan
    if (raw / "tract_income.csv").exists():
        inc = pd.read_csv(raw / "tract_income.csv", dtype={"GEOID": str})
        inc["GEOID"] = inc["GEOID"].str.zfill(11)
        bg["median_hh_income"] = bg["tract"].map(inc.set_index("GEOID")["median_hh_income"])

    def pts(name):
        f = raw / f"{name}.parquet"
        return _points(pd.read_parquet(f)) if f.exists() else None

    vac = pts("vacant")
    liq = pts("liquor")
    gro = pts("grocery")
    if liq is not None and "STATUS" in liq and liq["STATUS"].astype(str).str.contains("activ", case=False).any():
        liq = liq[liq["STATUS"].astype(str).str.contains("activ", case=False)]
    rest = None
    if liq is not None:
        is_rest = pd.Series(False, index=liq.index)
        if "TYPE" in liq:
            is_rest |= liq["TYPE"].astype(str).str.contains("restaurant", case=False)
        if "CLASS" in liq:
            is_rest |= liq["CLASS"].astype(str).str.upper().isin(["CR", "DR"])
        rest = liq[is_rest] if is_rest.any() else liq
    if gro is not None:
        for col in ("PRESENT26", "PRESENT25"):
            if col in gro and _truthy(gro[col]).any():
                gro = gro[_truthy(gro[col])]
                break
    bg = bg.set_index("GEOCODE")
    bg["vacant_n"] = _count_in(bg.reset_index(), vac, "vacant_n")
    bg["venues_n"] = _count_in(bg.reset_index(), liq, "venues_n")
    bg["restaurants_n"] = _count_in(bg.reset_index(), rest, "restaurants_n")
    bg["grocery_n"] = _count_in(bg.reset_index(), gro, "grocery_n")
    bg = bg.reset_index()
    bg["pop_density"] = bg["POP100"] / bg["area_km2"].clip(lower=0.01)
    return bg


def load_311(raw, cfg) -> pd.DataFrame:
    files = sorted(raw.glob("sr_*.parquet"))
    if not files:
        raise SystemExit(f"No 311 files in {raw}. Run `python run.py download` (or `python run.py synthetic`).")
    sr = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if "SERVICEREQUESTID" in sr:
        sr = sr.drop_duplicates("SERVICEREQUESTID")
    mapping = type_to_group(cfg)
    sr["group"] = sr["SERVICECODEDESCRIPTION"].map(lambda x: mapping.get(norm_type(x)))
    sr = sr[sr["group"].notna()].copy()
    sr["ADDDATE"] = pd.to_datetime(sr["ADDDATE"])
    return sr


def run(synthetic: bool = False) -> dict:
    cfg = load_config()
    raw = raw_dir(synthetic)
    print(f"[build] reading {'SYNTHETIC' if synthetic else 'real'} data from {raw}")
    bg = attach_context(load_block_groups(raw), raw)
    sr = load_311(raw, cfg)

    pts = _points(sr, lon="LONGITUDE", lat="LATITUDE")
    j = gpd.sjoin(pts, bg[["GEOCODE", "geometry"]], predicate="within", how="inner")
    matched = len(j) / max(len(sr), 1)
    print(f"[build] {len(sr):,} relevant 311 requests; {matched:.1%} placed in a block group")

    if bg["ward"].isna().all() and "WARD" in j:
        wmode = j.assign(w=j["WARD"].map(_ward_num)).groupby("GEOCODE")["w"].agg(lambda s: s.mode().iloc[0] if s.notna().any() else np.nan)
        bg["ward"] = bg["GEOCODE"].map(wmode)

    last = j["ADDDATE"].max()
    end_m = last.to_period("M")
    if (last + pd.Timedelta(days=1)).month == last.month:  # month not finished yet
        end_m = end_m - 1
    start_m = pd.Period(f"{min(cfg['years'])}-01", "M")
    j["month"] = j["ADDDATE"].dt.to_period("M")
    j = j[(j["month"] >= start_m) & (j["month"] <= end_m)]

    counts = j.pivot_table(index=["GEOCODE", "month"], columns="group", values="ADDDATE", aggfunc="count")
    months = pd.period_range(start_m, end_m, freq="M")
    full = pd.MultiIndex.from_product([bg["GEOCODE"], months], names=["GEOCODE", "month"])
    panel = counts.reindex(full).fillna(0)
    for g in GROUPS:
        if g not in panel:
            panel[g] = 0.0
    panel = panel[GROUPS].astype(float).reset_index()

    # weekly counts for the lag analysis
    j["week"] = j["ADDDATE"].dt.to_period("W-SUN").dt.start_time
    wk = j[j["group"].isin(["rodent", "food"])].pivot_table(index=["GEOCODE", "week"], columns="group", values="ADDDATE", aggfunc="count")
    weeks = pd.date_range(j["week"].min(), j["week"].max(), freq="7D")
    wk = wk.reindex(pd.MultiIndex.from_product([bg["GEOCODE"], weeks], names=["GEOCODE", "week"])).fillna(0)
    for g in ("rodent", "food"):
        if g not in wk:
            wk[g] = 0.0
    wk = wk[["rodent", "food"]].reset_index()
    wk = wk[wk["week"] < end_m.end_time]

    panel = add_features(panel, bg, raw, cfg)

    # rodent request outcomes text, to see what DETAILS can tell us about confirmed rats
    if "DETAILS" in sr:
        top = sr.loc[sr["group"] == "rodent", "DETAILS"].fillna("(blank)").astype(str).str.strip().str[:120]
        top.value_counts().head(40).rename_axis("DETAILS").reset_index(name="n").to_csv(REPORTS / "rodent_details_top40.csv", index=False)

    static_cols = ["GEOCODE", "ward", "neighborhood", "tract", "POP100", "HU100", "area_km2", "pop_density",
                   "median_hh_income", "vacant_n", "venues_n", "restaurants_n", "grocery_n", "rep_lon", "rep_lat"]
    bg_static = pd.DataFrame(bg[static_cols])
    panel.to_parquet(PROCESSED / "panel.parquet", index=False)
    wk.to_parquet(PROCESSED / "weekly.parquet", index=False)
    bg_static.to_parquet(PROCESSED / "bg_static.parquet", index=False)

    shapes = bg[["GEOCODE", "geometry"]].copy()
    shapes["geometry"] = shapes.geometry.simplify(0.00008, preserve_topology=True)
    shapes = shapes.merge(bg_static[["GEOCODE", "ward", "neighborhood"]], on="GEOCODE")
    shapes.to_file(APP_DATA / "bg.geojson", driver="GeoJSON")

    meta = {
        "data_mode": "synthetic" if synthetic else "real",
        "first_month": str(start_m),
        "last_full_month": str(end_m),
        "last_request_date": str(last.date()),
        "block_groups": int(len(bg)),
        "requests_used": {g: int((j["group"] == g).sum()) for g in GROUPS},
        "share_placed_in_block_group": round(float(matched), 4),
    }
    json.dump(meta, open(PROCESSED / "build_meta.json", "w"), indent=2)
    print(f"[build] panel: {len(panel):,} rows ({len(bg)} block groups x {len(months)} months), {start_m} to {end_m}")
    return meta


def add_features(panel: pd.DataFrame, bg: pd.DataFrame, raw, cfg) -> pd.DataFrame:
    """Rolling history, static context, weather, season, and the 3-month-ahead label."""
    h = int(cfg["model"]["horizon_months"])
    panel = panel.sort_values(["GEOCODE", "month"]).reset_index(drop=True)
    g = panel.groupby("GEOCODE", sort=False)
    for c in GROUPS:
        panel[f"{c}_3"] = g[c].transform(lambda s: s.rolling(3, min_periods=1).sum())
        panel[f"{c}_12"] = g[c].transform(lambda s: s.rolling(12, min_periods=1).sum())
    panel["rodent_months_12"] = g["rodent"].transform(lambda s: (s > 0).astype(float).rolling(12, min_periods=1).sum())
    panel["food_trend"] = panel["food_3"] / 3 - panel["food_12"] / 12

    idx = panel.groupby("GEOCODE", sort=False).cumcount()
    last_hit = idx.where(panel["rodent"] > 0)
    last_hit = last_hit.groupby(panel["GEOCODE"], sort=False).ffill()
    panel["months_since_rodent"] = (idx - last_hit).fillna(24).clip(upper=24)

    fut = sum(g["rodent"].shift(-k) for k in range(1, h + 1))
    panel["y_next"] = np.where(fut.notna(), (fut > 0).astype(float), np.nan)
    panel["rodent_next"] = fut

    static = bg.set_index("GEOCODE")[["vacant_n", "venues_n", "restaurants_n", "grocery_n", "pop_density", "POP100"]]
    panel = panel.join(static, on="GEOCODE")
    panel["log_pop"] = np.log1p(panel.pop("POP100"))

    m = panel["month"].dt.month
    panel["month_sin"] = np.sin(2 * np.pi * m / 12)
    panel["month_cos"] = np.cos(2 * np.pi * m / 12)
    wf = raw / "weather_daily.csv"
    if wf.exists():
        w = pd.read_csv(wf, parse_dates=["date"])
        w["month"] = w["date"].dt.to_period("M")
        wm = w.groupby("month").agg(temp_c=("temp_c", "mean"), precip_mm=("precip_mm", "sum"))
        panel = panel.join(wm, on="month")
    for c in ("temp_c", "precip_mm"):
        if c not in panel:
            panel[c] = np.nan
    return panel
