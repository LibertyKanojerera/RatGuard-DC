"""Synthetic stand-in data in exactly the downloader's format.

Purpose: test the pipeline and the app without network access. Everything here is FAKE.
Files go to data/raw_synthetic/ and every output built from them is labelled SYNTHETIC.
The generator plants three patterns on purpose (so tests can check the code finds them):
  1. rat requests rise 1-4 weeks after sanitation (food) requests on the same block group;
  2. vacant buildings and food businesses raise rat activity;
  3. low-income areas report fewer of the rats they have (311 under-reporting).
"""
from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd
from shapely.geometry import box, mapping
from shapely.ops import unary_union

from .config import RAW_SYNTHETIC, load_config

CENTER = (-77.016, 38.897)
LON0, LON1, LAT0, LAT1 = -77.12, -76.91, 38.79, 39.0


def _cells(n: int = 26):
    lons = np.linspace(LON0, LON1, n + 1)
    lats = np.linspace(LAT0, LAT1, n + 1)
    cells = []
    for i in range(n):
        for j in range(n):
            cx, cy = (lons[i] + lons[i + 1]) / 2, (lats[j] + lats[j + 1]) / 2
            # keep a diamond, roughly DC's footprint
            if abs(cx - CENTER[0]) / 0.105 + abs(cy - CENTER[1]) / 0.105 <= 1.0:
                cells.append((box(lons[i], lats[j], lons[i + 1], lats[j + 1]), cx, cy))
    return cells


def generate(seed: int = 7, end: dt.date | None = None) -> None:
    cfg = load_config()
    rng = np.random.default_rng(seed)
    out = RAW_SYNTHETIC
    end = end or (dt.date.today() - dt.timedelta(days=2))
    cells = _cells()
    n = len(cells)
    cx = np.array([c[1] for c in cells])
    cy = np.array([c[2] for c in cells])
    dx, dy = (cx - CENTER[0]) / 0.1, (cy - CENTER[1]) / 0.1

    # --- static geography ---------------------------------------------------
    angle = (np.degrees(np.arctan2(dy, dx)) + 360) % 360
    ward = (np.floor(angle / 45).astype(int) % 8) + 1
    ward[(np.abs(dx) < 0.18) & (np.abs(dy) < 0.18)] = 2            # downtown core
    income = 110000 * np.exp(0.30 * (dy - dx)) * rng.lognormal(0, 0.45, n)   # richer NW, poorer SE, plus local variation
    income = np.clip(income, 22000, 250000)
    pop = rng.integers(600, 2600, n)
    core = np.exp(-((dx + 0.05) ** 2 + (dy - 0.12) ** 2) / 0.12)   # restaurant-dense corridor
    venues_rate = 0.5 + 9 * core
    vacant_rate = 0.3 + 3.0 * np.clip(dx - dy, 0, None)
    tract_ix = np.arange(n) // 3
    geocode = np.array([f"11001{900000 + t:06d}{(i % 3) + 1}" for i, t in enumerate(tract_ix)])

    feats = []
    for k, (poly, _, _) in enumerate(cells):
        feats.append({
            "type": "Feature",
            "geometry": mapping(poly),
            "properties": {
                "GEOCODE": geocode[k], "TRACT": geocode[k][5:11], "BLKGRP": geocode[k][-1],
                "POP100": int(pop[k]), "HU100": int(pop[k] * 0.48), "ALAND": float(poly.area * 9.6e9),
            },
        })
    json.dump({"type": "FeatureCollection", "features": feats}, open(out / "block_groups.geojson", "w"))

    wards = []
    for w in range(1, 9):
        polys = [cells[k][0] for k in range(n) if ward[k] == w]
        if polys:
            wards.append({"type": "Feature", "geometry": mapping(unary_union(polys)), "properties": {"WARD": w, "NAME": f"Ward {w}"}})
    json.dump({"type": "FeatureCollection", "features": wards}, open(out / "wards.geojson", "w"))

    clusters = []
    cl = (np.floor((cx - LON0) / 0.035).astype(int) * 10 + np.floor((cy - LAT0) / 0.035).astype(int))
    for m, c in enumerate(np.unique(cl)):
        polys = [cells[k][0] for k in range(n) if cl[k] == c]
        clusters.append({"type": "Feature", "geometry": mapping(unary_union(polys)),
                         "properties": {"NAME": f"Cluster {m + 1}", "NBH_NAMES": f"Synthetic Area {m + 1}"}})
    json.dump({"type": "FeatureCollection", "features": clusters}, open(out / "clusters.geojson", "w"))

    tracts = pd.DataFrame({"GEOID": [g[:11] for g in geocode], "median_hh_income": income}).groupby("GEOID", as_index=False).mean()
    tracts["NAMELSAD"] = "Synthetic tract " + tracts["GEOID"].str[-4:]
    tracts["per_capita_income"] = tracts["median_hh_income"] * 0.55
    tracts.to_csv(out / "tract_income.csv", index=False)

    def points(k, extra):
        idx = np.repeat(np.arange(n), k)
        b = np.array([cells[i][0].bounds for i in idx]) if len(idx) else np.zeros((0, 4))
        df = pd.DataFrame({
            "OBJECTID": np.arange(1, len(idx) + 1),
            "lon": rng.uniform(b[:, 0], b[:, 2]) if len(idx) else [],
            "lat": rng.uniform(b[:, 1], b[:, 3]) if len(idx) else [],
        })
        for col, vals in extra.items():
            df[col] = rng.choice(vals, len(df)) if len(df) else []
        return df

    venues_n = rng.poisson(venues_rate)
    vac = rng.poisson(vacant_rate)
    points(vac, {"STATUS": ["Vacant", "Blighted"]}).to_parquet(out / "vacant.parquet", index=False)
    points(venues_n, {"TYPE": ["Restaurant", "Restaurant", "Tavern", "Nightclub"], "CLASS": ["CR", "DR", "CT", "CN"], "STATUS": ["Active"]}).to_parquet(out / "liquor.parquet", index=False)
    points(rng.poisson(0.2 + 0.4 * core), {"STORENAME": ["Synthetic Grocer"], "PRESENT25": ["Yes"]}).to_parquet(out / "grocery.parquet", index=False)

    # --- weather ----------------------------------------------------------------
    days = pd.date_range(f"{min(cfg['years'])}-01-01", end, freq="D")
    doy = days.dayofyear.values
    temp = 14 + 11.5 * np.sin(2 * np.pi * (doy - 105) / 365) + rng.normal(0, 3, len(days))
    precip = np.where(rng.random(len(days)) < 0.3, rng.gamma(1.2, 6, len(days)), 0.0)
    pd.DataFrame({"date": days.date.astype(str), "temp_c": temp.round(1), "precip_mm": precip.round(1)}).to_csv(out / "weather_daily.csv", index=False)

    # --- weekly 311 process -------------------------------------------------
    weeks = pd.date_range(f"{min(cfg['years'])}-01-04", end, freq="W-SUN")
    W = len(weeks)
    season = 1 + 0.45 * np.sin(2 * np.pi * (weeks.dayofyear.values - 110) / 365)
    food_base = 0.3 + 0.18 * venues_n + 0.002 * pop
    food = np.zeros((n, W))
    burst = np.zeros(n)
    for t in range(W):
        burst = 0.7 * burst + (rng.random(n) < 0.03) * rng.gamma(2, 1.5, n)   # sanitation episodes
        food[:, t] = rng.poisson(food_base * (0.8 + 0.2 * season[t]) * (1 + burst))
    colony = rng.gamma(1.5, 0.35, n) * (1 + 0.25 * vac)
    rats_true = np.zeros((n, W))
    for t in range(W):
        lagged = food[:, max(0, t - 4):max(1, t - 0)].mean(axis=1) if t > 0 else food[:, 0]
        lam = 0.22 * colony * season[t] * (1 + 0.9 * np.log1p(lagged)) * (1 + 0.12 * vac) * (1 + 0.04 * venues_n)
        if t > 0:
            lam = lam + 0.25 * rats_true[:, t - 1]
        rats_true[:, t] = rng.poisson(np.clip(lam, 0, 20))
    report_p = np.clip(0.25 + 0.55 * (np.log(income) - np.log(22000)) / np.log(250000 / 22000), 0.15, 0.85)
    rodent = rng.binomial(rats_true.astype(int), report_p[:, None])
    shelter = rng.poisson((0.05 + 0.08 * vac)[:, None] * np.ones(W))
    containers = rng.poisson(0.02 + 0.15 * np.minimum(rodent, 3))

    labels = {g: names for g, names in cfg["service_groups"].items()}
    weights = {"food": [0.12, 0.2, 0.1, 0.25, 0.05, 0.1, 0.04, 0.06, 0.04, 0.02, 0.02]}
    rows = []
    for g, mat in (("rodent", rodent), ("food", food), ("shelter", shelter), ("containers", containers)):
        ii, tt = np.nonzero(mat)
        reps = mat[ii, tt].astype(int)
        ii, tt = np.repeat(ii, reps), np.repeat(tt, reps)
        names = labels[g]
        p = np.array(weights.get(g, [1] * len(names))[: len(names)], dtype=float)
        p = p / p.sum()
        b = np.array([cells[i][0].bounds for i in ii])
        ts = weeks.values[tt] - np.timedelta64(6, "D") + (rng.random(len(tt)) * 7 * 86400).astype("timedelta64[s]")
        rows.append(pd.DataFrame({
            "SERVICECODEDESCRIPTION": rng.choice(names, len(ii), p=p),
            "ADDDATE": ts,
            "LATITUDE": rng.uniform(b[:, 1], b[:, 3]),
            "LONGITUDE": rng.uniform(b[:, 0], b[:, 2]),
            "WARD": [f"Ward {w}" for w in ward[ii]],
            "group": g,
        }))
    sr = pd.concat(rows, ignore_index=True)
    sr = sr[sr["ADDDATE"] <= pd.Timestamp(end)].sort_values("ADDDATE").reset_index(drop=True)
    sr["OBJECTID"] = np.arange(1, len(sr) + 1)
    sr["SERVICEREQUESTID"] = "SYN-" + sr["OBJECTID"].astype(str)
    sr["RESOLUTIONDATE"] = sr["ADDDATE"] + pd.to_timedelta(rng.integers(1, 20, len(sr)), unit="D")
    sr["SERVICEORDERSTATUS"] = "Closed"
    sr["DETAILS"] = np.where(sr["group"] == "rodent", rng.choice(["Burrows treated", "No rodent activity found", "Exterior inspected"], len(sr)), None)
    sr["year_layer"] = sr["ADDDATE"].dt.year
    for y, d in sr.groupby("year_layer"):
        d.to_parquet(out / f"sr_{y}.parquet", index=False)
        d.groupby("SERVICECODEDESCRIPTION").size().rename("n").reset_index().to_csv(out / f"service_types_{y}.csv", index=False)
    json.dump({"mode": "synthetic", "generated": dt.datetime.now().isoformat(timespec="seconds"), "rows": len(sr)},
              open(out / "download_log.json", "w"), indent=2)
    print(f"Synthetic data written to {out} ({n} fake block groups, {len(sr):,} fake 311 rows). THIS IS NOT REAL DATA.")
