"""Analysis: trend, lead-lag test, rat-request forecasts (with ablation), SHAP drivers, equity audit.

Forecast target: number of 311 rat requests in a block group over the NEXT 3 months.
Why a count and not "any request yes/no": in busy parts of DC almost every block group gets
a rat request every quarter, so a yes/no target saturates and every model looks perfect.
Ranking block groups by expected requests is what an inspector or BID actually needs.

Headline metric - capture rate: each month, flag the top 10% of block groups by forecast;
what share of next quarter's rat requests happen in the flagged block groups?
(Random flagging captures about 10%.)

Reads data/processed/, writes:
  reports/   PNG figures + CSV tables + results.md (numbers to paste into the memo)
  app_data/  small files the Streamlit app reads
"""
from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from . import plotstyle as ps
from .config import APP_DATA, PROCESSED, REPORTS, load_config
from .fixes import category, label

COUNT_FEATS = ["rodent", "rodent_3", "rodent_12", "rodent_months_12", "months_since_rodent", "food", "food_3", "food_12",
               "shelter_3", "shelter_12", "containers_3", "containers_12", "vacant_n", "venues_n", "restaurants_n",
               "grocery_n", "pop_density"]
FULL = ["rodent", "rodent_3", "rodent_12", "rodent_months_12", "months_since_rodent",
        "food", "food_3", "food_12", "food_trend", "shelter_3", "shelter_12", "containers_3", "containers_12",
        "vacant_n", "venues_n", "restaurants_n", "grocery_n", "pop_density", "log_pop",
        "temp_c", "precip_mm", "month_sin", "month_cos"]
RODENT_ONLY = ["rodent", "rodent_3", "rodent_12", "rodent_months_12", "months_since_rodent",
               "temp_c", "precip_mm", "month_sin", "month_cos"]
CONDITIONS = ["vacant_n", "venues_n", "restaurants_n", "grocery_n", "pop_density", "log_pop", "area_km2"]

MODEL_NAMES = {
    "naive": "Naive: next 3 months = last 3 months",
    "glm": "Poisson regression",
    "rf": "Random forest",
    "xgb_rodent": "XGBoost, rat history only",
    "xgb_full": "XGBoost, full RatGuard",
}
SLICES = {"all": "All block groups", "recent": "Recurrence: block groups with a rat request this month"}


# ---------------------------------------------------------------- helpers
def _p(s: str) -> pd.Period:
    return pd.Period(s, "M")


def splits(panel: pd.DataFrame, cfg: dict):
    m = cfg["model"]
    h = int(m["horizon_months"])
    start, vs, ts, te = (_p(m[k]) for k in ("start", "val_start", "test_start", "test_end"))
    mo = panel["month"]
    lab = panel["rodent_next"].notna() & (mo >= start)
    train = lab & (mo <= vs - (h + 1))           # purge gap: training labels end before validation starts
    val = lab & (mo >= vs) & (mo <= ts - (h + 1))
    test = lab & (mo >= ts) & (mo <= te)
    if test.sum() == 0:
        raise SystemExit("No labelled rows in the test window - check model dates in config.yaml against the data range.")
    return train, val, test, lab


def _pre(feats):
    cnt = [f for f in feats if f in COUNT_FEATS]
    oth = [f for f in feats if f not in COUNT_FEATS]
    return ColumnTransformer([
        ("cnt", make_pipeline(SimpleImputer(strategy="median"), FunctionTransformer(np.log1p)), cnt),
        ("oth", SimpleImputer(strategy="median"), oth),
    ])


def make_glm(feats):
    return Pipeline([("pre", _pre(feats)), ("sc", StandardScaler()), ("glm", PoissonRegressor(alpha=1e-3, max_iter=3000))])


def make_rf(seed):
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("rf", RandomForestRegressor(n_estimators=300, min_samples_leaf=25, max_features=0.5, n_jobs=-1, random_state=seed))])


def make_xgb(seed, n_estimators=3000, early=True):
    return xgb.XGBRegressor(
        objective="count:poisson", n_estimators=n_estimators, learning_rate=0.03, max_depth=4, min_child_weight=5,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist", eval_metric="poisson-nloglik",
        early_stopping_rounds=100 if early else None, random_state=seed, n_jobs=-1,
    )


def capture_at_k(y, s, m, frac, rng):
    """Each month flag the top `frac` of rows by score; share of all outcome counts that fall in flagged rows."""
    n = len(y)
    order = np.lexsort((rng.random(n), -s, m))   # month, then score desc, then random tie-break
    ms, ys = m[order], y[order]
    starts = np.r_[0, np.flatnonzero(np.diff(ms)) + 1]
    sizes = np.diff(np.r_[starts, n])
    pos = np.arange(n) - np.repeat(starts, sizes)
    k = np.maximum(1, np.ceil(frac * sizes)).astype(int)
    sel = pos < np.repeat(k, sizes)
    return ys[sel].sum() / max(ys.sum(), 1e-9)


def spearman_by_month(y, s, m):
    df = pd.DataFrame({"y": y, "s": s, "m": m})
    df["ry"] = df.groupby("m")["y"].rank()
    df["rs"] = df.groupby("m")["s"].rank()
    c = df.groupby("m")[["ry", "rs"]].corr().unstack().iloc[:, 1]
    return float(c.mean())


def evaluate(y, preds: dict, m, bgs, frac, reps, rng):
    """Point metrics + block-group bootstrap CIs, plus paired differences between key models."""
    y = np.asarray(y, float)
    out = {}
    for name, s in preds.items():
        s = np.asarray(s, float)
        out[name] = {
            "rows": int(len(y)), "mean_next3": y.mean(), "share_any_next3": (y > 0).mean(),
            "capture_top": capture_at_k(y, s, m, frac, rng),
            "spearman": spearman_by_month(y, s, m),
            "poisson_dev": mean_poisson_deviance(y, np.clip(s, 0.05, None)),
            "mae": mean_absolute_error(y, s),
        }
    rows_by = pd.Series(np.arange(len(y))).groupby(np.asarray(bgs)).apply(lambda x: x.to_numpy()).to_dict()
    keys = np.array(list(rows_by))
    boot = {name: [] for name in preds}
    for _ in range(reps):
        idx = np.concatenate([rows_by[k] for k in rng.choice(keys, len(keys), replace=True)])
        draw = rng.random(len(idx))  # shared tie-break so paired differences are fair
        for name, s in preds.items():
            order_rng = np.random.default_rng(int(draw[0] * 1e9))
            boot[name].append(capture_at_k(y[idx], np.asarray(s)[idx], m[idx], frac, order_rng))
    for name in preds:
        b = np.array(boot[name])
        out[name]["capture_lo"], out[name]["capture_hi"] = np.percentile(b, [2.5, 97.5])
    diffs = {}
    for a, bname in (("xgb_full", "xgb_rodent"), ("xgb_full", "naive")):
        if a in boot and bname in boot:
            d = np.array(boot[a]) - np.array(boot[bname])
            diffs[f"{a}_minus_{bname}"] = {"point": out[a]["capture_top"] - out[bname]["capture_top"],
                                           "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5))}
    return out, diffs


# ---------------------------------------------------------------- analyses
def trend(panel, bg_static):
    t = panel.groupby("month")[["rodent", "food", "shelter", "containers"]].sum().reset_index()
    t["month"] = t["month"].astype(str)
    pw = panel.join(bg_static.set_index("GEOCODE")["ward"], on="GEOCODE")
    pw["year"] = pw["month"].dt.year
    tw = pw.groupby(["year", "ward"])["rodent"].sum().reset_index()
    return t, tw


def lag_analysis(wk, reps, rng, lags=range(-6, 11)):
    """Within-block-group, season-removed correlation of rat requests with sanitation requests k weeks earlier."""
    R = np.log1p(wk.pivot(index="GEOCODE", columns="week", values="rodent").fillna(0).to_numpy(float))
    F = np.log1p(wk.pivot(index="GEOCODE", columns="week", values="food").fillna(0).to_numpy(float))

    def demean(X):  # remove each block group's average and each week's citywide average
        return X - X.mean(1, keepdims=True) - X.mean(0, keepdims=True) + X.mean()

    Rd, Fd = demean(R), demean(F)
    T = Rd.shape[1]
    lags = list(lags)

    def corr(rows):
        out = []
        for k in lags:
            if k >= 0:
                a, b = Rd[rows, k:], Fd[rows, : T - k]
            else:
                a, b = Rd[rows, :k], Fd[rows, -k:]
            out.append((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))
        return np.array(out)

    n = Rd.shape[0]
    point = corr(np.arange(n))
    boots = np.array([corr(rng.integers(0, n, n)) for _ in range(reps)])
    lead = [i for i, k in enumerate(lags) if 1 <= k <= 4]
    lagb = [i for i, k in enumerate(lags) if -4 <= k <= -1]
    diff_b = boots[:, lead].mean(1) - boots[:, lagb].mean(1)
    df = pd.DataFrame({"lag_weeks": lags, "corr": point,
                       "lo": np.percentile(boots, 2.5, axis=0), "hi": np.percentile(boots, 97.5, axis=0)})
    summary = {
        "food_leads_mean_corr": float(point[lead].mean()),
        "rats_lead_mean_corr": float(point[lagb].mean()),
        "difference": float(point[lead].mean() - point[lagb].mean()),
        "difference_lo": float(np.percentile(diff_b, 2.5)),
        "difference_hi": float(np.percentile(diff_b, 97.5)),
        "block_groups": int(n), "weeks": int(T),
    }
    return df, summary


def equity(panel, bg_static, end_m, seed, reps, rng):
    """Observed vs expected rat requests, where 'expected' uses physical conditions only (no 311, no income)."""
    last12 = panel[(panel["month"] > end_m - 12) & (panel["month"] <= end_m)]
    obs = last12.groupby("GEOCODE")["rodent"].sum()
    st = bg_static.set_index("GEOCODE").loc[obs.index].copy()
    st["log_pop"] = np.log1p(st["POP100"])
    model = HistGradientBoostingRegressor(loss="poisson", max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                          min_samples_leaf=15, random_state=seed)
    exp = cross_val_predict(model, st[CONDITIONS], obs.to_numpy(), cv=KFold(5, shuffle=True, random_state=seed))
    exp = exp * obs.sum() / max(exp.sum(), 1e-9)
    df = pd.DataFrame({"GEOCODE": obs.index, "observed_12m": obs.to_numpy(), "expected_12m": exp})
    df = df.join(st[["ward", "neighborhood", "median_hh_income"]], on="GEOCODE")
    df["oe_ratio"] = (df["observed_12m"] + 1) / (df["expected_12m"] + 1)
    df["income_quintile"] = pd.qcut(df["median_hh_income"], 5,
                                    labels=["Q1 (lowest income)", "Q2", "Q3", "Q4", "Q5 (highest income)"])
    df["blind_spot"] = (df["expected_12m"] >= df["expected_12m"].quantile(0.7)) & (df["oe_ratio"] < 0.6)

    def table(key):
        rows = []
        for k, g in df.dropna(subset=[key]).groupby(key, observed=True):
            o, e = g["observed_12m"].to_numpy(), g["expected_12m"].to_numpy()
            b = [o[ix].sum() / max(e[ix].sum(), 1e-9) for ix in (rng.integers(0, len(g), len(g)) for _ in range(reps))]
            rows.append({key: str(k), "block_groups": len(g), "observed": o.sum(), "expected": e.sum(),
                         "oe": o.sum() / max(e.sum(), 1e-9), "oe_lo": np.percentile(b, 2.5), "oe_hi": np.percentile(b, 97.5),
                         "blind_spots": int(g["blind_spot"].sum())})
        return pd.DataFrame(rows)

    return df, table("income_quintile"), table("ward")


# ---------------------------------------------------------------- figures
def fig_trend(t, path, src):
    import matplotlib.pyplot as plt
    ps.setup()
    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = pd.PeriodIndex(t["month"], freq="M").to_timestamp()
    ax.plot(x, t["rodent"], color=ps.BLUE, lw=2)
    yr = t.assign(y=x.year).groupby("y")["rodent"].sum()
    ax.set_title("Rat requests to DC 311, by month", pad=26)
    ax.set_ylabel("Requests per month")
    ax.set_ylim(bottom=0)
    ax.grid(axis="x", visible=False)
    ax.text(0.0, 1.025, "Yearly totals: " + ", ".join(f"{y}: {v:,.0f}" for y, v in yr.items()) + " (last year partial)",
            transform=ax.transAxes, fontsize=9, color=ps.INK_2)
    ps.finish(fig, path, src)


def fig_lag(lag, summ, path, src):
    import matplotlib.pyplot as plt
    ps.setup()
    fig, ax = plt.subplots(figsize=(10, 4.4))
    colors = [ps.BLUE if k >= 1 else ps.GREY for k in lag["lag_weeks"]]
    ax.bar(lag["lag_weeks"], lag["corr"], color=colors, width=0.7)
    ax.errorbar(lag["lag_weeks"], lag["corr"], yerr=[lag["corr"] - lag["lo"], lag["hi"] - lag["corr"]],
                fmt="none", ecolor=ps.INK_2, elinewidth=1, capsize=2)
    ax.axhline(0, color=ps.AXIS, lw=1)
    ax.axvline(0.5, color=ps.AXIS, lw=1, ls="--")
    supported = summ["difference_lo"] > 0
    ax.set_title("Rat requests rise in the weeks after sanitation requests on the same block group" if supported
                 else "Sanitation and rat requests by lag: no clear lead found")
    ax.set_xlabel("Weeks that sanitation requests come before rat requests  (left of dashed line: rat requests first)")
    ax.set_ylabel("Correlation (within block group,\nseason removed)")
    ax.set_xticks(lag["lag_weeks"])
    ax.grid(axis="x", visible=False)
    ps.finish(fig, path, src)


def fig_models(metrics, frac, path, src):
    import matplotlib.pyplot as plt
    ps.setup()
    order = list(MODEL_NAMES)
    hi = float(metrics["capture_hi"].max()) * 100
    xmax = min(100, np.ceil((hi + 8) / 10) * 10)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, (sl, title) in zip(axes, SLICES.items()):
        d = metrics[metrics["slice"] == sl].set_index("model").loc[order]
        y = np.arange(len(order))
        col = [ps.BLUE if m == "xgb_full" else ps.GREY for m in order]
        ax.hlines(y, d["capture_lo"] * 100, d["capture_hi"] * 100, color=col, lw=2)
        ax.scatter(d["capture_top"] * 100, y, color=col, s=60, zorder=3)
        ax.axvline(frac * 100, color=ps.INK_2, lw=1, ls="--")
        ax.text(frac * 100 + 0.8, -0.55, "random", fontsize=8.5, color=ps.INK_2, va="bottom")
        for yi, v, h in zip(y, d["capture_top"], d["capture_hi"]):
            ax.text(h * 100 + 1.2, yi, f"{v:.0%}", va="center", fontsize=9.5, color=ps.INK)
        ax.set_title("All block groups" if sl == "all" else "Recently treated block groups (recurrence)", fontsize=11.5, pad=8)
        ax.set_xlim(0, xmax)
        ax.set_ylim(len(order) - 0.4, -0.8)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(np.arange(len(order)))
    axes[0].set_yticklabels([MODEL_NAMES[m] for m in order])
    fig.supxlabel(f"Share of next quarter's rat requests that fall in each month's top {frac:.0%} of flagged block groups (%)",
                  fontsize=10.5, color=ps.INK_2, y=0.07)
    fig.suptitle(f"Test year: how much of next quarter's rat activity each model's top {frac:.0%} catches", x=0.01, ha="left",
                 fontsize=13, fontweight="bold", color=ps.INK)
    ps.finish(fig, path, src)


def fig_table(metrics, frac, path, src):
    import matplotlib.pyplot as plt
    ps.setup()
    rows = []
    for sl in SLICES:
        for m in MODEL_NAMES:
            r = metrics[(metrics["slice"] == sl) & (metrics["model"] == m)].iloc[0]
            rows.append(["Recurrence" if sl == "recent" else "All blocks", MODEL_NAMES[m],
                         f"{r.capture_top:.1%} ({r.capture_lo:.1%}-{r.capture_hi:.1%})", f"{r.capture_top / frac:.1f}x",
                         f"{r.spearman:.3f}", f"{r.poisson_dev:.3f}", f"{r.mae:.2f}"])
    fig, ax = plt.subplots(figsize=(12, 0.3 * len(rows) + 1.25))
    ax.axis("off")
    tb = ax.table(cellText=rows, colLabels=["Slice", "Model", f"Top-{frac:.0%} capture (95% CI)", "Lift", "Rank corr.",
                                            "Poisson dev.*", "MAE*"],
                  cellLoc="left", colLoc="left", colWidths=[0.11, 0.3, 0.22, 0.07, 0.1, 0.12, 0.08], bbox=[0, 0, 1, 1])
    tb.auto_set_font_size(False)
    tb.set_fontsize(9.5)
    for (r, c), cell in tb.get_celld().items():
        cell.set_edgecolor(ps.GRID)
        if r == 0:
            cell.set_text_props(weight="bold", color=ps.INK)
        elif rows[r - 1][1] == MODEL_NAMES["xgb_full"]:
            cell.set_facecolor("#cde2fb")
    ax.set_title("Model summary on the held-out test year (forecast: rat requests in the next 3 months)", loc="left", pad=10)
    ps.finish(fig, path, src + "   * lower is better. Lift = capture / share flagged.")


def shap_by_label(sg: pd.DataFrame) -> pd.DataFrame:
    """Sum features that share a plain label (e.g. the two time-of-year terms)."""
    return (sg.groupby(["label", "category"], as_index=False)["mean_abs_shap"].sum()
              .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))


def fig_shap(shap_global, path, src):
    import matplotlib.pyplot as plt
    ps.setup()
    d = shap_by_label(shap_global).head(12).iloc[::-1]
    fixable = {"food", "food_business", "shelter"}
    fig, ax = plt.subplots(figsize=(10.5, 5))
    ax.barh(d["label"], d["mean_abs_shap"], color=[ps.BLUE if c in fixable else ps.GREY for c in d["category"]], height=0.65)
    fig.suptitle("What drives the forecast (blue = conditions people can fix)", x=0.01, ha="left",
                 fontsize=13, fontweight="bold", color=ps.INK)
    ax.set_xlabel("Average impact on the forecast (mean |SHAP|, log scale)")
    ax.grid(axis="y", visible=False)
    ps.finish(fig, path, src)


def fig_equity(eq_q, path, src):
    import matplotlib.pyplot as plt
    ps.setup()
    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = np.arange(len(eq_q))
    ax.bar(x, eq_q["oe"], color=ps.BLUE, width=0.6)
    ax.errorbar(x, eq_q["oe"], yerr=[eq_q["oe"] - eq_q["oe_lo"], eq_q["oe_hi"] - eq_q["oe"]], fmt="none",
                ecolor=ps.INK_2, elinewidth=1, capsize=3)
    ax.axhline(1, color=ps.INK_2, lw=1, ls="--")
    for xi, v in zip(x, eq_q["oe"]):
        ax.text(xi, 0.06, f"{v:.2f}", ha="center", va="bottom", fontsize=10, color="white", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(eq_q["income_quintile"])
    low, high = eq_q["oe"].iloc[0], eq_q["oe"].iloc[-1]
    if abs(low - high) < 0.15:
        title = "Rat requests vs. what physical conditions predict, by neighborhood income"
    elif low < high:
        title = "Lower-income areas report fewer rats than their conditions predict"
    else:
        title = "Higher-income areas report fewer rats than their conditions predict"
    ax.set_title(title)
    ax.set_ylabel("Observed ÷ expected rat requests")
    ax.grid(axis="x", visible=False)
    ps.finish(fig, path, src)


# ---------------------------------------------------------------- main
def run() -> dict:
    cfg = load_config()
    mc = cfg["model"]
    seed = int(mc["seed"])
    reps = int(mc["bootstrap_reps"])
    frac = float(mc["top_k_frac"])
    rng = np.random.default_rng(seed)
    meta = json.load(open(PROCESSED / "build_meta.json"))
    synthetic = meta["data_mode"] == "synthetic"
    src = ("SYNTHETIC TEST DATA - not real DC results." if synthetic else
           f"Source: DC Open Data 311 service requests, {meta['first_month']} to {meta['last_full_month']}; RatGuard analysis.")

    panel = pd.read_parquet(PROCESSED / "panel.parquet")
    if not isinstance(panel["month"].dtype, pd.PeriodDtype):
        panel["month"] = pd.PeriodIndex(panel["month"].astype(str), freq="M")
    bg_static = pd.read_parquet(PROCESSED / "bg_static.parquet")
    wk = pd.read_parquet(PROCESSED / "weekly.parquet")
    end_m = _p(meta["last_full_month"])

    print("[analyze] trend")
    t, tw = trend(panel, bg_static)
    t.to_csv(REPORTS / "trend_monthly.csv", index=False)
    tw.to_csv(REPORTS / "trend_ward_year.csv", index=False)

    print("[analyze] lead-lag test")
    lag, lag_sum = lag_analysis(wk, reps, rng)
    lag.to_csv(REPORTS / "lag_correlation.csv", index=False)

    print("[analyze] forecasting models")
    train, val, test, lab = splits(panel, cfg)
    print(f"  rows: train {train.sum():,} | validation {val.sum():,} | test {test.sum():,}")
    Xtr, ytr = panel.loc[train], panel.loc[train, "rodent_next"]
    Xva, yva = panel.loc[val], panel.loc[val, "rodent_next"]
    Xte, yte = panel.loc[test], panel.loc[test, "rodent_next"]
    preds = {"naive": Xte["rodent_3"].to_numpy(float)}
    preds["glm"] = make_glm(FULL).fit(Xtr[FULL], ytr).predict(Xte[FULL])
    preds["rf"] = make_rf(seed).fit(Xtr[FULL], ytr).predict(Xte[FULL])
    xr = make_xgb(seed).fit(Xtr[RODENT_ONLY], ytr, eval_set=[(Xva[RODENT_ONLY], yva)], verbose=False)
    preds["xgb_rodent"] = xr.predict(Xte[RODENT_ONLY])
    xf = make_xgb(seed).fit(Xtr[FULL], ytr, eval_set=[(Xva[FULL], yva)], verbose=False)
    preds["xgb_full"] = xf.predict(Xte[FULL])
    best_it = int(getattr(xf, "best_iteration", 500) or 500)

    mcode = pd.factorize(Xte["month"].astype(str))[0]
    rows, diffs_all = [], {}
    for sl in SLICES:
        mask = np.ones(len(Xte), bool) if sl == "all" else (Xte["rodent"].to_numpy() >= 1)
        res, diffs = evaluate(yte.to_numpy()[mask], {k: v[mask] for k, v in preds.items()}, mcode[mask],
                              Xte["GEOCODE"].to_numpy()[mask], frac, reps, rng)
        diffs_all[sl] = diffs
        for name, r in res.items():
            r.update({"slice": sl, "model": name, "model_name": MODEL_NAMES[name]})
            rows.append(r)
            print(f"  {sl:6s} {name:10s} capture@{frac:.0%} {r['capture_top']:.1%} [{r['capture_lo']:.1%}-{r['capture_hi']:.1%}]"
                  f"  rank corr {r['spearman']:.3f}  deviance {r['poisson_dev']:.3f}")
    metrics = pd.DataFrame(rows)
    metrics.to_csv(REPORTS / "model_metrics.csv", index=False)
    json.dump(diffs_all, open(REPORTS / "model_differences.json", "w"), indent=2)

    # SHAP on the test year (XGBoost's built-in TreeSHAP)
    contrib = xf.get_booster().predict(xgb.DMatrix(Xte[FULL], feature_names=FULL), pred_contribs=True,
                                       iteration_range=(0, best_it + 1))[:, :-1]
    sg = pd.DataFrame({"feature": FULL, "mean_abs_shap": np.abs(contrib).mean(0)})
    sg["label"] = sg["feature"].map(label)
    sg["category"] = sg["feature"].map(category)
    sg = sg.sort_values("mean_abs_shap", ascending=False)
    sg.to_csv(REPORTS / "shap_global.csv", index=False)
    cat_share = (sg.groupby("category")["mean_abs_shap"].sum() / sg["mean_abs_shap"].sum()).sort_values(ascending=False)

    print("[analyze] final model on all labelled months + latest-month forecast")
    allx = panel.loc[lab]
    final = make_xgb(seed, n_estimators=max(50, int((best_it + 1) * 1.1)), early=False).fit(allx[FULL], allx["rodent_next"], verbose=False)
    latest = panel[panel["month"] == end_m].copy()
    latest["expected_next3"] = final.predict(latest[FULL])
    latest["p_any_next3"] = 1 - np.exp(-latest["expected_next3"])
    latest["risk_pct"] = latest["expected_next3"].rank(pct=True)
    lc = final.get_booster().predict(xgb.DMatrix(latest[FULL], feature_names=FULL), pred_contribs=True)[:, :-1]
    vals = latest[FULL].to_numpy(float)
    latest["drivers"] = [
        json.dumps([{"feature": FULL[j], "contribution": round(float(lc[i, j]), 4),
                     "value": None if np.isnan(vals[i, j]) else round(float(vals[i, j]), 2)}
                    for j in np.argsort(-lc[i])[:6]])
        for i in range(len(latest))
    ]

    print("[analyze] equity audit")
    eq, eq_q, eq_w = equity(panel, bg_static, end_m, seed, reps, rng)
    eq.to_csv(REPORTS / "equity_block_groups.csv", index=False)
    eq_q.to_csv(REPORTS / "equity_by_income.csv", index=False)
    eq_w.to_csv(REPORTS / "equity_by_ward.csv", index=False)

    print("[analyze] figures")
    fig_trend(t[pd.PeriodIndex(t["month"], freq="M") >= _p(mc["start"])], REPORTS / "fig1_trend.png", src)
    fig_lag(lag, lag_sum, REPORTS / "fig2_lag.png", src)
    fig_models(metrics, frac, REPORTS / "fig3_models.png", src)
    fig_table(metrics, frac, REPORTS / "fig4_model_table.png", src)
    fig_shap(sg, REPORTS / "fig5_drivers.png", src)
    if len(eq_q) == 5:
        fig_equity(eq_q, REPORTS / "fig6_equity.png", src)

    # ------------------------------------------------ files for the Streamlit app
    keep = ["GEOCODE", "expected_next3", "p_any_next3", "risk_pct", "drivers", "rodent", "rodent_3", "rodent_12", "food_3",
            "food_12", "shelter_12", "containers_12", "months_since_rodent", "vacant_n", "venues_n", "restaurants_n",
            "grocery_n", "pop_density"]
    sc = latest[keep].merge(bg_static[["GEOCODE", "ward", "neighborhood", "median_hh_income", "POP100"]], on="GEOCODE", how="left")
    sc = sc.merge(eq[["GEOCODE", "observed_12m", "expected_12m", "oe_ratio", "blind_spot", "income_quintile"]], on="GEOCODE", how="left")
    sc["income_quintile"] = sc["income_quintile"].astype(str)
    sc.to_parquet(APP_DATA / "scores.parquet", index=False)
    mon = panel[["GEOCODE", "month", "rodent", "food", "shelter", "containers"]].copy()
    mon["month"] = mon["month"].astype(str)
    mon.to_parquet(APP_DATA / "monthly.parquet", index=False)
    t.to_csv(APP_DATA / "trend.csv", index=False)
    lag.to_csv(APP_DATA / "lag.csv", index=False)
    metrics.to_csv(APP_DATA / "metrics.csv", index=False)
    shap_by_label(sg).to_csv(APP_DATA / "shap_global.csv", index=False)
    eq_q.to_csv(APP_DATA / "equity_income.csv", index=False)
    eq_w.to_csv(APP_DATA / "equity_ward.csv", index=False)

    def pick(sl, m):
        return metrics[(metrics.slice == sl) & (metrics.model == m)].iloc[0]

    rec = test & (panel["rodent"] >= 1)
    out_meta = dict(meta)
    out_meta.update({
        "generated": dt.datetime.now().isoformat(timespec="minutes"),
        "scored_month": str(end_m),
        "test_window": f"{mc['test_start']} to {mc['test_end']}",
        "horizon_months": int(mc["horizon_months"]),
        "top_k_frac": frac,
        "pilot_ward": int(cfg.get("app", {}).get("pilot_ward", 1)),
        "headline": {
            "recurrence_share_any_next3": float((panel.loc[rec, "rodent_next"] > 0).mean()),
            "capture_full_all": float(pick("all", "xgb_full").capture_top),
            "capture_rodent_all": float(pick("all", "xgb_rodent").capture_top),
            "capture_naive_all": float(pick("all", "naive").capture_top),
            "capture_full_recent": float(pick("recent", "xgb_full").capture_top),
            "capture_rodent_recent": float(pick("recent", "xgb_rodent").capture_top),
            "capture_naive_recent": float(pick("recent", "naive").capture_top),
            "diffs": diffs_all,
            "fixable_signal_share": float(cat_share.reindex(["food", "food_business", "shelter"]).fillna(0).sum()),
            "lag": lag_sum,
            "oe_lowest_income": float(eq_q["oe"].iloc[0]) if len(eq_q) else None,
            "oe_highest_income": float(eq_q["oe"].iloc[-1]) if len(eq_q) else None,
            "blind_spots": int(eq["blind_spot"].sum()),
        },
    })
    json.dump(out_meta, open(APP_DATA / "meta.json", "w"), indent=2, default=str)
    write_results(out_meta, metrics, sg, cat_share, synthetic)
    print(f"[analyze] done. Figures + results.md in {REPORTS}; app files in {APP_DATA}")
    return out_meta


def _md(df: pd.DataFrame) -> str:
    """Minimal markdown table (avoids an extra dependency)."""
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r.to_numpy()) + " |")
    return "\n".join(out)


def write_results(meta, metrics, sg, cat_share, synthetic):
    h = meta["headline"]
    L = h["lag"]
    frac = meta["top_k_frac"]
    d_rec = h["diffs"].get("recent", {}).get("xgb_full_minus_xgb_rodent", {})
    d_all = h["diffs"].get("all", {}).get("xgb_full_minus_xgb_rodent", {})
    lines = []
    if synthetic:
        lines += ["> **SYNTHETIC TEST DATA. These numbers are fake and only prove the code runs. "
                  "Run `python run.py all` on real data before quoting anything.**", ""]
    lines += [
        "# RatGuard results (auto-generated)",
        "",
        f"Data: {meta['first_month']} to {meta['last_full_month']}, {meta['block_groups']} census block groups, "
        f"{meta['requests_used']['rodent']:,} rat requests and {meta['requests_used']['food']:,} sanitation requests "
        f"({meta['share_placed_in_block_group']:.1%} of requests had a usable location). "
        f"Test year {meta['test_window']} was never used for training.",
        "",
        "## Sentences for the memo (check every number first)",
        "",
        f"- **Recurrence is the norm:** {h['recurrence_share_any_next3']:.0%} of block groups with a rat request in a given month "
        "had another within 3 months. Finding rats is not the bottleneck; stopping them coming back is.",
        f"- **Targeting:** each month, the {frac:.0%} of block groups RatGuard flags accounted for {h['capture_full_all']:.0%} of the next "
        f"quarter's rat requests (naive 'last quarter' rule: {h['capture_naive_all']:.0%}; random: {frac:.0%}).",
        f"- **Recurrence slice:** among recently treated block groups, the top {frac:.0%} caught {h['capture_full_recent']:.0%} of next-quarter "
        f"requests vs {h['capture_naive_recent']:.0%} for the naive rule.",
        f"- **Does sanitation data add anything?** Rat history only: {h['capture_rodent_all']:.0%}; with sanitation, shelter and food-business "
        f"signals: {h['capture_full_all']:.0%} (difference {d_all.get('point', float('nan')):+.1%}, 95% CI "
        f"{d_all.get('lo', float('nan')):+.1%} to {d_all.get('hi', float('nan')):+.1%}). Recurrence slice: "
        f"{d_rec.get('point', float('nan')):+.1%} ({d_rec.get('lo', float('nan')):+.1%} to {d_rec.get('hi', float('nan')):+.1%}). "
        "Only claim an improvement if the interval is above zero.",
        f"- **Fixable signal:** {h['fixable_signal_share']:.0%} of the model's total SHAP impact comes from conditions people can fix "
        "(sanitation, shelter, food businesses).",
        f"- **Lead-lag:** within the same block group (season removed), sanitation requests 1-4 weeks earlier correlate with rat requests at "
        f"{L['food_leads_mean_corr']:.3f} on average, vs {L['rats_lead_mean_corr']:.3f} in the reverse direction "
        f"(difference {L['difference']:.3f}, 95% CI {L['difference_lo']:.3f} to {L['difference_hi']:.3f}). "
        + ("The interval is above zero: sanitation problems tend to come first." if L["difference_lo"] > 0 else
           "The interval includes zero: no clear evidence that sanitation problems come first."),
    ]
    if h.get("oe_lowest_income") is not None:
        lines.append(f"- **Equity:** given physical conditions, the lowest-income fifth of block groups filed {h['oe_lowest_income']:.2f}x "
                     f"the expected rat requests vs {h['oe_highest_income']:.2f}x in the highest-income fifth. "
                     f"{h['blind_spots']} block groups are flagged as possible blind spots (high expected, low reported). "
                     "A low ratio can mean fewer rats OR under-reporting; field checks decide.")
    lines += ["", "## Model metrics (test year)", "",
              _md(metrics[["slice", "model_name", "capture_top", "capture_lo", "capture_hi", "spearman", "poisson_dev", "mae",
                           "mean_next3", "rows"]]),
              "", "## Signal by driver category (share of mean |SHAP|)", "", _md(cat_share.rename("share").reset_index()),
              "", "## Top drivers", "", _md(sg[["label", "category", "mean_abs_shap"]].head(12)),
              "", "## Method notes (for the appendix)", "",
              "- Unit: 2020 census block group x month. Target: number of 311 'Rodent Inspection and Treatment' requests in the next 3 months.",
              "- Models: naive persistence, Poisson regression, random forest, XGBoost (Poisson objective) with rat history only, "
              "and XGBoost with all signals. The two XGBoost models differ only in their inputs (ablation).",
              "- Time split with a purge gap so no label overlaps the next period: train through the year before validation, "
              "validate (early stopping only), test on a held-out year.",
              f"- Capture rate: each month, flag the top {frac:.0%} of block groups by forecast; share of next-quarter requests that fall in them.",
              "- 95% intervals: 200 bootstrap resamples of whole block groups (keeps each block's months together); "
              "model differences use paired resamples.",
              "- SHAP values from XGBoost's built-in TreeSHAP (pred_contribs).",
              "- Equity: expected requests from a Poisson gradient-boosting model using only physical conditions "
              "(vacant buildings, food businesses, density), 5-fold out-of-fold. Income is used only to audit, never to predict. "
              "Limit: where low income and poor physical conditions sit in the same places, this audit cannot fully separate them "
              "and will understate under-reporting.",
              "- Caveat: a 311 request is not a confirmed rat. In FY17 only 46.1% of rodent requests led to a burrow treatment (DC CapSTAT, 2017)."]
    (REPORTS / "results.md").write_text("\n".join(lines), encoding="utf-8")
