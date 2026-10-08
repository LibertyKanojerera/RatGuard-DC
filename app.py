"""RatGuard DC - Streamlit demo.

Sense -> Diagnose -> Fix -> Verify, plus the evidence behind the model.
Reads only the small precomputed files in app_data/ (made by `python run.py analyze`).

Optional: set ANTHROPIC_API_KEY in .streamlit/secrets.toml (or the environment) to turn on
live AI photo tagging and AI-drafted notices. Without a key those features use templates
and are clearly labelled as simulated.
"""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ratguard import fixes

APP = Path(__file__).resolve().parent / "app_data"
BLUE, GREY, INK, INK2, GRID, CRIT = "#2a78d6", "#898781", "#0b0b0b", "#52514e", "#e1e0d9", "#d03b3b"
SEQ = ["#e9f2fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CENTER = {"lat": 38.905, "lon": -77.016}

st.set_page_config(page_title="RatGuard DC", layout="wide")


# ------------------------------------------------------------------ data
@st.cache_data
def load():
    if not (APP / "meta.json").exists():
        return None
    d = {
        "meta": json.loads((APP / "meta.json").read_text()),
        "scores": pd.read_parquet(APP / "scores.parquet"),
        "monthly": pd.read_parquet(APP / "monthly.parquet"),
        "geo": json.loads((APP / "bg.geojson").read_text()),
        "trend": pd.read_csv(APP / "trend.csv"),
        "lag": pd.read_csv(APP / "lag.csv"),
        "metrics": pd.read_csv(APP / "metrics.csv"),
        "shap": pd.read_csv(APP / "shap_global.csv"),
        "eq_income": pd.read_csv(APP / "equity_income.csv"),
        "eq_ward": pd.read_csv(APP / "equity_ward.csv"),
    }
    s = d["scores"]
    s["GEOCODE"] = s["GEOCODE"].astype(str)
    s["ward_label"] = s["ward"].map(lambda w: f"Ward {int(w)}" if pd.notna(w) else "Ward ?")
    def short(n):
        n = (n or "Unnamed area").strip()
        return n if len(n) <= 48 else n[:46].rsplit(",", 1)[0] + "…"
    s["place"] = s.apply(lambda r: f"{short(r['neighborhood'])} · {r['ward_label']} · BG {r['GEOCODE'][-7:]}", axis=1)
    d["monthly"]["GEOCODE"] = d["monthly"]["GEOCODE"].astype(str)
    return d


def secret(name: str, default=None):
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name, default)


def ai_client():
    key = secret("ANTHROPIC_API_KEY")
    if not key:
        return None
    try:
        import anthropic
        return anthropic.Anthropic(api_key=key)
    except Exception:
        return None


AI_MODEL = secret("RATGUARD_MODEL", "claude-haiku-5-5")


def style(fig, height=360):
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", size=13, color=INK2),
        title_font=dict(size=15, color=INK), hoverlabel=dict(bgcolor="white"),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def bg_map(df, col, title, geo, selected=None, label=None, height=520, scale=SEQ):
    fig = px.choropleth_map(
        df, geojson=geo, locations="GEOCODE", featureidkey="properties.GEOCODE", color=col,
        color_continuous_scale=scale, map_style=st.session_state.get("basemap_style", "carto-positron"),
        zoom=10.35, center=CENTER, opacity=0.78,
        hover_name="place", hover_data={col: ":.1f", "GEOCODE": False}, labels={col: label or col},
    )
    if selected is not None:
        sel = [f for f in geo["features"] if f["properties"]["GEOCODE"] == selected]
        if sel:
            fig.add_trace(go.Choroplethmap(
                geojson={"type": "FeatureCollection", "features": sel}, locations=[selected], z=[1],
                featureidkey="properties.GEOCODE", colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
                showscale=False, marker_line_width=4, marker_line_color=CRIT, hoverinfo="skip"))
    fig.update_layout(title=title, coloraxis_colorbar=dict(title=label or col, thickness=12))
    return style(fig, height)


# ------------------------------------------------------------------ page
D = load()
if D is None:
    st.error("No app data yet. Run `python run.py all` (real data) or `python run.py all --synthetic` (test data).")
    st.stop()

meta, S, M = D["meta"], D["scores"], D["monthly"]
synthetic = meta.get("data_mode") == "synthetic"
H = meta["headline"]
frac = meta.get("top_k_frac", 0.1)

st.title("RatGuard DC")
st.markdown("**Find the food. Fix the source. Prove it worked.**")
st.caption(f"Data: DC 311 service requests {meta['first_month']} to {meta['last_full_month']} · "
           f"{meta['block_groups']} census block groups · forecast window: next {meta['horizon_months']} months · "
           f"built {meta['generated']}")
if synthetic:
    st.error("SYNTHETIC TEST DATA. Every number on this page is fake and only shows that the app works. "
             "Run `python run.py all` on real DC data before any demo.", icon=":material/warning:")

with st.sidebar:
    st.header("Choose a block")
    wards = sorted(int(w) for w in S["ward"].dropna().unique())
    pilot = meta.get("pilot_ward", 1)
    ward_pick = st.selectbox("Ward", ["All wards"] + [f"Ward {w}" for w in wards],
                             index=(wards.index(pilot) + 1) if pilot in wards else 0)
    pool = S if ward_pick == "All wards" else S[S["ward_label"] == ward_pick]
    pool = pool.sort_values("expected_next3", ascending=False)
    choice = st.selectbox("Block group (highest forecast first)", pool["place"].tolist(), index=0)
    B = S[S["place"] == choice].iloc[0]
    st.divider()
    base = st.radio("Basemap", ["Streets", "Plain (works offline)"], horizontal=True)
    st.session_state["basemap_style"] = "carto-positron" if base == "Streets" else "white-bg"
    ai = ai_client()
    st.caption("AI features: " + (f"live ({AI_MODEL})" if ai else "simulated (add ANTHROPIC_API_KEY to enable)"))

drivers = json.loads(B["drivers"])
pos_drivers = [d for d in drivers if d["contribution"] > 0]
plan = fixes.plan_for(drivers)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["1 · Sense", "2 · Diagnose", "3 · Fix", "4 · Verify", "Evidence"])

# ------------------------------------------------------------------ 1 SENSE
with tab1:
    T = D["trend"].copy()
    T["date"] = pd.PeriodIndex(T["month"], freq="M").to_timestamp()
    last12, prev12 = T["rodent"].iloc[-12:].sum(), T["rodent"].iloc[-24:-12].sum()
    c = st.columns(4)
    c[0].metric("Rat requests, last 12 months", f"{last12:,.0f}", f"{(last12 / prev12 - 1):+.0%} vs prior 12" if prev12 else None,
                delta_color="inverse")
    c[1].metric("Sanitation requests, last 12 months", f"{T['food'].iloc[-12:].sum():,.0f}")
    recent_bg = M[M["month"] >= T["month"].iloc[-3]].groupby("GEOCODE")["rodent"].sum()
    c[2].metric("Blocks with rat requests (3 mo)", f"{(recent_bg > 0).sum():,} of {len(S):,}",
                help="Census block groups with at least one rat request in the last 3 months.")
    c[3].metric("Recurrence within 3 months", f"{H['recurrence_share_any_next3']:.0%}",
                help="Share of block groups with a rat request in a month that had another in the next 3 months (test year).")

    left, right = st.columns([1.1, 1])
    with left:
        fig = px.line(T[T["date"] >= T["date"].max() - pd.DateOffset(years=5)], x="date", y="rodent",
                      labels={"date": "", "rodent": "Rat requests per month"})
        fig.update_traces(line_color=BLUE, line_width=2)
        st.plotly_chart(style(fig.update_layout(title="Rat requests to 311, citywide"), 330))
        st.markdown("##### Report a problem (resident view)")
        up = st.file_uploader("Photo of a rat, burrow, overflowing trash or dumping", type=["jpg", "jpeg", "png", "webp"])
        if up is not None:
            st.image(up, width=260)
            if st.button("Tag photo and file to 311"):
                if ai:
                    try:
                        mime = up.type or "image/jpeg"
                        msg = ai.messages.create(
                            model=AI_MODEL, max_tokens=300,
                            messages=[{"role": "user", "content": [
                                {"type": "image", "source": {"type": "base64", "media_type": mime,
                                                             "data": base64.b64encode(up.getvalue()).decode()}},
                                {"type": "text", "text": "You triage photos for a city rat-prevention service. Reply with JSON only: "
                                 '{"category": one of ["rat","burrow","overflowing trash","illegal dumping","other"], '
                                 '"severity": one of ["low","medium","high"], "reason": "<15 words", '
                                 '"people_or_plates_visible": true/false}'}]}])
                        txt = msg.content[0].text
                        res = json.loads(txt[txt.find("{"): txt.rfind("}") + 1])
                        st.success(f"AI tag: **{res['category']}** · severity **{res['severity']}** · {res['reason']}")
                        if res.get("people_or_plates_visible"):
                            st.warning("People or plates visible: blur before storing (planned feature).")
                    except Exception as e:
                        st.warning(f"AI tagging failed ({type(e).__name__}); showing the simulated result.")
                        st.info("Simulated tag: **overflowing trash** · severity **medium**")
                else:
                    st.info("Simulated tag: **overflowing trash** · severity **medium**. "
                            "Add an API key to tag photos with a real vision model.")
                st.caption("Prototype: filing to DC 311 is simulated. Planned: blur faces and plates before storage.")
    with right:
        S["rat_3m"] = S["GEOCODE"].map(recent_bg).fillna(0)
        st.plotly_chart(bg_map(S, "rat_3m", "Rat requests, last 3 months", D["geo"], B["GEOCODE"], "Requests"))

# ------------------------------------------------------------------ 2 DIAGNOSE
with tab2:
    left, right = st.columns([1, 1.1])
    with left:
        st.subheader(B["place"])
        c = st.columns(3)
        top = max(1, int(round((1 - B["risk_pct"]) * 100)))
        c[0].metric("Forecast (next 3 mo)", f"{B['expected_next3']:.1f}", help="Expected 311 rat requests in the next 3 months.")
        c[1].metric("Risk rank in DC", f"Top {top}%", help="Share of DC block groups with a forecast at least this high.")
        c[2].metric("Rat requests (12 mo)", f"{B['rodent_12']:.0f}")
        st.markdown("##### Why: top drivers raising the forecast")
        if pos_drivers:
            dd = pd.DataFrame([{"Driver": fixes.describe_driver(d), "Impact": d["contribution"],
                                "fixable": fixes.category(d["feature"]) in ("food", "food_business", "shelter")}
                               for d in pos_drivers[:5]])
            fig = px.bar(dd.iloc[::-1], x="Impact", y="Driver", orientation="h",
                         color="fixable", color_discrete_map={True: BLUE, False: GREY})
            fig.update_layout(showlegend=False, xaxis_title="Impact (SHAP)", yaxis_title="")
            st.plotly_chart(style(fig, 260))
            st.caption("Blue = conditions people can fix (food, shelter). Grey = rat history, season, density.")
        else:
            st.info("Nothing is pushing this block's forecast above the citywide average.")
        hist = M[M["GEOCODE"] == B["GEOCODE"]].tail(24).copy()
        hist["date"] = pd.PeriodIndex(hist["month"], freq="M").to_timestamp()
        h1, h2 = st.columns(2)
        f1 = px.bar(hist, x="date", y="rodent", labels={"date": "", "rodent": ""}, title="Rat requests, last 24 months")
        f1.update_traces(marker_color=BLUE)
        f2 = px.bar(hist, x="date", y="food", labels={"date": "", "food": ""}, title="Sanitation requests, last 24 months")
        f2.update_traces(marker_color=GREY)
        h1.plotly_chart(style(f1, 230))
        h2.plotly_chart(style(f2, 230))
    with right:
        st.plotly_chart(bg_map(S, "expected_next3", "Forecast: rat requests in the next 3 months", D["geo"], B["GEOCODE"],
                               "Expected"))

# ------------------------------------------------------------------ 3 FIX
with tab3:
    st.subheader(f"Fix plan for {B['place']}")
    st.caption("Ordered non-toxic first. Poison is the last resort, used only if source fixes and safer treatments fail.")
    pdf = pd.DataFrame(plan).rename(columns={"driver": "Why", "action": "Fix", "owner": "Who can do it", "toxicity": "Toxicity"})
    st.dataframe(pdf[["Fix", "Who can do it", "Toxicity", "Why"]], hide_index=True)

    st.markdown("##### Send a fix request")
    c = st.columns([1.4, 1, 0.8])
    row = c[0].selectbox("Fix", [f"{p['action']} — {p['owner']}" for p in plan])
    lang = c[1].selectbox("Language", ["English", "Español"])
    use_ai = c[2].toggle("AI draft", value=bool(ai), disabled=not ai)
    p = plan[[f"{q['action']} — {q['owner']}" for q in plan].index(row)]
    reasons = [fixes.describe_driver(d) for d in pos_drivers[:3]] or ["repeated rat requests nearby"]
    if st.button("Draft notice"):
        text = fixes.notice_template(B["place"], p["owner"], p["action"].lower(), reasons, "es" if lang == "Español" else "en")
        if use_ai and ai:
            try:
                prompt = (f"Write a short, friendly, non-threatening notice ({'in Spanish' if lang == 'Español' else 'in English'}, "
                          f"under 140 words) to {p['owner']} asking them to: {p['action']}. Location: {B['place']}, Washington DC. "
                          f"Reasons from city data: {'; '.join(reasons)}. Make clear it is a request for help, not a fine, "
                          "and offer support (lidded containers, BID help). Plain text, no markdown.")
                msg = ai.messages.create(model=AI_MODEL, max_tokens=400, messages=[{"role": "user", "content": prompt}])
                text = msg.content[0].text.strip()
                st.caption(f"Drafted by {AI_MODEL}. A person must review before sending.")
            except Exception as e:
                st.warning(f"AI draft failed ({type(e).__name__}); using the template.")
        else:
            st.caption("Template draft. Add an API key to draft with AI.")
        st.session_state["notice"] = text
    if "notice" in st.session_state:
        txt = st.text_area("Review and edit before sending", st.session_state["notice"], height=230)
        ok = st.checkbox("I reviewed this notice (human approval required)")
        st.download_button("Download notice", txt, file_name="ratguard_notice.txt", disabled=not ok)

# ------------------------------------------------------------------ 4 VERIFY
with tab4:
    st.subheader("How the pilot proves it worked")
    st.markdown("RatGuard compares the pilot block with **matched control blocks**: similar rat history, sanitation load, "
                "food businesses, vacancy and density, but no RatGuard fixes. Success = fewer repeat rat requests than the controls.")
    feats = ["rodent_12", "food_12", "shelter_12", "venues_n", "vacant_n", "pop_density"]
    X = S[feats].astype(float).apply(np.log1p)
    Z = (X - X.mean()) / X.std(ddof=0).replace(0, 1)
    dist = np.sqrt(((Z - Z.loc[B.name]) ** 2).sum(axis=1))
    cand = S.assign(distance=dist)
    cand = cand[(cand["GEOCODE"] != B["GEOCODE"]) & (cand["neighborhood"] != B["neighborhood"])]
    ctrl = cand.nsmallest(3, "distance")
    tbl = pd.concat([S.loc[[B.name]].assign(role="Pilot"), ctrl.assign(role="Control")])
    show = tbl[["role", "place", "rodent_12", "food_12", "venues_n", "vacant_n", "expected_next3"]].rename(columns={
        "role": "Role", "place": "Block group", "rodent_12": "Rat requests (12 mo)", "food_12": "Sanitation requests (12 mo)",
        "venues_n": "Bars/restaurants", "vacant_n": "Vacant buildings", "expected_next3": "Forecast (3 mo)"})
    st.dataframe(show, hide_index=True,
                 column_config={"Forecast (3 mo)": st.column_config.NumberColumn(format="%.1f")})
    st.markdown("##### What we track (baseline from data; after-values come from the pilot)")
    kpi = pd.DataFrame([
        ["Repeat rat requests per quarter, pilot vs controls", f"{B['rodent_3']:.0f} vs {ctrl['rodent_3'].mean():.1f} (last 3 months)", "Pilot"],
        ["Days from fix request to fix done", "Not tracked today", "Pilot"],
        ["Share of fix requests completed", "Not tracked today", "Pilot"],
        ["Rodenticide treatments avoided", "Not tracked today", "Pilot (DC Health records)"],
        ["Requests per 1,000 residents, by ward", "See blind spots below", "Ongoing"],
    ], columns=["Metric", "Baseline", "After"])
    st.dataframe(kpi, hide_index=True)

    st.markdown("##### Blind spots: high expected, low reported")
    st.caption("Expected rat requests come from physical conditions only (vacant buildings, food businesses, density). "
               "A low ratio can mean fewer rats or under-reporting; a field check decides. Income is used only to audit.")
    bs = S[S["blind_spot"].fillna(False)].sort_values("expected_12m", ascending=False)
    st.dataframe(bs[["place", "observed_12m", "expected_12m", "oe_ratio", "income_quintile"]].head(12).rename(columns={
        "place": "Block group", "observed_12m": "Reported (12 mo)", "expected_12m": "Expected (12 mo)",
        "oe_ratio": "Reported ÷ expected", "income_quintile": "Income fifth"}), hide_index=True,
        column_config={"Expected (12 mo)": st.column_config.NumberColumn(format="%.1f"),
                       "Reported ÷ expected": st.column_config.NumberColumn(format="%.2f")})

# ------------------------------------------------------------------ EVIDENCE
with tab5:
    mt = D["metrics"]
    dif = H.get("diffs", {}).get("all", {}).get("xgb_full_minus_xgb_rodent", {})
    st.subheader("Does it work? Tested on a year the model never saw")
    c = st.columns(3)
    c[0].metric(f"Rat requests caught by the top {frac:.0%} (RatGuard)", f"{H['capture_full_all']:.0%}",
                f"{(H['capture_full_all'] - H['capture_naive_all']) * 100:+.1f} pts vs naive rule")
    c[1].metric("Naive rule (next quarter = last quarter)", f"{H['capture_naive_all']:.0%}")
    c[2].metric("Random targeting", f"{frac:.0%}")
    if dif:
        st.caption(f"Adding sanitation, shelter and food-business signals to rat history changed capture by {dif['point']:+.1%} "
                   f"(95% CI {dif['lo']:+.1%} to {dif['hi']:+.1%}).")
    order = ["naive", "glm", "rf", "xgb_rodent", "xgb_full"]
    left, right = st.columns(2)
    with left:
        d = mt[mt["slice"] == "all"].set_index("model").loc[order].reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=d["capture_top"] * 100, y=d["model_name"], mode="markers",
            marker=dict(size=12, color=[BLUE if m == "xgb_full" else GREY for m in d["model"]]),
            error_x=dict(type="data", symmetric=False, array=(d["capture_hi"] - d["capture_top"]) * 100,
                         arrayminus=(d["capture_top"] - d["capture_lo"]) * 100, color=INK2),
            hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
        fig.add_vline(x=frac * 100, line_dash="dash", line_color=INK2, annotation_text="random")
        fig.update_layout(title=f"Share of next-quarter rat requests in each model's top {frac:.0%}",
                          xaxis=dict(title="% of next-quarter rat requests", range=[0, min(100, d["capture_hi"].max() * 100 + 10)]),
                          yaxis=dict(autorange="reversed"), showlegend=False)
        st.plotly_chart(style(fig, 330))
    with right:
        L = D["lag"]
        fig = px.bar(L, x="lag_weeks", y="corr", labels={"lag_weeks": "Weeks sanitation requests come first", "corr": "Correlation"})
        fig.update_traces(marker_color=[BLUE if k >= 1 else GREY for k in L["lag_weeks"]],
                          error_y=dict(type="data", symmetric=False, array=L["hi"] - L["corr"], arrayminus=L["corr"] - L["lo"],
                                       color=INK2))
        lg = H["lag"]
        fig.update_layout(title="Do sanitation problems come before rat requests?" + (" Yes." if lg["difference_lo"] > 0 else ""))
        st.plotly_chart(style(fig, 330))
    left, right = st.columns(2)
    with left:
        sg = D["shap"].head(10).iloc[::-1]
        fig = px.bar(sg, x="mean_abs_shap", y="label", orientation="h",
                     color=sg["category"].isin(["food", "food_business", "shelter"]), color_discrete_map={True: BLUE, False: GREY})
        fig.update_layout(title="What drives the forecast (blue = fixable)", showlegend=False,
                          xaxis_title="Mean |SHAP|", yaxis_title="")
        st.plotly_chart(style(fig, 360))
    with right:
        E = D["eq_income"]
        if len(E):
            fig = px.bar(E, x="income_quintile", y="oe", labels={"income_quintile": "", "oe": "Reported ÷ expected"})
            fig.update_traces(marker_color=BLUE, error_y=dict(type="data", symmetric=False, array=E["oe_hi"] - E["oe"],
                                                              arrayminus=E["oe"] - E["oe_lo"], color=INK2))
            fig.add_hline(y=1, line_dash="dash", line_color=INK2)
            fig.update_layout(title="Equity audit: reported vs expected, by neighborhood income")
            st.plotly_chart(style(fig, 360))
    with st.expander("Full model table and method notes"):
        st.dataframe(mt[["slice", "model_name", "capture_top", "capture_lo", "capture_hi", "spearman", "poisson_dev", "mae"]],
                     hide_index=True)
        st.markdown(
            "- **Unit:** 2020 census block group × month. **Target:** number of 311 *Rodent Inspection and Treatment* requests in the next 3 months.\n"
            "- **Models:** naive persistence, Poisson regression, random forest, XGBoost (Poisson) with rat history only, and with all signals.\n"
            f"- **Test:** {meta['test_window']}, never used for training; time split with a purge gap.\n"
            "- **Intervals:** 200 bootstrap resamples of whole block groups.\n"
            "- **Caveat:** a 311 request is not a confirmed rat (in FY17, 46.1% of rodent requests led to a burrow treatment, DC CapSTAT).\n"
            "- **Data:** DC Open Data (311 requests, census block groups, ACS income, vacant buildings, ABCA licensees, grocery stores, "
            "neighborhood clusters, wards); weather from Open-Meteo.")
