# RatGuard DC

**Find the food. Fix the source. Prove it worked.**

RatGuard turns DC's open data into a block-by-block rat-prevention tool:

1. **Sense:** 311 rat and sanitation requests, plus vacant buildings, food businesses, census and weather data.
2. **Diagnose:** forecast rat requests for the next 3 months in every census block group, and explain why with SHAP.
3. **Fix:** turn the drivers into fixes, non-toxic first, each routed to the owner who can make it.
4. **Verify:** compare each pilot block with matched control blocks, and audit where reporting may be missing.

> **The repo ships with SYNTHETIC data** so the app opens immediately. Every page shows a red banner while it does.
> Run the real pipeline (step 2 below) before you quote any number or demo to judges.

---

## 1. Set up (5 minutes)

You need Python 3.10–3.13.

```bash
git clone <your-repo-url> ratguard-dc && cd ratguard-dc
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-pipeline.txt
```

## 2. Run on real DC data

```bash
python run.py download    # 10-30 min the first time; cached afterwards
python run.py types       # check the 311 service types RatGuard uses, per year
python run.py build       # block group x month panel + features
python run.py analyze     # models, lag test, equity audit, figures, app data
# or all at once:
python run.py all
```

- **If a download stops part-way,** run it again. Finished years are cached, and only the current year re-downloads.
- **To limit years while testing:** `python run.py download --years 2024 2025`
- **If DC renames a 311 type,** `python run.py types` writes `data/service_types_by_year.csv`. Edit `service_groups` in `config.yaml` to match.
- **DC already renamed the rat category once.** From 1 May 2026, rat requests appear as "DC Health Rodent & Vector Control" (same service code, S0311). `config.yaml` includes both names. If a future rename happens, `build` stops the data at the last healthy month and prints a "Rat requests collapse" warning. The app and `results.md` show the same warning.

## 3. Run the app

```bash
streamlit run app.py
```

If the map is blank, switch **Basemap → Plain** in the sidebar. The street tiles need internet access.

### Optional: live AI features

Live AI tags uploaded photos and drafts notices in English or Spanish. To turn it on, create `.streamlit/secrets.toml`:

```toml
ANTHROPIC_API_KEY = "sk-ant-..."
RATGUARD_MODEL = "claude-haiku-5-5"   # optional
```

Without a key, both features use templates and say so on screen. A person must tick "reviewed" before any notice can be downloaded.

## 4. Deploy free on Streamlit Community Cloud

1. Run step 2 so that `app_data/` holds **real** results, then commit `app_data/`. The `data/` folder is git-ignored and stays local.
2. Push the repo to GitHub. A public repo is simplest.
3. Go to **share.streamlit.io**, choose **New app**, then pick the repo, the `main` branch and `app.py`.
4. Optional: under **Advanced settings → Secrets**, paste the two lines above.
5. Deploy, and put the URL in the slides and the memo.

The cloud app installs only `requirements.txt` (no geopandas or xgboost). It reads the small precomputed files in `app_data/`.

---

## What goes where in the memo and deck

| File (in `reports/`) | Use it for |
|---|---|
| `results.md` | Auto-written sentences with the real numbers. **Check each one before pasting.** |
| `fig4_model_table.png` | **Appendix: the required "model summary or data table" visual** |
| `fig2_lag.png` | Deck: "Do sanitation problems come before rat requests?" |
| `fig1_trend.png` | Deck/memo problem section: rat requests over time |
| `fig3_models.png` | Deck: how much of next quarter's rat activity each model's top 10% catches |
| `fig5_drivers.png` | Deck/demo: what drives the forecast (blue = fixable) |
| `fig6_equity.png` | Memo: equity audit by neighborhood income |
| `rodent_details_top40.csv` | What inspectors write in 311 rat requests. If it shows "burrows treated", it could become a better outcome label |

## Method (for the appendix)

- **Unit and target.** The unit is a 2020 census block group × month (about 570 block groups). The target is the number of 311 *Rodent Inspection and Treatment* requests in the next 3 months.
- **Why a count, not yes/no.** In busy wards almost every block group gets a rat request every quarter, so a yes/no target saturates. Ranking by expected count is what an inspector or BID needs.
- **Models:**
  1. Naive persistence: next quarter equals last quarter.
  2. Poisson regression.
  3. Random forest.
  4. XGBoost (Poisson objective) with rat history only.
  5. XGBoost with all signals.

  Models 4 and 5 differ only in their inputs. That **ablation** answers "does sanitation data add anything?"
- **Features:**
  - Rat history: 1, 3 and 12 months, months active, and months since the last request.
  - Sanitation requests: food sources (trash, dumping, alley cleaning…).
  - Shelter requests: vacant lots, abandoned cars, overgrowth.
  - Rat-resistant can requests.
  - Context: vacant and blighted buildings, ABCA-licensed bars and restaurants, grocery stores, population density.
  - Seasonality and weather.
- **Validation.** The split is by time with a purge gap: train on January 2021 to September 2023, validate in 2024 (early stopping only), and test on 2025, which no model ever saw.
- **Headline metric: capture rate.** Each month, flag the top 10% of block groups by forecast. Capture is the share of next quarter's rat requests that fall in them; random targeting captures about 10%. The table also reports rank correlation, Poisson deviance and MAE.
- **Uncertainty.** 95% intervals come from 200 bootstrap resamples of whole block groups. Model differences use paired resamples.
- **Explanations.** XGBoost's built-in TreeSHAP gives per-block drivers in the app and global importance in fig5.
- **Lead-lag test.** This is the weekly correlation, within each block group and with season removed, between rat requests and sanitation requests k weeks earlier. Negative lags act as a placebo.
- **Equity audit.** Expected rat requests come from **physical conditions only** (vacancy, food businesses, density), using 5-fold out-of-fold predictions. Comparing observed with expected gives a ratio by income fifth and by ward. Income is used only to audit, never to predict.

### Be honest about the limits (judges will ask)

- **A 311 request is not a confirmed rat.** In FY17, 46.1% of rodent requests led to a burrow treatment (DC CapSTAT, Nov 2017).
- **The equity audit has a blind spot of its own.** Where low income and poor physical conditions overlap geographically, it cannot fully separate them, so it will *understate* under-reporting. Treat flagged blind spots as places for a field check, not proof.
- **The 2017–18 Lab @ DC model ranked complaints well but could not find unreported burrows.** RatGuard forecasts *recurrence and volume* and leads with fixing sources; it does not claim to find hidden burrows.
- **What is real vs simulated in the prototype:**
  - Real: forecasts, drivers, fix plans, matched controls and blind spots, all computed from data.
  - Real only with an API key: photo tagging and AI-drafted notices.
  - Simulated: filing to 311, routing to agencies, and pilot "after" values, which come from a future pilot.

## Project layout

```
run.py                   command line: download | types | build | analyze | all | synthetic
config.yaml              endpoints, 311 type groups, model dates
ratguard/arcgis.py       paged ArcGIS REST client (maps2.dcgis.dc.gov)
ratguard/download.py     311 per year, block groups, income, vacancy, licensees, grocery, neighborhoods, wards, weather
ratguard/build.py        spatial joins -> block group x month panel and features
ratguard/analyze.py      trend, lag test, models, SHAP, equity, figures, app data, results.md
ratguard/fixes.py        plain-language driver labels and driver -> fix rules (non-toxic first)
ratguard/synthetic.py    fake data in the same format, for offline testing
app.py                   Streamlit app (reads app_data/ only)
tests/test_download_mock.py   offline test of the downloader against a fake ArcGIS server
```

Testing offline: `python run.py all --synthetic`, then `python run.py synthetic && python -m tests.test_download_mock`.

## Data sources

- DC Open Data / DCGIS REST services (maps2.dcgis.dc.gov): All Service Requests (311) by year; Census Block Groups 2020; ACS 5-Year Economic Characteristics of DC Census Tracts; Vacant and Blighted Building Addresses; ABCA Liquor License Locations; Grocery Store Locations; Neighborhood Clusters; Ward 2022.
- Daily weather: Open-Meteo historical weather API (Reagan National area).
