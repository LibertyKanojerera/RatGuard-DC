> **SYNTHETIC TEST DATA. These numbers are fake and only prove the code runs. Run `python run.py all` on real data before quoting anything.**

# RatGuard results (auto-generated)

Data: 2020-01 to 2026-09, 338 census block groups, 48,882 rat requests and 612,129 sanitation requests (100.0% of requests had a usable location). Test year 2025-01 to 2025-12 was never used for training.

## Sentences for the memo (check every number first)

- **Recurrence is the norm:** 93% of block groups with a rat request in a given month had another within 3 months. Finding rats is not the bottleneck; stopping them coming back is.
- **Targeting:** each month, the 10% of block groups RatGuard flags accounted for 31% of the next quarter's rat requests (naive 'last quarter' rule: 29%; random: 10%).
- **Recurrence slice:** among recently treated block groups, the top 10% caught 26% of next-quarter requests vs 25% for the naive rule.
- **Does sanitation data add anything?** Rat history only: 31%; with sanitation, shelter and food-business signals: 31% (difference +0.2%, 95% CI -0.4% to +0.4%). Recurrence slice: -0.1% (-0.8% to +0.5%). Only claim an improvement if the interval is above zero.
- **Fixable signal:** 8% of the model's total SHAP impact comes from conditions people can fix (sanitation, shelter, food businesses).
- **Lead-lag:** within the same block group (season removed), sanitation requests 1-4 weeks earlier correlate with rat requests at 0.049 on average, vs 0.013 in the reverse direction (difference 0.036, 95% CI 0.030 to 0.040). The interval is above zero: sanitation problems tend to come first.
- **Equity:** given physical conditions, the lowest-income fifth of block groups filed 0.93x the expected rat requests vs 1.03x in the highest-income fifth. 61 block groups are flagged as possible blind spots (high expected, low reported). A low ratio can mean fewer rats OR under-reporting; field checks decide.

## Model metrics (test year)

| slice | model_name | capture_top | capture_lo | capture_hi | spearman | poisson_dev | mae | mean_next3 | rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | Naive: next 3 months = last 3 months | 0.287 | 0.250 | 0.310 | 0.701 | 4.652 | 3.333 | 5.261 | 4056 |
| all | Poisson regression | 0.312 | 0.282 | 0.329 | 0.796 | 1.714 | 2.179 | 5.261 | 4056 |
| all | Random forest | 0.309 | 0.280 | 0.328 | 0.795 | 1.775 | 2.230 | 5.261 | 4056 |
| all | XGBoost, rat history only | 0.307 | 0.278 | 0.327 | 0.792 | 1.775 | 2.219 | 5.261 | 4056 |
| all | XGBoost, full RatGuard | 0.309 | 0.279 | 0.327 | 0.794 | 1.749 | 2.200 | 5.261 | 4056 |
| recent | Naive: next 3 months = last 3 months | 0.249 | 0.219 | 0.268 | 0.656 | 4.001 | 4.241 | 7.116 | 2474 |
| recent | Poisson regression | 0.264 | 0.239 | 0.282 | 0.767 | 1.794 | 2.673 | 7.116 | 2474 |
| recent | Random forest | 0.260 | 0.237 | 0.279 | 0.763 | 1.877 | 2.728 | 7.116 | 2474 |
| recent | XGBoost, rat history only | 0.259 | 0.235 | 0.278 | 0.758 | 1.869 | 2.716 | 7.116 | 2474 |
| recent | XGBoost, full RatGuard | 0.258 | 0.234 | 0.279 | 0.761 | 1.845 | 2.694 | 7.116 | 2474 |

## Signal by driver category (share of mean |SHAP|)

| category | share |
| --- | --- |
| colony | 0.623 |
| season | 0.277 |
| food | 0.046 |
| shelter | 0.026 |
| containers | 0.011 |
| density | 0.011 |
| food_business | 0.006 |

## Top drivers

| label | category | mean_abs_shap |
| --- | --- | --- |
| Rat requests, last 12 months | colony | 0.557 |
| Time of year | season | 0.175 |
| Months with rat requests (last 12) | colony | 0.163 |
| Time of year | season | 0.129 |
| Rat requests, last 3 months | colony | 0.031 |
| Sanitation requests this month | food | 0.029 |
| Monthly temperature | season | 0.029 |
| Vacant or blighted buildings | shelter | 0.014 |
| Sanitation requests, last 12 months | food | 0.013 |
| Shelter requests (vacant lots, abandoned cars), 12 mo | shelter | 0.013 |
| Rat-resistant can requests, last 12 months | containers | 0.012 |
| Population density | density | 0.011 |

## Method notes (for the appendix)

- Unit: 2020 census block group x month. Target: number of 311 'Rodent Inspection and Treatment' requests in the next 3 months.
- Models: naive persistence, Poisson regression, random forest, XGBoost (Poisson objective) with rat history only, and XGBoost with all signals. The two XGBoost models differ only in their inputs (ablation).
- Time split with a purge gap so no label overlaps the next period: train through the year before validation, validate (early stopping only), test on a held-out year.
- Capture rate: each month, flag the top 10% of block groups by forecast; share of next-quarter requests that fall in them.
- 95% intervals: 200 bootstrap resamples of whole block groups (keeps each block's months together); model differences use paired resamples.
- SHAP values from XGBoost's built-in TreeSHAP (pred_contribs).
- Equity: expected requests from a Poisson gradient-boosting model using only physical conditions (vacant buildings, food businesses, density), 5-fold out-of-fold. Income is used only to audit, never to predict. Limit: where low income and poor physical conditions sit in the same places, this audit cannot fully separate them and will understate under-reporting.
- Caveat: a 311 request is not a confirmed rat. In FY17 only 46.1% of rodent requests led to a burrow treatment (DC CapSTAT, 2017).