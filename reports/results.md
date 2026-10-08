# RatGuard results (auto-generated)

Data: 2020-01 to 2026-09, 571 census block groups, 90,880 rat requests and 929,717 sanitation requests (100.0% of requests had a usable location). Test year 2025-01 to 2025-12 was never used for training.

## Sentences for the memo (check every number first)

- **Recurrence is the norm:** 93% of block groups with a rat request in a given month had another within 3 months. Finding rats is not the bottleneck; stopping them coming back is.
- **Targeting:** each month, the 10% of block groups RatGuard flags accounted for 37% of the next quarter's rat requests (naive 'last quarter' rule: 36%; random: 10%).
- **Recurrence slice:** among recently treated block groups, the top 10% caught 27% of next-quarter requests vs 26% for the naive rule.
- **Does sanitation data add anything?** Rat history only: 37%; with sanitation, shelter and food-business signals: 37% (difference +0.1%, 95% CI -0.2% to +0.3%). Recurrence slice: +0.0% (-0.2% to +0.5%). Only claim an improvement if the interval is above zero.
- **Fixable signal:** 14% of the model's total SHAP impact comes from conditions people can fix (sanitation, shelter, food businesses).
- **Lead-lag:** within the same block group (season removed), sanitation requests 1-4 weeks earlier correlate with rat requests at 0.021 on average, vs 0.017 in the reverse direction (difference 0.004, 95% CI 0.001 to 0.007). The interval is above zero: sanitation problems tend to come first.
- **Equity:** given physical conditions, the lowest-income fifth of block groups filed 0.44x the expected rat requests vs 1.21x in the highest-income fifth. 81 block groups are flagged as possible blind spots (high expected, low reported). A low ratio can mean fewer rats OR under-reporting; field checks decide.

## Model metrics (test year)

| slice | model_name | capture_top | capture_lo | capture_hi | spearman | poisson_dev | mae | mean_next3 | rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | Naive: next 3 months = last 3 months | 0.356 | 0.327 | 0.386 | 0.831 | 5.115 | 4.063 | 7.373 | 6852 |
| all | Poisson regression | 0.369 | 0.342 | 0.399 | 0.872 | 2.420 | 3.049 | 7.373 | 6852 |
| all | Random forest | 0.369 | 0.340 | 0.399 | 0.872 | 2.494 | 3.103 | 7.373 | 6852 |
| all | XGBoost, rat history only | 0.367 | 0.341 | 0.399 | 0.869 | 2.461 | 3.053 | 7.373 | 6852 |
| all | XGBoost, full RatGuard | 0.369 | 0.342 | 0.401 | 0.873 | 2.396 | 2.998 | 7.373 | 6852 |
| recent | Naive: next 3 months = last 3 months | 0.260 | 0.231 | 0.289 | 0.785 | 5.457 | 6.106 | 11.817 | 3872 |
| recent | Poisson regression | 0.273 | 0.249 | 0.301 | 0.833 | 2.902 | 4.417 | 11.817 | 3872 |
| recent | Random forest | 0.274 | 0.247 | 0.300 | 0.831 | 3.018 | 4.491 | 11.817 | 3872 |
| recent | XGBoost, rat history only | 0.274 | 0.246 | 0.300 | 0.830 | 2.934 | 4.398 | 11.817 | 3872 |
| recent | XGBoost, full RatGuard | 0.275 | 0.249 | 0.300 | 0.834 | 2.872 | 4.336 | 11.817 | 3872 |

## Signal by driver category (share of mean |SHAP|)

| category | share |
| --- | --- |
| colony | 0.650 |
| season | 0.133 |
| food | 0.090 |
| density | 0.047 |
| shelter | 0.027 |
| containers | 0.027 |
| food_business | 0.026 |

## Top drivers

| label | category | mean_abs_shap |
| --- | --- | --- |
| Rat requests, last 12 months | colony | 0.731 |
| Months with rat requests (last 12) | colony | 0.190 |
| Rat requests, last 3 months | colony | 0.165 |
| Time of year | season | 0.129 |
| Sanitation requests, last 12 months | food | 0.129 |
| Time of year | season | 0.092 |
| Rat requests this month | colony | 0.087 |
| Population density | density | 0.074 |
| Rat-resistant can requests, last 12 months | containers | 0.043 |
| Vacant or blighted buildings | shelter | 0.033 |
| Licensed bars and restaurants | food_business | 0.033 |
| Months since last rat request | colony | 0.032 |

## Method notes (for the appendix)

- Unit: 2020 census block group x month. Target: number of 311 'Rodent Inspection and Treatment' requests in the next 3 months.
- Models: naive persistence, Poisson regression, random forest, XGBoost (Poisson objective) with rat history only, and XGBoost with all signals. The two XGBoost models differ only in their inputs (ablation).
- Time split with a purge gap so no label overlaps the next period: train through the year before validation, validate (early stopping only), test on a held-out year.
- Capture rate: each month, flag the top 10% of block groups by forecast; share of next-quarter requests that fall in them.
- 95% intervals: 200 bootstrap resamples of whole block groups (keeps each block's months together); model differences use paired resamples.
- SHAP values from XGBoost's built-in TreeSHAP (pred_contribs).
- Equity: expected requests from a Poisson gradient-boosting model using only physical conditions (vacant buildings, food businesses, density), 5-fold out-of-fold. Income is used only to audit, never to predict. Limit: where low income and poor physical conditions sit in the same places, this audit cannot fully separate them and will understate under-reporting.
- Caveat: a 311 request is not a confirmed rat. In FY17 only 46.1% of rodent requests led to a burrow treatment (DC CapSTAT, 2017).