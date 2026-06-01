# Data Structure Report — Shiny App Tab Analysis

> Read-only scan of `environmental-time-series/app/`. No app files were modified.
> Scope: `server.R`, `src/scenario_analysis.R`, `src/visualization.R`, `src/generate_plots.R`, `src/utilities.R`.

---

## Summary

This report documents every statistical computation, data loading pattern, and hardcoded parameter in the Shiny app. It is the reference spec for designing a pre-processing pipeline that generates Parquet summary tables and derived raster products.

### Most expensive runtime computations

| Rank | Tab | Operation | Why expensive |
|---|---|---|---|
| 1 | Scenario Explorer (all 3 subtabs) | `scenario_ndvi_data` reactive loads ALL years × ALL 7 LC classes | Every file for every year read via `terra::global()` per class |
| 2 | Burned Area Map — Fire Return Period | Per-pixel burn count across all available years | `terra::app(rast, sum)` over full raster stack; vectorized to polygons |
| 3 | NDVI Delta Map — Annual change | Load and average two full-year raster stacks | `terra::mean()` per pixel over 12 monthly layers × 2 years |
| 4 | Burned Area — Daily Activity | Load all BA rasters for selected years | `terra::values()` per file; count per Julian day |
| 5 | NDVI Time Series + LC Explorer | Load train + test rasters, mask to AoI, spatial mean | Multiple raster loads on Generate click |

### Highest priority to pre-compute

1. **Per-class per-month NDVI means** (all years × 7 LC classes) — eliminates the most expensive reactive
2. **Monthly AoI-wide NDVI means** (all years) — feeds Time Series tab
3. **Historical 95% CI ribbon** per month per AoI and per LC class — pure aggregate, stable output
4. **Fire Return Period raster** — currently re-computed when file count changes; pre-compute as static TIF
5. **Monthly burned area km²** per AoI per year — feeds Seasonal Overview and Daily Activity

### Surprising findings

- **Monthly time series ribbon = min/max** (not percentiles). Annual view ribbon = Q1–Q3. LC Explorer ribbon = 95% CI. Three different methods across three tabs.
- **Delta NDVI on the Leaflet monthly map is normalized**: `(test − train) / (test + train)`. Annual change view uses plain subtraction. These are NOT the same formula.
- **Land cover GeoJSONs have no properties at all** — the class name is encoded entirely in the filename. There is nothing to join on; masking uses polygon geometry only.
- **Only 2023 land cover** is used for all Scenario Explorer and LC Explorer work regardless of selected year.
- **Mann-Kendall trend line drawn at p < 0.1** but status badge uses p < 0.05. Different thresholds for the same test in the same tab.
- **Fire Return Period output is polygons** (via `terra::as.polygons()`), not a raster TIF.
- **Seasonal Mann-Kendall requires ≥ 60 monthly samples** (5 years) before it runs at all.
- **Pixel area for 500 m BA computed dynamically** from resolution string (`(res_m/1000)^2`), not hardcoded — but the math yields 0.25 km².

---

## 1. NDVI Time Series Tab

### Monthly view

**Historical range band (blue ribbon) = min / max of historical years — NOT percentiles:**

```r
get_monthly_historic_range <- function(train_monthly) {
  train_monthly %>%
    dplyr::group_by(Month) %>%
    dplyr::summarise(
      lower = min(NDVI, na.rm = TRUE),
      upper = max(NDVI, na.rm = TRUE),
      .groups = "drop"
    )
}
```

Baseline = all years where `year < year_in_test` AND month is in the set of months present in the test year:

```r
train_files_df <- files_df %>%
  dplyr::filter(month %in% months_in_test & year < year_in_test)
```

**Current year line:** spatial mean across all pixels for each month of the selected year (`terra::global(ndvi_rast, fun = "mean", na.rm = TRUE)`).

**Anomaly bars and color:**

```r
get_monthly_climatology <- function(train_df) {
  train_df %>%
    dplyr::group_by(Month) %>%
    dplyr::summarise(climatology = mean(NDVI, na.rm = TRUE), .groups = "drop")
}

make_anomaly_data <- function(test_df, climatology_df) {
  test_df %>%
    dplyr::left_join(climatology_df, by = "Month") %>%
    dplyr::mutate(
      anomaly   = NDVI - climatology,
      bar_color = ifelse(anomaly >= 0, "#009E73", "#D55E00")
    )
}
```

Green `#009E73` if anomaly ≥ 0; orange `#D55E00` if anomaly < 0.

**Overall Status badge:**

```r
status <- if (!is.null(s$smk_p) && !is.na(s$smk_p) && s$smk_p < 0.05 &&
              !is.null(s$sen_slope) && !is.na(s$sen_slope) && s$sen_slope < 0) {
  "Degrading"
} else if (!is.null(s$wilcox_p) && !is.na(s$wilcox_p) && s$wilcox_p < 0.05 &&
           !is.null(s$wilcox_median) && !is.na(s$wilcox_median) && s$wilcox_median < 0) {
  "Mild stress"
} else {
  "Stable"
}
```

Priority: **Degrading** (SMK p < 0.05 AND Sen slope < 0) → **Mild stress** (Wilcoxon p < 0.05 AND median anomaly < 0) → **Stable**.

**Wilcoxon test (applied to monthly anomalies of the selected year):**

```r
anom <- stats::na.omit(plot_df$anomaly)
if (length(anom) >= 3L) {
  wt <- stats::wilcox.test(anom, mu = 0)
  wilcox_p <- unname(wt$p.value)
  wilcox_median <- stats::median(anom)
}
```

Minimum: 3 anomaly values. Tests whether median anomaly differs from 0.

**Seasonal Mann-Kendall + Sen's slope:**

```r
smk_min_months <- 60L
if (smk_n_months >= smk_min_months) {
  st <- ndvi_monthly_full$YearMonth[1]
  ndvi_ts <- stats::ts(
    ndvi_monthly_full$NDVI,
    start = c(lubridate::year(st), lubridate::month(st)),
    frequency = 12
  )
  smk <- tryCatch(trend::smk.test(ndvi_ts), error = function(e) NULL)
  sen <- tryCatch(trend::sens.slope(ndvi_ts), error = function(e) NULL)
  if (!is.null(smk)) smk_p <- unname(smk$p.value)
  if (!is.null(sen)) sen_slope <- as.numeric(sen$estimates)[1]
}
```

Package: `trend`. Full series = train + test combined. Minimum: **60 monthly observations**. P-value threshold: **0.05** for badge.

### Annual view

**Annual NDVI computation:**

```r
layer_means <- terra::global(yr_rast, "mean", na.rm = TRUE)$mean
data.frame(year = yr, mean_ndvi = mean(layer_means, na.rm = TRUE), n_months = n_mo)
```

Mean of per-month spatial means. Does NOT weight by days per month.

**Incomplete year (orange circles):** year with `n_months < 12`. Excluded from all statistics.

```r
annual_df$is_complete <- annual_df$n_months == 12L
```

**Typical range band = Q1–Q3 of complete years:**

```r
ndvi_q1 <- unname(quantile(complete_df$mean_ndvi, 0.25, na.rm = TRUE))
ndvi_q3 <- unname(quantile(complete_df$mean_ndvi, 0.75, na.rm = TRUE))
```

Hover label: `"Typical range (25th-75th percentile)"`. Whisker lines = historical min/max.

**Annual Mann-Kendall + Sen's slope:**

```r
# Requires >= 5 complete years
if (n_years >= 5L) {
  mk  <- tryCatch(trend::mk.test(complete_df$mean_ndvi),    error = function(e) NULL)
  sen <- tryCatch(trend::sens.slope(complete_df$mean_ndvi), error = function(e) NULL)
}
# Trend line drawn at p < 0.1 (badge uses p < 0.05)
if (!is.na(mk_result$p) && mk_result$p < 0.1 && !is.na(mk_result$slope)) {
  # draw trend line
}
```

---

## 2. NDVI Land Cover Explorer Tab

**Loading and masking:**

```r
land_use_lc <- get_aoi_vector(aoi_files = land_cover_file, aoi_path = lulc_path, projection = "EPSG:4326")
test_ndvi_lc  <- mask(test_ndvi_msk, land_use_lc)
train_ndvi_lc <- mask(train_ndvi_msk, land_use_lc)
```

Packages: `terra` (mask/rast) and `sf` (st_read). No raster-to-vector spatial join — NDVI raster is masked to each LC polygon; pixels outside become NA.

**Per-class NDVI = spatial mean:**

```r
test_ndvi_df_lc <- get_ndvi_global_means_df(ndvi_rast = test_ndvi_lc, dates = test_files_df$dates)
```

`get_ndvi_global_means_df` calls `terra::global(ndvi_rast, fun = "mean", na.rm = TRUE)` — mean across all non-NA pixels per monthly layer.

**Historical range ribbon = 95% CI:**

```r
get_summary_ndvi_df <- function(ndvi_df = NULL) {
  ndvi_df %>%
    group_by(Year, Month) %>%
    summarize(mean_ym_ndvi = mean(NDVI)) %>%
    group_by(Month) %>%
    summarize(
      mean_val = mean(mean_ym_ndvi),
      lower_ci = mean(mean_ym_ndvi) - 1.96 * sd(mean_ym_ndvi) / sqrt(length(mean_ym_ndvi)),
      upper_ci = mean(mean_ym_ndvi) + 1.96 * sd(mean_ym_ndvi) / sqrt(length(mean_ym_ndvi))
    )
}
```

95% CI = mean ± 1.96 × SE. NOT percentiles. Lower CI is NOT clamped.

**Land cover area (ha and % of study area):**

```r
total_study_area_ha <- sum(as.numeric(sf::st_area(aoi_wgs)), na.rm = TRUE) / 10000
area_ha <- sum(as.numeric(sf::st_area(geojson_data))) / 10000
pct_label <- paste0(" (", round((area_ha / total_study_area_ha) * 100), "% of study area)")
```

`sf::st_area()` in EPSG:4326, divided by 10000 = hectares. Percentage rounded to integer.

**Land cover classes (7 canonical):** `Bare_ground`, `Built_Area`, `Crops`, `Flooded_vegetation`, `Rangeland`, `Trees`, `Water`.

**Year used:** Only **2023**, hardcoded: `land_use_src <- "S2_10m_LULC_2023"`.

---

## 3. NDVI Delta Map Tab

### Monthly view

**4 static raster maps:** Last 4 available years for selected month: `this_and_last_year <- tail(this_and_last_year, n = 4)`.

**Delta NDVI on Leaflet = normalized difference:**

```r
get_delta_ndvi_df <- function(train_ndvi_df = NULL, test_ndvi_df = NULL) {
  train_ndvi_summary <- train_ndvi_df %>%
    group_by(x, y, Month) %>%
    summarize(mean_ndvi = mean(NDVI))

  test_ndvi_summary <- test_ndvi_df %>%
    group_by(x, y, Month) %>%
    summarize(mean_ndvi = mean(NDVI))

  ndvi_comparison <- train_ndvi_summary %>%
    inner_join(test_ndvi_summary, by = c("x", "y", "Month"), suffix = c("_train", "_test"))

  delta_ndvi_df <- ndvi_comparison %>%
    mutate(delta_ndvi = (mean_ndvi_test - mean_ndvi_train) / (mean_ndvi_test + mean_ndvi_train))
}
```

Formula: `(test − train) / (test + train)`. Color domain clamped to `c(-0.25, 0.25)` via `scales::squish`. Continuous `darkred` → `darkgreen`.

### Annual change view

**Annual mean per pixel and delta:**

```r
load_year_mean <- function(yr) {
  rast <- get_ndvi_raster(yr_files$filenames, data_path, "EPSG:4326", yr_files$dates, aoi_proj)
  terra::mean(rast, na.rm = TRUE)
}
delta_rast <- rast_b - rast_a  # plain subtraction — NOT normalized
```

**Gain/loss km² computation:**

```r
pos_rast   <- terra::ifel(delta_rast > 0, 1, NA)
neg_rast   <- terra::ifel(delta_rast < 0, 1, NA)
valid_rast <- terra::ifel(!is.na(delta_rast), 1, NA)
pos_km2    <- round(sum(terra::expanse(pos_rast,   unit = "km"), na.rm = TRUE), 1)
neg_km2    <- round(sum(terra::expanse(neg_rast,   unit = "km"), na.rm = TRUE), 1)
total_km2  <- round(sum(terra::expanse(valid_rast, unit = "km"), na.rm = TRUE), 1)
```

Threshold: delta > 0 = gain, delta < 0 = loss. No minimum magnitude. Color range = 2nd–98th percentile:

```r
quants <- quantile(delta_df$delta, c(0.02, 0.98), na.rm = TRUE)
max_abs <- max(abs(quants), na.rm = TRUE)
```

---

## 4. Burned Area Explorer — Seasonal Overview

**Monthly burned area in km² (fast path, used for Plotly):**

```r
get_ba_summary_fast <- function(files_df, data_path, aoi_proj) {
  aoi_vect <- terra::vect(aoi_proj)
  results <- lapply(seq_len(nrow(files_df)), function(i) {
    r <- terra::rast(file.path(data_path, files_df$filenames[i]))
    r <- terra::project(r, "EPSG:4326")
    r <- terra::mask(r, aoi_vect)
    burned_r    <- terra::ifel(r > 0, 1L, NA)
    burned_size <- sum(terra::expanse(burned_r, unit = "km"), na.rm = TRUE)
    ...
  })
}
```

`terra::expanse()` computes geodesic cell areas — pixel area is NOT fixed here. BurnDate > 0 = burned.

**Slow path (PNG only):**

```r
burned_rast <- classify(raster_layer, matrix(c(-Inf, 0, NA), ncol = 3, byrow = TRUE))
BurnedArea_Size <- sum(expanse(burned_rast, unit = "km"), na.rm = TRUE)
Percentage_Burned <- ifelse(BurnedArea_Size <= 1, 0, (BurnedArea_Size / TotalArea_Size) * 100)
```

Percentage set to 0 if burned area ≤ 1 km².

**Historical range ribbon = 95% CI (lower clamped to 0):**

```r
get_summary_ba_df <- function(ba_df = NULL) {
  ba_df %>%
    group_by(Year, Month) %>%
    summarize(mean_ym_ba = mean(BurnedArea_Size)) %>%
    group_by(Month) %>%
    summarize(
      mean_val = mean(mean_ym_ba),
      lower_ci = mean(mean_ym_ba) - 1.96 * sd(mean_ym_ba) / sqrt(length(mean_ym_ba)),
      upper_ci = mean(mean_ym_ba) + 1.96 * sd(mean_ym_ba) / sqrt(length(mean_ym_ba))
    ) %>%
    mutate(lower_ci = if_else(lower_ci < 0, 0, lower_ci))
}
```

Historical monthly average = `mean_val`. Same 95% CI formula as NDVI. `lower_ci` clamped to 0.

---

## 5. Burned Area Explorer — Daily Activity

**EXACT R code:**

```r
get_ba_daily_activity <- function(files_df, data_path, year_val, pixel_area_km2 = 0.25) {
  year_files <- files_df %>% dplyr::filter(year == year_val)
  if (nrow(year_files) == 0) return(NULL)

  results <- lapply(seq_len(nrow(year_files)), function(i) {
    r    <- terra::rast(file.path(data_path, year_files$filenames[i]))
    vals <- as.numeric(terra::values(r, na.rm = TRUE))
    vals <- vals[vals > 0]
    if (length(vals) == 0) return(NULL)
    dates       <- as.Date(vals - 1, origin = paste0(year_val, "-01-01"))
    date_counts <- table(dates)
    data.frame(
      date = as.Date(names(date_counts)),
      km2  = as.numeric(date_counts) * pixel_area_km2,
      year = as.character(year_val),
      stringsAsFactors = FALSE
    )
  })
  results <- Filter(Negate(is.null), results)
  if (length(results) == 0) return(NULL)
  dplyr::bind_rows(results)
}
```

Julian day: `as.Date(vals - 1, origin = paste0(year_val, "-01-01"))`. Values ≤ 0 excluded.

**Pixel area:** computed at runtime: `pixel_area_km2 <- (res_m / 1000)^2`. For 500 m: 0.25 km². Default arg is also 0.25.

**Fire season slider:** `files_df` filtered by `month %in% season_months` before call.

**Years shown:** user-selectable; default = last 3 available years (`head(as.character(available_years), 3)`).

**Plotly data structure — one row per day per year where burning occurred:**

| Column | Type | Description |
|---|---|---|
| `date` | Date | Calendar date derived from Julian day |
| `km2` | numeric | Burned area km² that day |
| `year` | character | Year label for color grouping |

---

## 6. Burned Area Map Explorer — Fire Return Period

**EXACT algorithm:**

```r
yearly_burned <- lapply(years_available, function(yr) {
  yr_files <- ba_files[grepl(paste0("^", yr, "-"), basename(ba_files))]
  if (length(yr_files) == 0) return(NULL)
  yr_max <- terra::app(terra::rast(yr_files), fun = max, na.rm = TRUE)
  terra::ifel(yr_max > 0, 1, 0)
})
yearly_burned <- Filter(Negate(is.null), yearly_burned)

burn_count    <- terra::app(terra::rast(yearly_burned), fun = sum, na.rm = TRUE)
return_period <- terra::ifel(burn_count > 0, n_years / burn_count, NA)
return_period <- terra::project(return_period, "EPSG:4326")

polys_rp <- terra::as.polygons(return_period) |> sf::st_as_sf()
```

Formula: `FRP = n_years / burn_count`. Example: burned in 3 of 20 years → FRP = 6.7 years.

Computed from **all available years**. Output = **sf polygon object** (NOT raster TIF).

Cached as RDS: `www/.cache/{country}_{resolution}_frp.rds`, invalidated by file count change.

---

## 7. Scenario Explorer — Land Cover Productivity

**Annual mean NDVI per LC class:**

```r
.compute_productivity_stats <- function(df, yr) {
  d_yr <- df[df$year == as.integer(yr), ]
  yr_stats <- dplyr::summarise(
    dplyr::group_by(d_yr, land_cover),
    annual_mean = mean(mean_ndvi, na.rm = TRUE),
    .groups = "drop"
  )
```

`mean_ndvi` = per-month spatial mean per class. `annual_mean` = mean of those monthly values.

**Historical min/max and CV:**

```r
annual_by_year <- dplyr::summarise(
  dplyr::group_by(df, land_cover, year),
  yr_mean = mean(mean_ndvi, na.rm = TRUE),
  .groups = "drop"
)
inter_stats <- dplyr::summarise(
  dplyr::group_by(annual_by_year, land_cover),
  hist_min      = min(yr_mean, na.rm = TRUE),
  hist_max      = max(yr_mean, na.rm = TRUE),
  hist_sd       = stats::sd(yr_mean, na.rm = TRUE),
  hist_mean_all = mean(yr_mean, na.rm = TRUE),
  .groups = "drop"
)
inter_stats <- dplyr::mutate(inter_stats,
  cv = ifelse(hist_mean_all != 0, hist_sd / abs(hist_mean_all), NA_real_)
)
```

Min/max/CV computed across ALL available years.

**Change column:**

```r
change_vals <- round(stats_df$annual_mean - comp_stats$annual_mean, 3)
```

Selected year annual mean minus comparison year annual mean. Rounded to 3 decimal places.

**CV interpretation thresholds:**

| Class | Threshold | Label |
|---|---|---|
| Trees | CV < 0.05 | "Stable & highly productive" |
| Rangeland | CV < 0.05 | "Stable & productive" |
| Crops | CV < 0.08 | "Consistent productivity" |

---

## 8. Scenario Explorer — Anomaly Resilience

**EXACT R code for anomaly detection and recovery:**

```r
hist_stats <- dplyr::summarise(
  dplyr::group_by(all_df, land_cover, month),
  hist_mean = mean(mean_ndvi, na.rm = TRUE),
  hist_sd   = stats::sd(mean_ndvi, na.rm = TRUE),
  .groups   = "drop"
)
hist_stats$hist_sd[is.na(hist_stats$hist_sd)] <- 0

anomaly_df <- all_df[all_df$year == as.integer(anomaly_year), ]
merged <- dplyr::left_join(anomaly_df, hist_stats, by = c("land_cover", "month"))
merged <- dplyr::mutate(merged, anomaly = mean_ndvi - hist_mean)

recovery_results <- lapply(lc_classes, function(lc) {
  df_lc <- dplyr::arrange(merged[merged$land_cover == lc, ], month)

  worst_row         <- df_lc[which.min(df_lc$anomaly), ]
  max_deficit       <- worst_row$anomaly
  deficit_month_int <- as.integer(worst_row$month)

  subsequent <- df_lc[df_lc$month > deficit_month_int, ]
  recovery_months <- NA_integer_
  for (i in seq_len(nrow(subsequent))) {
    row_i <- subsequent[i, ]
    if (abs(row_i$mean_ndvi - row_i$hist_mean) <= row_i$hist_sd) {
      recovery_months <- as.integer(row_i$month - deficit_month_int)
      break
    }
  }
  ...
})
```

Key parameters:

| Parameter | Value |
|---|---|
| Anomaly | `mean_ndvi − hist_mean` (vs ALL years, not just prior years) |
| Deficit month | Month with `min(anomaly)` in selected year |
| Recovery threshold | `abs(NDVI − hist_mean) ≤ hist_sd` (within 1 SD) |
| Months-to-recovery | `month_recovered − deficit_month` |
| No recovery detected | `NA_integer_`; shown as "—" |

**Resilience rank:**

```r
max_rec <- if (any(!is.na(recovery_df$recovery_months)))
  max(recovery_df$recovery_months, na.rm = TRUE) * 2L else 12L
recovery_df$resilience_score <- abs(recovery_df$max_deficit) *
  ifelse(is.na(recovery_df$recovery_months), max_rec, recovery_df$recovery_months)
recovery_df$resilience_rank  <- rank(recovery_df$resilience_score,
                                     ties.method = "min", na.last = TRUE)
```

Score = `|deficit| × recovery_months`. No recovery penalty = `max_observed × 2` (or 12 if no class recovered). Rank 1 = most resilient.

**Anomaly severity thresholds:**

```r
.anomaly_severity <- function(deficit) {
  if (is.na(deficit) || deficit >= 0)  return("Minimal")
  if (deficit < -0.15)                 return("Severe")
  if (deficit < -0.10)                 return("Moderate")
  if (deficit < -0.05)                 return("Mild")
  return("Minimal")
}
```

---

## 9. Scenario Explorer — Agricultural Monitoring

**Green-up detection:**

```r
baseline_ndvi <- ndvi[which(months == profile$green_up_baseline_month)[1L]]
gu_win <- .months_in_window(profile$green_up_window[1L], profile$green_up_window[2L])
gu_idx <- which(months %in% gu_win)

green_up_rise <- max(ndvi[gu_idx], na.rm = TRUE) - baseline_ndvi
if (green_up_rise >= profile$green_up_conf_lo_min) {
  thr_val <- baseline_ndvi + profile$green_up_threshold
  crossed <- gu_idx[ndvi[gu_idx] > thr_val]
  green_up_month <- if (length(crossed) > 0L) months[crossed[1L]]
                    else months[gu_idx[which.max(ndvi[gu_idx])]]
}
```

First month in green-up window where NDVI > `baseline + threshold`. If no crossing but rise ≥ `conf_lo_min`, uses month of max NDVI in window.

**Peak detection:**

```r
pk_win <- .months_in_window(profile$peak_window[1L], profile$peak_window[2L])
pk_idx <- which(months %in% pk_win)
best   <- pk_idx[which.max(ndvi[pk_idx])]
peak_month    <- months[best]
peak_ndvi_val <- ndvi[best]
```

Peak confidence:

```r
.score_peak <- function(peak_ndvi, hist_avg, profile) {
  dev <- peak_ndvi - hist_avg
  if (dev >  profile$peak_conf_delta) return("High")
  if (dev < -profile$peak_conf_delta) return("Low")
  return("Medium")
}
```

**Senescence detection:**

```r
for (k in seq_along(sen_idx)) {
  i <- sen_idx[k]
  if (!is.na(ndvi[i]) && ndvi[i] < profile$senescence_threshold) {
    senescence_month <- months[i]
    break
  }
}
```

First month in senescence window where NDVI < `senescence_threshold`.

**Season length:**

```r
season_length_months <- as.integer((senescence_month + 12L - green_up_month) %% 12L)
```

Modular arithmetic handles year-boundary wrap.

**Phenology profiles — Zambia (Maize/Crops):**

| Parameter | Value |
|---|---|
| `green_up_window` | c(11, 12) |
| `green_up_baseline_month` | 10 |
| `green_up_threshold` | 0.08 NDVI above baseline |
| `green_up_conf_lo_min` | 0.03 |
| `peak_window` | c(1, 3) |
| `peak_conf_delta` | 0.05 |
| `senescence_window` | c(4, 6) |
| `senescence_threshold` | 0.30 NDVI |

**Phenology profiles — Generic Rangeland:**

| Parameter | Value |
|---|---|
| `green_up_window` | c(10, 12) |
| `green_up_baseline_month` | 9 |
| `green_up_threshold` | 0.05 |
| `green_up_conf_lo_min` | 0.02 |
| `peak_window` | c(1, 3) |
| `peak_conf_delta` | 0.04 |
| `senescence_window` | c(4, 7) |
| `senescence_threshold` | 0.20 NDVI |

---

## 10. Land Cover GeoJSON Details

**Data root:** `app/www/data/LandUse/{country}/S2_10m_LULC_2023/`

**Zambia_Mponda — 7 files:**

| Filename | Geometry | Features | Properties |
|---|---|---|---|
| `Zambia_Mponda_Bare_ground_2023.geojson` | MultiPolygon | 1 | None (empty `{}`) |
| `Zambia_Mponda_Built_Area_2023.geojson` | MultiPolygon | 1 | None |
| `Zambia_Mponda_Crops_2023.geojson` | MultiPolygon | 1 | None |
| `Zambia_Mponda_Flooded_vegetation_2023.geojson` | MultiPolygon | 1 | None |
| `Zambia_Mponda_Rangeland_2023.geojson` | MultiPolygon | 1 | None |
| `Zambia_Mponda_Trees_2023.geojson` | MultiPolygon | 1 | None |
| `Zambia_Mponda_Water_2023.geojson` | MultiPolygon | 1 | None |

**CRITICAL:** All GeoJSONs have no attribute properties. Class name encoded in filename only. R code globs `*_{class}_2023.geojson`.

**AoI GeoJSON:** `AoI_Zambia_Mponda_By_Life_Connected.geojson` — Polygon, 1 feature, property `{'NAME': 'Mponda Seniorities'}`, CRS: EPSG:4326.

**Land cover year:** Only 2023. No multi-year LULC. All scenario/explorer work uses 2023 LC with any NDVI year.

---

## 11. Reactive Dependencies and Loading Performance

**At startup:** Libraries loaded, all `src/` files sourced, `plan(multisession)` set. AoI GeoJSON read immediately on country input change (`ignoreInit = FALSE`).

**Reactively (Generate button):** All raster I/O inside `observeEvent(input$generate_*)`.

**Scenario Explorer shared reactive — most expensive:**

```r
scenario_ndvi_data <- reactive({
  req(input$country, input$resolution)
  df <- .load_all_years(avail_years, country_name, resolution, data_dir, lulc_dir)
  df
})
```

Loads ALL years × ALL 7 LC classes in one call. Shared across all 3 Scenario subtabs. Fires on country/resolution change.

**Caching:**

| Cache file | Invalidation |
|---|---|
| `www/.cache/ba_ts_{country}_{resolution}_train.rds` | File count change |
| `www/.cache/{country}_{resolution}_frp.rds` | File count change |

No `memoise` or Shiny built-in cache.

**Performance signals:**
- `withProgress(message = "Analysing burn patterns across available years...")` wraps FRP computation
- Comment: `# Fast path: read one file at a time, skip pixel-level raster->dataframe conversion`
- Separate `get_ba_summary_fast` vs `get_ba_df` (labeled "slow path, PNG only")

---

## 12. Complete List of Statistical Parameters

| Parameter | Value | Location | Context |
|---|---|---|---|
| 95% CI multiplier | `1.96` | `get_summary_ndvi_df`, `get_summary_ba_df` | Ribbon: LC Explorer, BA Seasonal Overview |
| Monthly NDVI ribbon method | min / max (NOT CI) | `get_monthly_historic_range` | Time Series monthly view only |
| Annual view Q1 | `0.25` | `plot_ndvi_annual` | IQR band lower bound |
| Annual view Q3 | `0.75` | `plot_ndvi_annual` | IQR band upper bound |
| BA lower CI floor | `0` | `get_summary_ba_df` | Clamped — BA cannot be negative |
| BA % threshold | `≤ 1 km²` → 0% | `get_ba_df` (slow path) | |
| Significance threshold | `0.05` | Multiple | Status badges, insight labels |
| Annual trend line p-value | `0.1` | `plot_ndvi_annual` | Line drawn at p < 0.1, NOT 0.05 |
| Min months for SMK | `60` | `compute_ndvi_explorer_stats` | ≥ 60 monthly points required |
| Min years for annual MK | `5` | `compute_ndvi_annual_stats` | ≥ 5 complete years required |
| Min anomaly samples (Wilcoxon) | `3` | `compute_ndvi_explorer_stats` | `length(anom) >= 3L` |
| Incomplete year threshold | `n_months < 12` | `generate_timeseries` | Excluded from all stats |
| Monthly delta NDVI color range | `c(-0.25, 0.25)` | `plot_delta_ndvi_streetview` | Clamped via squish |
| Annual delta color range | 2nd–98th pctl | `plot_annual_ndvi_leaflet` | `quantile(c(0.02, 0.98))` |
| Gain threshold | delta > 0 | Annual change view | No minimum magnitude |
| Loss threshold | delta < 0 | Annual change view | No minimum magnitude |
| Default pixel area 500 m | `0.25 km²` | `get_ba_daily_activity` | `(500/1000)^2` |
| FRP formula | `n_years / burn_count` | `build_ba_frp_leaflet` | |
| Anomaly Severe | `< -0.15` NDVI | `.anomaly_severity` | |
| Anomaly Moderate | `< -0.10` NDVI | `.anomaly_severity` | |
| Anomaly Mild | `< -0.05` NDVI | `.anomaly_severity` | |
| Recovery threshold | `abs(NDVI − mean) ≤ 1 SD` | `plot_anomaly_resilience` | |
| Recovery penalty (no recovery) | `max_rec × 2` or `12` | `plot_anomaly_resilience` | |
| CV threshold Trees | `0.05` | `.lc_interpretation` | |
| CV threshold Rangeland | `0.05` | `.lc_interpretation` | |
| CV threshold Crops | `0.08` | `.lc_interpretation` | |
| Zambia green-up threshold | `+0.08` NDVI above Oct baseline | `.phenology_profiles` | |
| Zambia senescence threshold | `0.30` NDVI | `.phenology_profiles` | |
| Zambia peak conf delta | `0.05` NDVI | `.phenology_profiles` | |
| Spain wheat green-up | `+0.10` NDVI | `.phenology_profiles` | |
| Spain senescence | `0.35` NDVI | `.phenology_profiles` | |
| Productivity bar Y range | `c(0, 0.8)` | `plot_productivity_comparison` | Fixed axis |
| Insight trend magnitude | `0.005 NDVI/yr` | `plot_agricultural_monitoring` | Linear regression slope |
| Insight complete year | `≥ 6 months data` | `plot_agricultural_monitoring` | |
| LC area % rounding | Integer | LC Explorer | `round(pct)` |
