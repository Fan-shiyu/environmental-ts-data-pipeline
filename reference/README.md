# Reference: Legacy GEE JavaScript Scripts

These are the original Google Earth Engine (GEE) JavaScript scripts from the
[environmental-time-series](https://github.com/SensingClues/environmental-time-series)
Shiny app, preserved verbatim from:
[archive/scripts/](https://github.com/SensingClues/environmental-time-series/tree/main/archive/scripts)

## Purpose

These scripts were the source pipeline that this Python repo replaces. They are
preserved here as reference and as a fallback for understanding the original
processing logic when implementing or debugging the Python equivalent.

## Files

| File | Role |
|------|------|
| `paramsGEE.js` | Configuration: AoI paths, date ranges, resolution, CRS, export folder |
| `preprocGEE.js` | Preprocessing: Sentinel-2 collection loading, SCL cloud masking, NDVI calculation |
| `getGEEdata.js` | Main script: loops over months, calls preproc functions, exports NDVI GeoTIFFs to Drive |

## Known Bug in `preprocGEE.js`

In the `sclMasker` function, three consecutive `.and(scl.neq(7))` clauses appear
where the comments indicate they should mask SCL classes 2 (Dark area), 3 (Cloud
shadow), and 7 (Unclassified):

```js
var mask = scl.neq(1) // 1 = Defective
              .and(scl.neq(7))  // 2 = Dark area      <-- should be neq(2)
              .and(scl.neq(7))  // 3 = Cloud shadow   <-- should be neq(3)
              .and(scl.neq(7))  // 7 = Unclassified
```

The result is that SCL classes 2 and 3 are **not masked** in the legacy pipeline —
only class 7 is masked (three times redundantly).

The new Python pipeline fixes this by masking all three classes explicitly:
`2` (Dark area), `3` (Cloud shadow), and `7` (Unclassified).
