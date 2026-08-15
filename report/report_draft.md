# CSE303 Term Project — Laptop Price Regression

## Problem Statement

Predict laptop selling prices using regression models trained on two data
sources: (A) a Kaggle dataset of ~1300 laptop listings with prices in Euros,
and (B) ~218 self-scraped listings from startech.com.bd (Bangladesh) with
prices in BDT. The core challenge is transforming messy, human-readable
specification strings into structured numerical features that capture the
price-determining characteristics of a laptop.

---

## Data Description

### Track A — Kaggle "Laptop Price" (1303 rows, 13 columns)

| Column | Type | Missing | Description |
|---|---|---|---|
| `laptop_ID` | int | 0% | Row identifier (dropped) |
| `Company` | str | 0% | Brand name (19 unique) |
| `Product` | str | 0% | Product name (618 unique, dropped) |
| `TypeName` | str | 0% | Device category (6: Ultrabook, Notebook, Gaming, etc.) |
| `Inches` | float | 0% | Screen diagonal (inches) |
| `ScreenResolution` | str | 0% | Compound: "IPS Panel Retina Display 2560x1600" |
| `Cpu` | str | 0% | Compound: "Intel Core i5 2.3GHz" |
| `Ram` | str | 0% | "8GB" → 8 |
| `Memory` | str | 0% | Compound: "256GB SSD + 1TB HDD" |
| `Gpu` | str | 0% | "NVIDIA GeForce GTX 1050" |
| `OpSys` | str | 0% | Operating System (9 unique) |
| `Weight` | str | 0% | "1.37kg" → 1.37 |
| `Price_euros` | float | 0% | Target variable (174 – 6,099 €) |

Zero missing values across all columns. The dataset arrived pre-cleaned in
that sense, with the messy work being string-to-numeric parsing.

### Track B — Startech.com.bd Scraped (218 rows, 10 columns)

| Column | Type | Missing | Description |
|---|---|---|---|
| `Brand` | str | 0% | Extracted from product name |
| `Model` | str | 0% | Full product name |
| `Processor` | str | 0% | e.g., "Intel Core i7 13700H 2.4GHz" |
| `RAM` | str | 1.4% | e.g., "16GB LPDDR5x" |
| `Storage` | str | 0.5% | e.g., "512GB PCIe 4.0 SSD" |
| `Display` | str | 11.5% | e.g., "13.8\" (1920x1280) PixelSense, IPS" |
| `GPU` | str | 85.3% | Often missing on category pages |
| `Price(BDT)` | str | 0% | Raw: "145,000৳"; 49 rows "Out Of Stock" |
| `Availability` | str | 0% | "In Stock" / "Out of Stock" |
| `Product_URL` | str | 0% | Unique link (used for dedup) |

The raw data was saved as unmodified HTML page files and a raw CSV before
any transformations. After dropping 49 rows with non-numeric price (out of
stock), 169 rows remain for modeling.

---

## Cleaning Decisions

### Compound String Parsing (Both Tracks)

- **Ram**: `"8GB"`, `"16GB LPDDR5x"` → extracted first `\d+` before "GB" → `Ram_GB` (int)
- **Weight** (Track A): `"1.37kg"` → stripped "kg" → `Weight_kg` (float)
- **Memory/Storage**: `"256GB SSD + 1TB HDD"` → `SSD_GB=256`, `HDD_GB=1024` via regex on segments split by "+". Flash Storage treated as SSD, Hybrid as HDD.
- **ScreenResolution/Display**: Parsed `Is_IPS`, `Is_Touchscreen` (booleans), `Screen_Width`, `Screen_Height` (ints). Track A also extracted `Res_Class` (HD/FHD/QHD/4K).
- **Cpu/Processor**: Extracted `CPU_Brand` (Intel/AMD/Apple/Qualcomm/Other) and `CPU_GHz` (float)
- **Gpu**: Extracted `GPU_Brand` (Nvidia/AMD/Intel/Apple/Other/Unknown)
- **Price(BDT)**: Stripped "৳" and commas → `Price_BDT` (int). Rows with "Out Of Stock" as price → dropped.

### Missing Value Strategy (Track B)

| Field | Missing | Strategy |
|---|---|---|
| GPU | 85.3% | Filled as "Unknown" — GPU rarely visible on category pages |
| Display | 11.5% | Screen_Size/Width/Height/PPI → median imputed |
| RAM_GB | 1.4% | Median imputed (16 GB) |
| CPU_GHz | <5% | Median imputed |
| Price_BDT | 22.5% | 49 out-of-stock rows dropped (cannot predict without target) |

No silent `dropna()` was used — every imputation is documented.

### Outlier Treatment

**Track A (Price_euros)**: 29 IQR outliers (2.2%), 12 z>3 outliers (0.9%).
**Track B (Price_BDT)**: 16 IQR outliers, 4 z>3 outliers.

**Decision for both tracks**: All outliers **retained**. They represent genuine
premium laptops (Razer Blade Pro, Alienware, ThinkPad P-series, RTX 5090
configs), not measurement errors. Log-transformation in feature engineering
handles the heavy right tail.

### Deduplication (Track B)

Listings deduplicated by `Product_URL`: 237 scraped → 218 unique (19
cross-listed between laptop and ultrabook categories).

---

## EDA Insights

### Track A

1. **Price distribution** is strongly right-skewed (skew > 1.5). Most laptops
   cluster below 2,000 € with a long tail of high-end machines above 3,000 €.
2. **RAM vs Price**: Clear positive relationship. 8 GB laptops average ~800 €,
   16 GB ~1,400 €, 32 GB ~2,500 €.
3. **Screen resolution**: 4K and QHD displays command a ~40% premium over
   equivalent HD laptops. IPS panels add ~15-20%.
4. **CPU brand**: Intel dominates (85%), but Apple M-series occupies the highest
   price bracket. AMD sits at mid-range.
5. **Weight** correlates with price: heavier laptops (>2.5 kg) are gaming
   workstations; ultra-light (<1.2 kg) are premium ultrabooks.
6. **Correlation heatmap**: No severe multicollinearity (max |r| = 0.85
   between Screen_Width and Screen_Height, expected by construction).

### Track B

1. **Price range**: 27,500 – 660,000 BDT. Wide spread driven by gaming vs
   ultrabook segments.
2. **RAM** is the strongest predictor: 32 GB configs cost 2-3× 8 GB ones.
3. **Intel** dominates CPUs (83%). AMD Ryzen and Qualcomm Snapdragon appear in
   premium/ARM-based segments.
4. **GPU**: 85% missing from category pages. When present, Nvidia associates
   with high price.
5. **PPI**: Retina-level displays (>200 PPI) show clear pricing premium.
6. **Screen size bimodality**: 13-14" ultrabooks vs 15-16" gaming machines
   cluster at different price points.
7. **Brand segmentation**: Razer and Apple at the top; HP/Dell span budget to
   mid-range; local brands (Walton) at entry level.

---

## Feature Engineering

### Target Transformation

Both tracks: **log-transformed Price** (`np.log1p`). Skewness decreased from
~1.5 to ~0.1 (Track A) and ~1.8 to ~0.3 (Track B). Before/after histograms
included in notebooks.

### Non-Trivial Features

- **PPI** (Pixels Per Inch): `sqrt(width² + height²) / screen_size`. Captures
  display sharpness/resolution quality.
- **Total_Storage_GB**: `SSD_GB + HDD_GB`. Total storage capacity regardless of
  type.
- **SSD_GB** and **HDD_GB**: Separate SSD and HDD capacities, since SSD storage
  is typically more valuable.
- **Is_IPS**, **Is_Touchscreen**, **Res_Class**: Screen quality flags.

### Categorical Encoding

**One-hot encoding** chosen over target encoding for both tracks. Rationale:
- Low-to-moderate cardinality (2–12 levels)
- One-hot avoids target leakage and cross-validation complexity
- Track B's small sample (N=169) makes target encoding risky
- `drop_first=True` to avoid dummy variable trap

Encoded: `TypeName`, `OpSys`, `CPU_Brand`, `GPU_Brand`, `Res_Class`,
`Is_IPS`, `Is_Touchscreen` (Track A); `Brand`, `CPU_Brand`, `GPU_Brand`,
`Availability` (Track B).

---

## Modeling Approach

### Models

| Model | Description |
|---|---|
| **OLS** | Baseline linear regression |
| **Ridge** | L2 regularization, 5-fold CV over α ∈ {0.01, 0.1, 1, 10, 50, 100, 200} |
| **Lasso** | L1 regularization, same CV grid |
| **Random Forest** | Stretch model: 200 trees, max_depth=15 (A) or 10 (B) |

### Train/Test Split

80/20 stratified split, random_state=42. Features standardized via
`StandardScaler` (fit on train only). All models trained on log-transformed
price, evaluated on held-out test set only.

---

## Results

### Track A (Price in Euros)

| Model | RMSE(log) | R²(log) | RMSE(€) | MAE(€) |
|---|---|---|---|---|
| OLS | 0.274 | 0.788 | 340 | 238 |
| Ridge (α=1) | 0.273 | 0.789 | 331 | 234 |
| Lasso (α=0.01) | 0.273 | 0.789 | 335 | 234 |
| **Random Forest** | **0.218** | **0.866** | **316** | **193** |

**Best model**: Random Forest (R²=0.87 on test set)

### Track B (Price in BDT)

| Model | RMSE(log) | R²(log) | RMSE(BDT) | MAE(BDT) |
|---|---|---|---|---|
| **OLS** | **0.217** | **0.831** | **54,014** | **27,509** |
| Ridge (α=1) | 0.222 | 0.823 | 53,573 | 27,236 |
| Lasso (α=1) | 0.232 | 0.806 | 65,527 | 29,184 |
| Random Forest | 0.212 | 0.838 | 59,230 | 25,631 |

**Best model**: OLS (R²=0.83 on test set); Ridge slightly better on raw RMSE.

---

## Limitations & Failure Analysis

### Where the models fail

1. **Extreme high-end laptops** (Track A: >4,000€; Track B: >500,000 BDT)
   are systematically underpredicted. These represent <3% of training data,
   so models regress toward the mean. More data at this tier would help.

2. **Conflicting spec signals**: A laptop with a 4K display but low-end CPU
   and no dedicated GPU confuses the model — display says "premium" but
   internals say "budget." These genuine edge cases are inherently difficult.

3. **Brand premium**: `Company` was dropped from Track A (19 levels → OLS
   overfitting). Brands like Razer and Apple carry intangible premiums not
   captured by specs. Target encoding with cross-validation could recover
   some signal if needed.

4. **GPU granularity**: Only GPU *brand* is captured, not model *tier*
   (RTX 3050 vs RTX 4090). This discards massive price signal — a GPU tier
   index would substantially improve predictions.

5. **Track B data scarcity**: 169 usable rows (after removing out-of-stock)
   yields a test set of only ~34 observations. Metrics have high variance.
   Minimum target of 300 listings was not reached (218 unique, 169 with
   valid price).

6. **Missing data (Track B)**: 85% of GPU info is missing from category
   pages. Scraping individual product detail pages would recover this.

7. **Bangladesh market**: Track B reflects a local market with different
   brand distribution and pricing dynamics from Track A's global/EU data.

8. **Log-transform asymmetry**: Back-transform (`expm1`) means overpredictions
   cost more in RMSE than underpredictions at high price points.

### Recommended improvements

- Scrape individual product detail pages for complete specs (GPU tier,
  refresh rate, build material)
- Collect >500 rows minimum for Track B
- Add GPU tier feature (parsing model numbers like "RTX 3060" → tier score)
- Explore polynomial features for RAM-price interaction
- Use target encoding for high-cardinality brands with cross-validated
  smoothing

---

*Report generated as part of the CSE303 Term Project pipeline. Both notebooks
run end-to-end with zero manual intervention (random_state=42).*
