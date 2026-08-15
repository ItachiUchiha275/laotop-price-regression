# CSE303 Term Project — Laptop Price Regression
## Project Report

---

## 1. Introduction

This project solves a supervised regression problem: predicting laptop selling prices from hardware specification data. Because laptop markets differ across regions, the work is organized as two independent tracks sharing one end-to-end pipeline. **Track A** uses a clean Kaggle dataset of 1,303 laptop listings priced in Euros (a global/EU market). **Track B** uses 472 listings hand-scraped from two Bangladeshi e-commerce sites — startech.com.bd and ryans.com — with prices in Bangladeshi Taka (BDT). The central challenge in both tracks is identical: raw specification strings such as `"IPS Panel Retina Display 2560x1600"` or `"512GB PCIe 4.0 SSD"` must be transformed into structured, numeric features before any model can learn from them.

A single reproducible pipeline implements compound-string parsing, documented missing-value handling, exploratory data analysis, feature engineering, modeling, and rigorous held-out evaluation. Four regression models — ordinary least squares (OLS), Ridge, Lasso, and Random Forest — were trained on a log-transformed price target using an 80/20 train/test split (`random_state=42`) and scored with RMSE, MAE, and R². Random Forest generalized best on both tracks (Track A R² = 0.877, Track B R² = 0.816). Track B is additionally evaluated as a classifier by discretizing price into four bands, reporting a confusion matrix with accuracy, precision, recall, and F1-score. The entire workflow runs with zero manual intervention via a one-click launcher.

---

## 2. Data Preprocessing

### 2.1 Compound string parsing (both tracks)

Most specification fields arrive as dense, human-readable strings that must be parsed with regular expressions:

| Raw field | Example | Parsed features |
|---|---|---|
| `Ram` | `"16GB LPDDR5x"` | `Ram_GB` (int) |
| `Weight` | `"1.37kg"` | `Weight_kg` (float) |
| `Memory` / `Storage` | `"256GB SSD + 1TB HDD"` | `SSD_GB`, `HDD_GB`, `Total_Storage_GB` |
| `ScreenResolution` / `Display` | `"IPS Panel ... 2560x1600"` | `Is_IPS`, `Is_Touchscreen`, `Screen_Width`, `Screen_Height`, `PPI` |
| `Cpu` / `Processor` | `"Intel Core i5 2.3GHz"` | `CPU_Brand`, `CPU_GHz` |
| `Gpu` | `"NVIDIA GeForce RTX 3050"` | `GPU_Brand` |
| `Price(BDT)` | `"145,000৳"`, `"Out Of Stock"` | `Price_BDT` (numeric; out-of-stock dropped) |

### 2.2 Missing values, deduplication, and outliers

**Track A** arrived pre-cleaned (0% missing in all 13 columns). The messy work was purely string parsing. `laptop_ID` and the high-cardinality `Product` column were dropped as uninformative.

**Track B** was noisy and incomplete. Of 472 scraped rows, 49 had no numeric price (out-of-stock listings) and were removed because a regression target cannot be constructed without them, leaving 423 rows. Listings were deduplicated by `Product_URL`. Remaining missing values were never dropped silently — every imputation is documented:

| Field | Missing | Strategy |
|---|---|---|
| GPU | 27 rows | Labeled `"Unknown"` (GPU rarely listed on category pages) |
| Display / Screen_Size / PPI | 25–52 rows | Median imputed (15.1 in) |
| RAM_GB | 3 rows | Median imputed (16 GB) |
| CPU_GHz | few rows | Median imputed |

Outliers were analyzed on both tracks (IQR and z-score methods) and **deliberately retained** — they are genuine premium machines (RTX 5090 gaming rigs, Razer/Apple flagships), not data errors. The heavy right tail is instead handled by log-transforming the target (Section 4.1).

### 2.3 Normalization and encoding

- **Standardization**: all numeric features scaled with `StandardScaler`, fitted on the **training split only** to prevent data leakage.
- **Encoding**: categorical features one-hot encoded with `drop_first=True` (avoids the dummy-variable trap). Encoding rationale in Section 4.4.
- **Dimensionality reduction**: no explicit PCA/truncated-SVD was applied; instead high-cardinality, low-value text columns (`Product`, `Model`) were dropped, Lasso provided implicit feature selection, and `drop_first` reduced the dummy columns.

---

## 3. Dataset Characteristics and Exploratory Data Analysis

### 3.1 Track A — Kaggle (1,303 × 13)

A clean dataset of 1,303 laptops with 13 columns and zero missing values. Target `Price_euros` ranges from **174 € to 6,099 €**. After cleaning and parsing the dataset grew to 20 columns (34 model features after encoding).

**Key findings:**

1. **Price is strongly right-skewed** (skew 1.52) — most laptops sit below 2,000 € with a long tail of premium machines above 3,000 €.
2. **RAM is the strongest single predictor**: 8 GB laptops average ~800 €, 16 GB ~1,400 €, 32 GB ~2,500 €.
3. **Display quality commands a premium**: 4K/QHD screens add ~40% over HD; IPS panels add ~15–20%.
4. **CPU brand matters**: Intel dominates (85%), but Apple's M-series occupies the highest price tier.
5. **Correlation structure**: no severe multicollinearity; the only strong pair is `Screen_Width`–`Screen_Height`, correlated by construction.

### 3.2 Track B — Scraped Bangladesh market (472 × 11 → 423 × 15)

Track B contains 472 raw listings (11 columns). After removing out-of-stock rows, 423 remain (15 columns, 32 model features). Target `Price_BDT` ranges from **27,500 to 660,000 BDT** (skew 2.14).

**Key findings:**

1. **RAM dominates price**: 32 GB configs cost 2–3× their 8 GB equivalents.
2. **Intel dominates CPUs** (83%); AMD Ryzen and Qualcomm Snapdragon appear in premium/ARM segments.
3. **GPU is informative when present**: Nvidia-branded machines sit at the top of the price ladder; GPU brand is missing for only ~6% of rows (imputed as `"Unknown"`).
4. **PPI captures display quality**: retina-level displays (>200 PPI) show a clear premium.
5. **Bimodal screen size**: 13–14" ultrabooks and 15–16" gaming machines cluster at distinct price points.
6. **Brand segmentation**: Razer and Apple at the top; HP/Dell span budget to mid-range; local brands (e.g. Walton) at entry level.

### 3.3 Visual exploration

| Track A | Track B |
|---|---|
| ![Track A price distribution](../data/track_a/cleaned/price_distribution.png) | ![Track B price distribution](../data/track_b/cleaned/price_distribution.png) |
| ![Track A correlation heatmap](../data/track_a/cleaned/correlation_heatmap.png) | ![Track B correlation heatmap](../data/track_b/cleaned/correlation_heatmap.png) |
| ![Track A category boxplots](../data/track_a/cleaned/category_boxplots.png) | ![Track B category boxplots](../data/track_b/cleaned/category_boxplots.png) |

Additional plots — numeric distributions, scatter matrices with price trend lines, and log-transform before/after histograms — are stored in `data/track_a/cleaned/` and `data/track_b/cleaned/`.

---

## 4. Feature Engineering

### 4.1 Log-transformed target

Both targets are log-transformed (`log1p`) before modeling. This compresses the heavy right tail, making residuals more homoscedastic and the relationship closer to linear. Skewness dropped from **1.52 → −0.17** (Track A) and **2.14 → 0.67** (Track B).

### 4.2 Derived hardware features

- **PPI** (pixels per inch): `√(width² + height²) / screen_size` — a single scalar capturing display sharpness.
- **Total_Storage_GB = SSD_GB + HDD_GB**, plus **separate SSD/HDD capacities** (SSD storage is more valuable per GB).
- **Is_IPS**, **Is_Touchscreen**, **Res_Class** (HD/FHD/QHD/4K): display-quality flags.
- **Spec Power Score** (Track B): `log1p( (RAM/8) · (Storage/256) · (CPU/2) )` — a synergy feature capturing that a machine with 32 GB + 1 TB + fast CPU is worth *non-linearly* more than the sum of its parts.
- **GPU Performance Index** (Track A): GPU model numbers mapped to a 0–100 performance score, capturing tier (e.g., GTX 1050 vs RTX 4090) rather than brand alone.

### 4.3 Categorical encoding rationale

One-hot encoding was chosen over target encoding because: (a) cardinality is low-to-moderate (2–12 levels), (b) one-hot avoids target leakage and CV complications, and (c) Track B's modest sample size makes target encoding risky. `drop_first=True` avoids the dummy-variable trap.

---

## 5. Regression Model and Performance Evaluation

### 5.1 Setup

All models were trained on the log-transformed target with an 80/20 split (`random_state=42`), features standardized on the train split only, and evaluated on the held-out test set. Raw-scale errors were recovered with `expm1` for interpretable €/৳ numbers.

| Model | Description |
|---|---|
| **OLS** | Baseline linear regression |
| **Ridge** | L2 regularization; 5-fold CV over α ∈ {0.01, 0.1, 1, 10, 50, 100} |
| **Lasso** | L1 regularization; same CV grid (implicit feature selection) |
| **Random Forest** | 200 trees, `max_depth` 10–15, bootstrap ensemble for non-linear interactions |

### 5.2 Results — Track A (Euros)

| Model | RMSE(log) | MAE(log) | R²(log) | RMSE (€) | MAE (€) |
|---|---:|---:|---:|---:|---:|
| OLS | 0.2759 | 0.2227 | 0.7847 | 341.77 | 244.59 |
| Ridge (α=1) | 0.2754 | 0.2226 | 0.7854 | 338.63 | 243.84 |
| Lasso (α=0.01) | 0.2732 | 0.2179 | 0.7889 | 333.53 | 233.94 |
| **Random Forest** | **0.2088** | **0.1624** | **0.8767** | **302.51** | **185.76** |

### 5.3 Results — Track B (BDT)

| Model | RMSE(log) | MAE(log) | R²(log) | RMSE (৳) | MAE (৳) |
|---|---:|---:|---:|---:|---:|
| OLS | 0.2633 | 0.2227 | 0.7789 | 62,704 | 38,760 |
| Ridge | 0.2616 | 0.2196 | 0.7818 | 58,551 | 37,255 |
| Lasso | 0.2654 | 0.2259 | 0.7754 | 65,903 | 39,903 |
| **Random Forest** | **0.2404** | **0.1862** | **0.8157** | **59,230** | **33,902** |

### 5.4 Diagnostics

Residual-vs-fitted plots and Q-Q plots for the best model on each track show that residuals are near-normal and roughly homoscedastic after the log transform.

| Track A | Track B |
|---|---|
| ![Track A diagnostics](../data/track_a/cleaned/diagnostics.png) | ![Track B diagnostics](../data/track_b/cleaned/diagnostics.png) |

### 5.5 Bonus: price-band classification (Track B)

To validate the engineered features from a second perspective, Track B price was discretized into four bands — Budget (<75k), Mid-range (75–125k), Premium (125–200k), High-end (≥200k BDT) — and a `RandomForestClassifier` trained on the identical train/test split:

| Metric | Value |
|---|---:|
| **Accuracy** | 0.6824 |
| **Precision** (macro / weighted) | 0.6934 / 0.6851 |
| **Recall** (macro / weighted) | 0.5846 / 0.6824 |
| **F1-Score** (macro / weighted) | 0.6150 / 0.6708 |

![Track B confusion matrix](../data/track_b/cleaned/confusion_matrix.png)

High-end laptops are classified almost perfectly (F1 = 0.85) while the Budget band is hardest (F1 = 0.40) — a direct consequence of heavy class imbalance (only 7 Budget samples in the test set).

---

## 6. Discussion

**Why Random Forest wins on both tracks.** Linear models (OLS/Ridge/Lasso) already explain ~78% of log-price variance, which tells us most of the price signal is additive in RAM, storage, CPU, and display specs. Random Forest adds ~9–12 points of R² (0.785 → 0.877 on Track A) because laptop pricing is inherently *interaction-driven*: a high-end GPU is worth far more when paired with 32 GB RAM and a fast CPU than with 4 GB and a Celeron. Tree ensembles capture these multiplicative synergies naturally, and their feature bagging also resists the small number of genuinely predictive columns. The Spec Power Score and GPU Performance Index explicitly encode the same non-linear intuition.

**Why the models do not perform better.** Several hypotheses, in descending order of impact:

1. **GPU granularity lost.** Only GPU *brand* is retained; `RTX 3050` and `RTX 4090` look identical to most models. GPU tier is likely the single largest untapped signal.
2. **Brand premium invisible.** `Company` (19 levels) was dropped from Track A to avoid OLS overfitting, so intangible Apple/Razer premiums are uncaptured. Cross-validated target encoding could recover some of this.
3. **Rare high-end tail.** Extreme machines (>4,000 € / >500,000 BDT) are <3% of the data; models regress toward the mean and systematically underprice them — visible in the worst-prediction analysis where a 6,099 € laptop is predicted at 4,160 €.
4. **Conflicting spec signals.** Genuine edge cases (4K screen, budget CPU, no discrete GPU) genuinely confuse the model.
5. **Data scarcity (Track B).** 423 usable rows yield only ~85 test observations; metrics carry high variance. The price-band classification confirms this: the least-populated class (Budget) has the worst F1 (0.40).
6. **Log-transform asymmetry.** Back-transforming (`expm1`) makes overpredictions costlier in raw RMSE than underpredictions at high prices, slightly inflating reported error.

**Consistency between regression and classification.** The two evaluations agree: both struggle on the same boundaries. The confusion matrix shows Mid-range ↔ Premium confusion (the bulk of the market), mirroring the regression's difficulty precisely where the linear-vs-nonlinear decision boundary is fuzzy.

---

## 7. Conclusion

This project delivers a complete, reproducible laptop price-regression system spanning two very different markets. Track A (Kaggle, €) achieved an R² of **0.877** and Track B (self-scraped, ৳) **0.816** using a Random Forest trained on log-transformed prices — with clean diagnostics, documented imputations, and explicit failure-mode analysis throughout. A bonus classification evaluation (confusion matrix, accuracy 0.6824, macro-F1 0.6150) confirms the feature set from a second angle. Everything — from scraping `startech.com.bd` and `ryans.com` with a polite 1.5 s request delay, to parsing regex-heavy spec strings, to the final residual plots — runs end-to-end with zero manual intervention via a one-click Windows launcher.

The biggest lessons were practical: real-world data is messy, and preprocessing choices (log transform, no silent `dropna`, retained-but-handled outliers) affected performance more than model choice. The largest opportunities for improvement are clear: scrape individual product detail pages to recover GPU tier and refresh rates, expand Track B to 500+ rows, target-encode high-cardinality brands, and publish the final model as a price-estimation API for the Bangladeshi market. Given more time, I would pursue exactly these directions; the pipeline is already structured so that each improvement is one small, testable change away.
