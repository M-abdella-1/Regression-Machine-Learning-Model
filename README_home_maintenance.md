# 🔧 Home Maintenance Technicians — Price Prediction

A full machine learning pipeline that predicts the **base service price (EGP)** of home maintenance technicians in Egypt. The project covers exploratory data analysis (EDA) with rich visualisations, feature engineering, and a comparison of three regression models.

---

## 📁 Project Structure

```
├── home_maintenance_technicians.ipynb   # Main notebook
└── home_maintenance_technicians.csv     # Input dataset
```

---

## 📊 Dataset

| Property | Detail |
|---|---|
| Format | `.csv` |
| Target variable | `base_price_EGP` — service price in Egyptian Pounds |
| Key features | `experience_years`, `rating`, `number_of_reviews`, `completed_jobs`, `service_type`, `availability_status` |
| Dropped columns | `technician_id` (identifier), `technician_name` (free text) |

---

## 🔍 Exploratory Data Analysis (EDA)

### 1. Data Quality Audit
- Missing value count per column (`isna().sum()`)
- Duplicate row detection (`duplicated().sum()`)
- Descriptive statistics (`describe()`) — mean, std, quartiles, min/max
- Schema inspection (`info()`) — dtypes and non-null counts

### 2. Visualisations

| Chart | What it reveals |
|---|---|
| **Histograms** (5 numeric cols) | Distribution shape of each feature — skewness, modality, spread |
| **Boxplots** (5 numeric cols) | Outliers and interquartile range for each numeric feature |
| **Correlation Heatmap** | Pairwise Pearson correlation between all numeric features |
| **KDE Distribution Plots** (4 features) | Smoothed probability distribution of each predictor variable |
| **Boxplot: Service Type vs Price** | Price distribution per service category — which services cost more |
| **Boxplot: Availability Status vs Price** | Price differences across availability states |
| **Line Plot: Rating vs Price** | Mean price at each rating level with confidence interval |
| **Correlation with Target** | Sorted numeric correlation of all features against `base_price_EGP` |

---

## ⚙️ ML Pipeline

### Feature Engineering
- **Dropped** identifier and free-text columns (`technician_id`, `technician_name`)
- **One-Hot Encoded** all categorical columns with `pd.get_dummies(drop_first=True)` to avoid multicollinearity

### Train / Test Split
- **80% training / 20% test** using `train_test_split(random_state=42)`

---

## 🤖 Models

### 1. Linear Regression (Baseline)
The simplest regression model — fits a weighted linear combination of features to minimise squared error. Used as the benchmark all other models must beat.

### 2. Random Forest Regressor
An ensemble of 100 decision trees, each trained on a random bootstrap sample of the data with random feature subsets at each split. Final prediction is the average of all trees. Handles non-linear relationships and feature interactions automatically.

- `n_estimators = 100`
- `random_state = 42`

### 3. Gradient Boosting Regressor
Builds trees **sequentially** — each new tree targets the residual errors of all previous trees. Often the most accurate model but more sensitive to hyperparameters.

- `n_estimators = 100`
- `learning_rate = 0.1`
- `random_state = 42`

---

## 📏 Evaluation Metrics

| Metric | Formula | Meaning |
|---|---|---|
| **MAE** | mean(abs(y_true − y_pred)) | Average prediction error in EGP |
| **RMSE** | sqrt(mean((y_true − y_pred)²)) | Error in EGP — penalises large mistakes more than MAE |
| **R²** | 1 − SS_res / SS_tot | Proportion of price variance explained (0 = baseline, 1 = perfect) |
| **CV R²** | 5-fold cross-validation mean R² | More robust performance estimate than a single split |

---

## 🔄 Cross-Validation

5-fold cross-validation is run on the **training set** for the Linear Regression model to produce a reliable, split-independent performance estimate. Every row is used as a test sample exactly once across the 5 folds.

---

## ⚙️ Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

| Library | Purpose |
|---|---|
| `pandas` | Data loading, manipulation, one-hot encoding |
| `numpy` | Numerical support |
| `matplotlib` | Base plotting engine |
| `seaborn` | Statistical visualisations |
| `scikit-learn` | ML models, train/test split, metrics, cross-validation |

---

## 🚀 How to Run

1. Place `home_maintenance_technicians.csv` in the same directory as the notebook
2. Install dependencies (see above)
3. Open and run all cells top to bottom:

```bash
jupyter notebook home_maintenance_technicians.ipynb
```

---

## 👤 Author

**Mostafa** — Mathematics & Computer Science student, Menoufia University  
ML Training at the National Telecommunication Institute (NTI)
