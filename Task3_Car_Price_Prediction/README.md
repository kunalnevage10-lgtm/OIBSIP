# 🚗 Task 3: Car Price Prediction with Machine Learning

<p align="center">
  <img src="https://img.shields.io/badge/Task-Regression%20%26%20Machine%20Learning-orange?style=for-the-badge" alt="ML" />
  <img src="https://img.shields.io/badge/Domain-Automotive%20Valuation-purple?style=for-the-badge" alt="Automotive" />
  <img src="https://img.shields.io/badge/Algorithms-Gradient%20Boosting%20%7C%20Random%20Forest-blue?style=for-the-badge" alt="Algorithms" />
</p>

## 🎯 Overview
An end-to-end Machine Learning regression solution designed to predict the resale price of used cars based on multi-dimensional vehicle attributes including **brand/make, vehicle age, odometer mileage, fuel type, transmission, seller type, and ownership tier**.

Using the authentic **CarDekho Vehicle Dataset** (`car_details.csv`), this project covers data hygiene, feature engineering, exploratory data analysis, categorical encoding, multi-model regression training, and feature importance diagnostics.

---

## 📊 Key Highlights & Workflow

1. **Data Cleaning & Sanitization:**
   - Audited null values and eliminated **763 duplicate records** (17.58% of raw data).
   - Normalized categorical string cases and whitespaces (standardizing `"Petrol"` vs `"petrol"`, seller types, and transmission).

2. **Domain Feature Engineering:**
   - Derived **`car_age`** from manufacturing year ($\text{Reference Year } 2020 - \text{year}$).
   - Extracted and cleaned **`brand`** from vehicle title (handling edge cases like *Land Rover*, *BMW*, *MG*, *Opel*).
   - Engineered **`km_per_year`** as an annualized usage intensity metric ($\frac{\text{km\_driven}}{\text{car\_age} + 1}$).

3. **Exploratory Data Analysis (EDA):**
   - Target price distribution analysis (addressing severe right-skewness of raw prices).
   - Price vs. Fuel Type box plots (highlighting Diesel price premium over Petrol/CNG).
   - Price vs. Car Age scatter plots with non-linear vehicle depreciation trendlines.
   - Annotated correlation heatmap uncovering primary price determinants.

4. **Machine Learning Model Benchmark:**
   - Evaluated 4 distinct regression architectures across **MAE, RMSE, and $R^2$**:
     - Baseline Linear Regression (OLS)
     - Ridge Regression ($L_2$ Regularized)
     - Random Forest Regressor (Non-linear Bagging Ensemble)
     - Gradient Boosting Regressor (Boosting Ensemble)

---

## 📈 Model Performance Comparison (Holdout Test Set)

| Regression Model | Test MAE (INR) | Test RMSE (INR) | Test $R^2$ | Train $R^2$ |
| :--- | :---: | :---: | :---: | :---: |
| **Linear Regression** | ₹190,726.41 | ₹389,518.61 | 0.5290 | 0.5813 |
| **Ridge Regression ($\alpha=10.0$)** | ₹191,185.12 | ₹403,443.99 | 0.4947 | 0.5829 |
| **Random Forest Regressor** | ₹160,001.14 | ₹363,974.08 | 0.5888 | 0.9019 |
| **Gradient Boosting Regressor** 🏆 | **₹152,155.59** | **₹363,956.98** | **0.5888** | **0.9186** |

> **Optimal Model:** **Gradient Boosting Regressor** achieved the best predictive accuracy, cutting mean absolute error down to **₹1.52 Lakhs** (a ₹38,570 improvement per vehicle over linear regression).

---

## 🧠 Top Price Determinants (Feature Importance)

According to the Gradient Boosting model, used car market valuations are driven by:
1. **Transmission Mechanism (28.5%):** Manual transmission carries the steepest discount; automatic gearboxes demand a high premium.
2. **Vehicle Age (22.0%):** Exponential depreciation over the initial 3–5 years.
3. **Usage Intensity & Mileage (16.2%):** Combined impact of `km_per_year` and `km_driven`.
4. **Powertrain & Fuel Type (12.3%):** Diesel vehicles retain higher resale equity than Petrol counterparts.
5. **Brand Hierarchy:** Luxury marques (*BMW, Audi, Mercedes-Benz, Toyota*) command distinct valuation floors.

---

## 🚀 How to Run

### Interactive Notebook
Open [`Car_Price_Prediction_Kunal.ipynb`](Car_Price_Prediction_Kunal.ipynb) in Jupyter Lab / Notebook:
```bash
jupyter notebook Car_Price_Prediction_Kunal.ipynb
```

### Standalone Python Pipeline
Execute the complete training and inference pipeline:
```bash
python car_price_prediction.py
```
Outputs and charts will be saved directly into `plots/` and `models/`.
