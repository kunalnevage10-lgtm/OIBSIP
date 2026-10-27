"""
=============================================================================
OASIS INFOBYTE DATA SCIENCE INTERNSHIP
TASK 3: Car Price Prediction with Machine Learning
Author: Kunal Nevage (OIBSIP Data Science Intern)
Dataset: Vehicle Dataset from CarDekho
Tech Stack: Python, Pandas, Scikit-Learn, Matplotlib, Seaborn
=============================================================================
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

# Set matplotlib aesthetic defaults
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300

os.makedirs("plots", exist_ok=True)
os.makedirs("models", exist_ok=True)


def load_and_clean_data(filepath="car_details.csv"):
    """
    Load raw vehicle dataset, perform missing value checks,
    drop duplicate observations, and sanitize categorical values.
    """
    print("=" * 60)
    print("STEP 1: DATA INGESTION & DATA CLEANING")
    print("=" * 60)

    df = pd.read_csv(filepath)
    print(f"Raw dataset loaded. Shape: {df.shape}")
    print("\nInitial Info:")
    df.info()

    # Null values audit
    null_counts = df.isnull().sum()
    print("\nMissing values per column:\n", null_counts)

    # Duplicate rows handling
    dup_count = df.duplicated().sum()
    print(f"\nDuplicate rows identified: {dup_count} ({dup_count / len(df) * 100:.2f}%)")
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Dataset shape after dropping duplicates: {df.shape}")

    # Standardize string formatting and categorical consistency
    # E.g. handles whitespace, casing anomalies like 'Petrol' vs 'petrol'
    for col in ["fuel", "seller_type", "transmission", "owner"]:
        df[col] = df[col].astype(str).str.strip().str.title()

    # Normalize 'owner' values if needed
    print("\nSanitized unique categorical values:")
    print("  Fuel types:    ", df["fuel"].unique())
    print("  Seller types:  ", df["seller_type"].unique())
    print("  Transmissions: ", df["transmission"].unique())
    print("  Owner categories:", df["owner"].unique())

    return df


def engineer_features(df, reference_year=2020):
    """
    Engineer key domain features:
    1. car_age: Reference year minus manufacturing year
    2. brand: Extracted automotive manufacturer name from car name
    3. km_per_year: Average annual distance traveled
    """
    print("\n" + "=" * 60)
    print("STEP 2: FEATURE ENGINEERING")
    print("=" * 60)

    df = df.copy()

    # 1. Car Age Calculation
    df["car_age"] = reference_year - df["year"]
    # Ensure no negative age
    df["car_age"] = df["car_age"].clip(lower=0)

    # 2. Brand Extraction & Standardization
    def extract_brand(name):
        parts = str(name).strip().split()
        if not parts:
            return "Unknown"
        first = parts[0].title()
        if first == "Land":
            return "Land Rover"
        elif first.lower() == "opelcorsa":
            return "Opel"
        elif first.lower() in ["bmw", "mg"]:
            return first.upper()
        return first

    df["brand"] = df["name"].apply(extract_brand)

    # 3. km_per_year (Annual usage intensity)
    df["km_per_year"] = (df["km_driven"] / (df["car_age"] + 1)).round(1)

    print(f"Extracted {df['brand'].nunique()} unique brands.")
    print("Top 10 brands by listing volume:")
    print(df["brand"].value_counts().head(10))

    print("\nEngineered Feature Summary:")
    print(df[["year", "car_age", "km_driven", "km_per_year", "brand", "selling_price"]].head())

    return df


def perform_eda(df):
    """
    Perform thorough Exploratory Data Analysis (EDA) and save visualizations:
    - Distribution of selling prices (raw & log)
    - Price vs. Fuel Type box plots
    - Price vs. Car Age scatter plot
    - Price vs. Transmission box plot
    - Top brands by market share and average price
    - Feature correlation heatmap
    """
    print("\n" + "=" * 60)
    print("STEP 3: EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 60)

    # 1. Distribution of Selling Prices
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(df["selling_price"] / 1e5, kde=True, color="#2b5c8f", ax=axes[0], bins=35)
    axes[0].set_title("Distribution of Selling Prices (in Lakhs INR)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Selling Price (Lakhs INR)", fontsize=11)
    axes[0].set_ylabel("Frequency", fontsize=11)

    # Log transformed price distribution (to highlight normal distribution after transformation)
    sns.histplot(np.log1p(df["selling_price"]), kde=True, color="#1b7837", ax=axes[1], bins=35)
    axes[1].set_title("Distribution of Log(Selling Price)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Log(Selling Price)", fontsize=11)
    axes[1].set_ylabel("Frequency", fontsize=11)
    plt.tight_layout()
    plt.savefig("plots/eda_price_distribution.png")
    plt.close()
    print("Saved plots/eda_price_distribution.png")

    # 2. Price vs. Fuel Type Box Plot
    plt.figure(figsize=(10, 6))
    order_fuel = df.groupby("fuel")["selling_price"].median().sort_values(ascending=False).index
    sns.boxplot(
        data=df,
        x="fuel",
        y=df["selling_price"] / 1e5,
        order=order_fuel,
        color="#2b5c8f",
        showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "red", "markeredgecolor": "black", "markersize": "7"}
    )
    plt.title("Car Selling Price vs. Fuel Type (Red dot = Mean)", fontsize=14, fontweight="bold")
    plt.xlabel("Fuel Type", fontsize=12)
    plt.ylabel("Selling Price (Lakhs INR)", fontsize=12)
    plt.ylim(0, 35)  # Zoom in to cover bulk of data while retaining key upper tier
    plt.tight_layout()
    plt.savefig("plots/eda_price_vs_fuel.png")
    plt.close()
    print("Saved plots/eda_price_vs_fuel.png")

    # 3. Price vs. Car Age Scatter Plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x="car_age",
        y=df["selling_price"] / 1e5,
        hue="fuel",
        palette="viridis",
        alpha=0.6,
        s=50
    )
    # Fit a smooth polynomial / linear trend line
    sns.regplot(
        data=df,
        x="car_age",
        y=df["selling_price"] / 1e5,
        scatter=False,
        color="#d95f02",
        line_kws={"linewidth": 2.5, "label": "Depreciation Trend"}
    )
    plt.title("Car Selling Price vs. Car Age (Depreciation Trajectory)", fontsize=14, fontweight="bold")
    plt.xlabel("Car Age (Years)", fontsize=12)
    plt.ylabel("Selling Price (Lakhs INR)", fontsize=12)
    plt.ylim(0, 45)
    plt.legend(title="Fuel Type", loc="upper right")
    plt.tight_layout()
    plt.savefig("plots/eda_price_vs_age.png")
    plt.close()
    print("Saved plots/eda_price_vs_age.png")

    # 4. Price vs Transmission & Seller Type
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.boxplot(data=df, x="transmission", y=df["selling_price"] / 1e5, color="#4daf4a", ax=axes[0])
    axes[0].set_title("Selling Price by Transmission Type", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Transmission", fontsize=11)
    axes[0].set_ylabel("Selling Price (Lakhs INR)", fontsize=11)
    axes[0].set_ylim(0, 35)

    sns.boxplot(data=df, x="seller_type", y=df["selling_price"] / 1e5, color="#ff7f00", ax=axes[1])
    axes[1].set_title("Selling Price by Seller Type", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Seller Type", fontsize=11)
    axes[1].set_ylabel("Selling Price (Lakhs INR)", fontsize=11)
    axes[1].set_ylim(0, 35)
    plt.tight_layout()
    plt.savefig("plots/eda_transmission_seller.png")
    plt.close()
    print("Saved plots/eda_transmission_seller.png")

    # 5. Top Brands Market Share & Average Valuation
    top10_brands = df["brand"].value_counts().head(10).index
    df_top10 = df[df["brand"].isin(top10_brands)]
    brand_stats = df_top10.groupby("brand")["selling_price"].agg(["count", "mean"]).reset_index()
    brand_stats["mean_lakhs"] = brand_stats["mean"] / 1e5
    brand_stats = brand_stats.sort_values(by="mean_lakhs", ascending=False)

    plt.figure(figsize=(11, 5))
    barplot = sns.barplot(data=brand_stats, x="brand", y="mean_lakhs", color="#3182bd")
    plt.title("Average Selling Price Across Top 10 High-Volume Brands", fontsize=13, fontweight="bold")
    plt.xlabel("Brand", fontsize=11)
    plt.ylabel("Average Selling Price (Lakhs INR)", fontsize=11)
    for p in barplot.patches:
        barplot.annotate(
            f"{p.get_height():.2f}L",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center", va="bottom", fontsize=9, xytext=(0, 3),
            textcoords="offset points"
        )
    plt.tight_layout()
    plt.savefig("plots/eda_top_brands_avg_price.png")
    plt.close()
    print("Saved plots/eda_top_brands_avg_price.png")

    # 6. Feature Correlation Heatmap
    # Create encoded subset for correlation
    df_corr = df[["car_age", "km_driven", "km_per_year", "selling_price"]].copy()
    # Add key binary indicators
    df_corr["is_automatic"] = (df["transmission"] == "Automatic").astype(int)
    df_corr["is_diesel"] = (df["fuel"] == "Diesel").astype(int)
    df_corr["is_individual"] = (df["seller_type"] == "Individual").astype(int)
    df_corr["is_first_owner"] = (df["owner"] == "First Owner").astype(int)

    plt.figure(figsize=(9, 7))
    corr_matrix = df_corr.corr()
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.8,
        cbar_kws={"shrink": 0.8}
    )
    plt.title("Feature Correlation Heatmap with Selling Price", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig("plots/eda_correlation_heatmap.png")
    plt.close()
    print("Saved plots/eda_correlation_heatmap.png")


def preprocess_and_split(df, target_col="selling_price", test_size=0.2, random_state=42):
    """
    Prepare feature matrix X and target y:
    - Group rare brands (< 10 listings) into 'Other' to curb high dimensionality
    - Perform One-Hot Encoding on nominal categorical variables
    - Split into 80/20 train/test sets
    """
    print("\n" + "=" * 60)
    print("STEP 4: CATEGORICAL ENCODING & TRAIN/TEST SPLIT")
    print("=" * 60)

    df_proc = df.copy()

    # Consolidate rare brands
    brand_counts = df_proc["brand"].value_counts()
    frequent_brands = brand_counts[brand_counts >= 10].index
    df_proc["brand"] = df_proc["brand"].apply(lambda b: b if b in frequent_brands else "Other")
    print(f"Retained {len(frequent_brands)} frequent brands, grouped rest into 'Other'.")

    # Define feature set
    feature_cols = ["brand", "car_age", "km_driven", "km_per_year", "fuel", "seller_type", "transmission", "owner"]
    X_raw = df_proc[feature_cols]
    y = df_proc[target_col]

    # One-Hot Encoding
    X = pd.get_dummies(X_raw, drop_first=True)
    print(f"Feature matrix shape after One-Hot Encoding: {X.shape}")

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"Train split: {X_train.shape[0]} samples ({1-test_size:.0%})")
    print(f"Test split:  {X_test.shape[0]} samples ({test_size:.0%})")

    return X_train, X_test, y_train, y_test, list(X.columns)


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Train 4 regression models:
    1. Linear Regression
    2. Ridge Regression
    3. Random Forest Regressor
    4. Gradient Boosting Regressor
    Evaluate using MAE, RMSE, and R2 on both train and test partitions.
    """
    print("\n" + "=" * 60)
    print("STEP 5: MODEL TRAINING & RIGOROUS EVALUATION")
    print("=" * 60)

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=10.0),
        "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_split=4, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.08, max_depth=5, random_state=42)
    }

    results = []
    fitted_models = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        fitted_models[name] = model

        # Predictions
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)

        # Metrics calculation
        train_mae = mean_absolute_error(y_train, train_preds)
        train_rmse = root_mean_squared_error(y_train, train_preds)
        train_r2 = r2_score(y_train, train_preds)

        test_mae = mean_absolute_error(y_test, test_preds)
        test_rmse = root_mean_squared_error(y_test, test_preds)
        test_r2 = r2_score(y_test, test_preds)

        results.append({
            "Model": name,
            "Train MAE": train_mae,
            "Test MAE": test_mae,
            "Train RMSE": train_rmse,
            "Test RMSE": test_rmse,
            "Train R2": train_r2,
            "Test R2": test_r2
        })

        print(f"  --> Test MAE:  {test_mae:,.2f} INR")
        print(f"  --> Test RMSE: {test_rmse:,.2f} INR")
        print(f"  --> Test R²:   {test_r2:.4f}")

    results_df = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print("SUMMARY COMPARISON TABLE")
    print("=" * 60)
    print(results_df.to_string(index=False))

    # Save metrics table to CSV
    results_df.to_csv("plots/model_evaluation_metrics.csv", index=False)

    # Plot Model Comparison Bar Chart
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    sns.barplot(data=results_df, x="Model", y="Test R2", color="#2b5c8f", ax=axes[0])
    axes[0].set_title("Test R² Score (Higher is Better)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("R² Score", fontsize=11)
    axes[0].tick_params(axis="x", rotation=25)
    for p in axes[0].patches:
        axes[0].annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha="center", va="bottom", fontsize=10, xytext=(0, 2), textcoords="offset points")

    sns.barplot(data=results_df, x="Model", y="Test MAE", color="#e6550d", ax=axes[1])
    axes[1].set_title("Test MAE in INR (Lower is Better)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("MAE (INR)", fontsize=11)
    axes[1].tick_params(axis="x", rotation=25)
    for p in axes[1].patches:
        axes[1].annotate(f"{p.get_height()/1e5:.2f}L", (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha="center", va="bottom", fontsize=10, xytext=(0, 2), textcoords="offset points")

    sns.barplot(data=results_df, x="Model", y="Test RMSE", color="#31a354", ax=axes[2])
    axes[2].set_title("Test RMSE in INR (Lower is Better)", fontsize=12, fontweight="bold")
    axes[2].set_ylabel("RMSE (INR)", fontsize=11)
    axes[2].tick_params(axis="x", rotation=25)
    for p in axes[2].patches:
        axes[2].annotate(f"{p.get_height()/1e5:.2f}L", (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha="center", va="bottom", fontsize=10, xytext=(0, 2), textcoords="offset points")

    plt.tight_layout()
    plt.savefig("plots/model_metrics_comparison.png")
    plt.close()
    print("Saved plots/model_metrics_comparison.png")

    # Select best model based on highest test R2 score
    best_row = results_df.sort_values(by="Test R2", ascending=False).iloc[0]
    best_model_name = best_row["Model"]
    best_model = fitted_models[best_model_name]
    print(f"\nOptimal Model Selected: {best_model_name} (R²: {best_row['Test R2']:.4f}, MAE: {best_row['Test MAE']:,.2f})")

    return results_df, fitted_models, best_model_name, best_model


def plot_feature_importance_and_diagnostics(best_model, best_model_name, X_test, y_test, feature_names):
    """
    Generate feature importance bar chart and diagnostic Actual vs. Predicted plots.
    """
    print("\n" + "=" * 60)
    print("STEP 6: FEATURE IMPORTANCE & DIAGNOSTIC VISUALIZATION")
    print("=" * 60)

    # Feature Importance (for tree-based models)
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        feat_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

        top15_feat = feat_df.head(15)

        plt.figure(figsize=(10, 7))
        barplot = sns.barplot(data=top15_feat, x="Importance", y="Feature", color="#2c7fb8")
        plt.title(f"Top 15 Feature Importances — {best_model_name}", fontsize=13, fontweight="bold")
        plt.xlabel("Gini / Reduction in Impurity Importance", fontsize=11)
        plt.ylabel("Predictor Feature", fontsize=11)
        for p in barplot.patches:
            barplot.annotate(
                f"{p.get_width():.3f}",
                (p.get_width(), p.get_y() + p.get_height() / 2.),
                ha="left", va="center", fontsize=9, xytext=(4, 0),
                textcoords="offset points"
            )
        plt.tight_layout()
        plt.savefig("plots/feature_importance.png")
        plt.close()
        print("Saved plots/feature_importance.png")

        print("\nTop 10 Most Influential Features:")
        print(feat_df.head(10).to_string(index=False))

    # Actual vs Predicted Plot
    test_preds = best_model.predict(X_test)
    residuals = y_test - test_preds

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Subplot 1: Actual vs Predicted
    axes[0].scatter(y_test / 1e5, test_preds / 1e5, alpha=0.5, color="#1f77b4", edgecolor="none")
    max_val = max((y_test / 1e5).max(), (test_preds / 1e5).max())
    axes[0].plot([0, max_val], [0, max_val], color="red", linestyle="--", linewidth=2, label="Perfect Fit (y = x)")
    axes[0].set_title(f"Actual vs. Predicted Selling Price ({best_model_name})", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Actual Selling Price (Lakhs INR)", fontsize=11)
    axes[0].set_ylabel("Predicted Selling Price (Lakhs INR)", fontsize=11)
    axes[0].legend()
    axes[0].set_xlim(0, 45)
    axes[0].set_ylim(0, 45)

    # Subplot 2: Residuals Distribution
    sns.histplot(residuals / 1e5, kde=True, color="#2ca02c", ax=axes[1], bins=35)
    axes[1].axvline(0, color="red", linestyle="--", linewidth=1.5)
    axes[1].set_title("Residual Error Distribution (Actual - Predicted)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Residual Error (Lakhs INR)", fontsize=11)
    axes[1].set_ylabel("Density / Count", fontsize=11)
    axes[1].set_xlim(-15, 15)

    plt.tight_layout()
    plt.savefig("plots/actual_vs_predicted.png")
    plt.close()
    print("Saved plots/actual_vs_predicted.png")


def save_best_pipeline_and_demonstrate(best_model, feature_names):
    """
    Persist the trained model and feature schema using joblib,
    and demonstrate single-sample inference.
    """
    print("\n" + "=" * 60)
    print("STEP 7: MODEL PERSISTENCE & INFERENCE DEMONSTRATION")
    print("=" * 60)

    model_path = "models/best_car_price_model.joblib"
    features_path = "models/model_features.joblib"

    joblib.dump(best_model, model_path)
    joblib.dump(feature_names, features_path)
    print(f"Model saved to {model_path}")
    print(f"Features saved to {features_path}")

    # Inference function
    def estimate_price(brand, car_age, km_driven, fuel, seller_type, transmission, owner):
        km_per_year = round(km_driven / (car_age + 1), 1)
        input_data = {
            "car_age": car_age,
            "km_driven": km_driven,
            "km_per_year": km_per_year
        }

        # Initialize all one-hot columns with 0
        for col in feature_names:
            if col not in input_data:
                input_data[col] = 0

        # Set specific active categories
        brand_col = f"brand_{brand}"
        if brand_col in input_data:
            input_data[brand_col] = 1
        elif "brand_Other" in input_data:
            input_data["brand_Other"] = 1

        fuel_col = f"fuel_{fuel}"
        if fuel_col in input_data:
            input_data[fuel_col] = 1

        seller_col = f"seller_type_{seller_type}"
        if seller_col in input_data:
            input_data[seller_col] = 1

        trans_col = f"transmission_{transmission}"
        if trans_col in input_data:
            input_data[trans_col] = 1

        owner_col = f"owner_{owner}"
        if owner_col in input_data:
            input_data[owner_col] = 1

        input_df = pd.DataFrame([input_data])[feature_names]
        predicted_val = best_model.predict(input_df)[0]
        return max(predicted_val, 20000)

    # Test cases
    test_cars = [
        {"desc": "Budget Hatchback (Maruti Swift, 4 yrs, 45k km, Petrol, Manual)",
         "brand": "Maruti", "car_age": 4, "km_driven": 45000, "fuel": "Petrol",
         "seller_type": "Individual", "transmission": "Manual", "owner": "First Owner"},
        {"desc": "Family Diesel Sedan (Hyundai Verna, 5 yrs, 60k km, Diesel, Manual)",
         "brand": "Hyundai", "car_age": 5, "km_driven": 60000, "fuel": "Diesel",
         "seller_type": "Dealer", "transmission": "Manual", "owner": "First Owner"},
        {"desc": "Luxury Automatic Sedan (BMW 3 Series, 6 yrs, 50k km, Diesel, Automatic)",
         "brand": "BMW", "car_age": 6, "km_driven": 50000, "fuel": "Diesel",
         "seller_type": "Dealer", "transmission": "Automatic", "owner": "First Owner"}
    ]

    print("\nDemonstration of Model Inference on Real-World Car Profiles:")
    for car in test_cars:
        price = estimate_price(
            brand=car["brand"],
            car_age=car["car_age"],
            km_driven=car["km_driven"],
            fuel=car["fuel"],
            seller_type=car["seller_type"],
            transmission=car["transmission"],
            owner=car["owner"]
        )
        print(f"\nProfile: {car['desc']}")
        print(f"  -> Predicted Market Price: {price:,.2f} INR (approx. {price/1e5:.2f} Lakhs)")


def main():
    print("=" * 70)
    print("OASIS INFOBYTE DATA SCIENCE TASK 3: CAR PRICE PREDICTION PIPELINE")
    print("=" * 70)

    # 1. Ingestion & Cleaning
    df = load_and_clean_data("car_details.csv")

    # 2. Feature Engineering
    df = engineer_features(df, reference_year=2020)

    # 3. Exploratory Data Analysis
    perform_eda(df)

    # 4. Preprocessing & Train/Test Split
    X_train, X_test, y_train, y_test, feature_names = preprocess_and_split(df)

    # 5. Model Training & Evaluation
    results_df, fitted_models, best_model_name, best_model = train_and_evaluate_models(
        X_train, X_test, y_train, y_test
    )

    # 6. Feature Importance & Diagnostics
    plot_feature_importance_and_diagnostics(
        best_model, best_model_name, X_test, y_test, feature_names
    )

    # 7. Model Persistence & Inference Demonstration
    save_best_pipeline_and_demonstrate(best_model, feature_names)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
