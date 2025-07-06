# Imports
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import shap
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor, BaggingRegressor
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from scipy import stats

# Set working directory and load data
os.chdir("C:\\Users\\zohaib khan\\OneDrive\\Desktop\\USE ME\\dump\\zk")
df = pd.read_csv("medical_insurance.csv")
df.drop_duplicates(inplace=True)

# Label Encoding
for col in df.select_dtypes(include='object').columns:
    df[col] = LabelEncoder().fit_transform(df[col])

# Log transform the target variable
y = np.log1p(df["premium"])
X = df.drop("premium", axis=1)

# Normality Test Summary
stat, p_value = stats.shapiro(y)
print("\n🧪 Normality Test (Shapiro-Wilk) on Log-Transformed Target:")
print(f"  Statistic: {stat:.4f}, p-value: {p_value:.4f}")
if p_value < 0.05:
    print("  ❗ The log-transformed target is not normally distributed (p < 0.05).\n  → Consider using non-parametric models or trying different transformations (e.g., Box-Cox).")
else:
    print("  ✅ The log-transformed target appears to be normally distributed (p ≥ 0.05).")

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standard Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_scaled = scaler.fit_transform(X)

# MAPE function
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_idx = y_true != 0
    return np.mean(np.abs((y_true[non_zero_idx] - y_pred[non_zero_idx]) / y_true[non_zero_idx])) * 100

# Evaluation function
def evaluate_model(y_true, y_pred, model_name, model_obj):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    print(f"\n📊 {model_name} Performance:")
    print(f"  Train Score: {model_obj.score(X_train_scaled, y_train):.4f}")
    print(f"  Test Score : {model_obj.score(X_test_scaled, y_test):.4f}")
    print(f"  R² Score   : {r2:.4f}")
    print(f"  RMSE       : {rmse:.2f}")
    print(f"  MAE        : {mae:.2f}")
    print(f"  MAPE       : {mape:.2f}%")

# Models dictionary
models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "Decision Tree": DecisionTreeRegressor(),
    "KNN": KNeighborsRegressor(),
    "SVR": SVR(),
    "Gradient Boosting": GradientBoostingRegressor(),
    "Extra Trees": ExtraTreesRegressor(),
    "XGBoost": XGBRegressor(),
    "CatBoost": CatBoostRegressor(verbose=0),
    "LGBM": LGBMRegressor(),
    "AdaBoost": AdaBoostRegressor(),
    "Bagging": BaggingRegressor(),
    "GaussianNB": GaussianNB(),
    "LDA": LinearDiscriminantAnalysis(),
    "QDA": QuadraticDiscriminantAnalysis()
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = []
feature_importances = {}

# Model loop
for name, model in models.items():
    try:
        model.fit(X_train_scaled, y_train)
        y_pred_log = model.predict(X_test_scaled)
        y_pred = np.expm1(y_pred_log)

        # Save predictions for residual plot
        if name == "Linear Regression":
            df["predicted_premium"] = np.expm1(model.predict(X_scaled))
            df["residual"] = df["premium"] - df["predicted_premium"]

        evaluate_model(np.expm1(y_test), y_pred, name, model)

        # Cross-validation
        cv_score = cross_val_score(model, X_scaled, y, cv=kf, scoring='r2').mean()
        print(f"  CV R² Score: {cv_score:.4f}")
        results.append((name, cv_score))

        # Collect feature importance if available
        if hasattr(model, 'feature_importances_'):
            feature_importances[name] = model.feature_importances_

    except Exception as e:
        print(f"{name} failed: {e}")

# Residual Plot
plt.figure(figsize=(10, 6))
sns.histplot(df["residual"], bins=50, kde=True, color='orange')
plt.title("Residual Distribution (Linear Regression)")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.grid()
plt.show()

# Rankings
print("\n🏁 Final Model Rankings (CV R²):")
results_sorted = sorted(results, key=lambda x: x[1], reverse=True)
for i, (name, score) in enumerate(results_sorted, 1):
    print(f"{i}. {name}: {score:.4f}")

# SHAP for Best Model
print("\n🧠 SHAP Analysis for Best Model")
best_model_name, best_model_score = results_sorted[0]
best_model = models[best_model_name]

print(f"\n✅ Best Model: {best_model_name} (CV R²: {best_model_score:.4f})")

# Retrain best model on full data
best_model.fit(X, y)

# SHAP Explanation (tree-based only)
if hasattr(best_model, "predict") and ("Tree" in str(type(best_model)) or "Boost" in str(type(best_model))):
    explainer = shap.Explainer(best_model, X)
    shap_values = explainer(X)

    print("\n📊 SHAP Feature Importance:")
    shap.plots.beeswarm(shap_values, max_display=10)
    shap.plots.bar(shap_values, max_display=10)
else:
    print("⚠️ SHAP skipped: Only tree-based models (like XGBoost, CatBoost, etc.) are supported in this block.")

# Optional: Feature Importance Comparison for Tree Models
if feature_importances:
    plt.figure(figsize=(12, 6))
    for name, importances in feature_importances.items():
        sorted_idx = np.argsort(importances)[::-1][:5]
        plt.bar([f"{name}\n{X.columns[i]}" for i in sorted_idx], importances[sorted_idx], alpha=0.7)
    plt.title("Top 5 Feature Importances Across Tree-Based Models")
    plt.ylabel("Importance")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid()
    plt.show()

# Save modified dataframe
save_path = os.path.join(os.getcwd(), "medical_insurance_with_predictions.csv")
df.to_csv(save_path, index=False)
print(f"\n📁 Final dataset saved with predictions and residuals at: {save_path}")

