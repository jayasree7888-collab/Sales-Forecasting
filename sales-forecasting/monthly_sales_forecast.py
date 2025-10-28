# =============================================================
# Sales Forecasting (Monthly) - Full Workflow
# Author: You
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import joblib

# =============================================================
# STEP 1 — Load and Inspect Data
# =============================================================
FILENAME = "monthly_sales.csv"  # your CSV

df = pd.read_csv(FILENAME, parse_dates=["date"])
df = df.set_index("date").sort_index()

print("✅ Data loaded successfully!\n")
print(df.head(), "\n")

# Plot raw sales trend
plt.figure(figsize=(8,4))
plt.plot(df.index, df["sales"], marker="o")
plt.title("Monthly Sales Trend")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.grid(True)
plt.tight_layout()
plt.show()

# =============================================================
# STEP 2 — Feature Engineering
# =============================================================
df_fe = df.copy()

# Lag features (previous months' sales)
for lag in [1, 2, 3, 6, 12]:
    df_fe[f"sales_lag_{lag}"] = df_fe["sales"].shift(lag)

# Rolling averages (trend indicators)
df_fe["rolling_mean_3"] = df_fe["sales"].shift(1).rolling(window=3).mean()
df_fe["rolling_mean_6"] = df_fe["sales"].shift(1).rolling(window=6).mean()

# Date-based features
df_fe["month"] = df_fe.index.month
df_fe["year"] = df_fe.index.year
df_fe["quarter"] = df_fe.index.quarter

# Drop rows with missing lag values
df_fe = df_fe.dropna()

print("✅ Feature Engineering Done — columns now:", list(df_fe.columns), "\n")

# =============================================================
# STEP 3 — Train/Test Split
# =============================================================
TEST_SIZE = 4  # last 4 months for testing

X = df_fe.drop(columns=["sales"])
y = df_fe["sales"]

train_size = len(df_fe) - TEST_SIZE
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

print(f"Training data: {len(X_train)} months, Testing data: {len(X_test)} months\n")

# =============================================================
# STEP 4 — Model Training (XGBoost)
# =============================================================
model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42
)
model.fit(X_train, y_train)

# =============================================================
# STEP 5 — Evaluation
# =============================================================
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"✅ Model trained successfully! Test RMSE = {rmse:.2f}\n")

# Compare actual vs predicted
result = pd.DataFrame({"Actual": y_test, "Predicted": y_pred}, index=y_test.index)
print(result, "\n")

plt.figure(figsize=(8,4))
plt.plot(y_test.index, y_test.values, label="Actual", marker="o")
plt.plot(y_test.index, y_pred, label="Predicted", marker="x")
plt.title("Actual vs Predicted Sales")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# =============================================================
# STEP 6 — Save Model and Processed Data
# =============================================================
joblib.dump(model, "sales_forecast_model.pkl")
df_fe.to_csv("processed_sales.csv")
print("📁 Model and processed data saved successfully.\n")

# =============================================================
# STEP 7 — Forecast Next 3 Months (2020-01 → 2020-03)
# =============================================================


# months to predict
future_months = pd.date_range(df_fe.index[-1] + pd.offsets.MonthBegin(1),
                              periods=6, freq="MS")

# working list of all historical sales
all_sales = list(df_fe["sales"].astype(float).values)

# same lags as we trained
lags = [1, 2, 3, 6, 12]

future_predictions = []
future_index = []

for month in future_months:
    # Build feature dictionary for this new month
    feat = {}

    # --- Lag features ---
    for lag in lags:
        if len(all_sales) >= lag:
            feat[f"sales_lag_{lag}"] = all_sales[-lag]
        else:
            feat[f"sales_lag_{lag}"] = np.mean(all_sales)

    # --- Rolling features ---
    feat["rolling_mean_3"] = np.mean(all_sales[-3:]) if len(all_sales) >= 3 else np.mean(all_sales)
    feat["rolling_mean_6"] = np.mean(all_sales[-6:]) if len(all_sales) >= 6 else np.mean(all_sales)

    # --- Time-based features ---
    feat["month"] = month.month
    feat["year"] = month.year
    feat["quarter"] = (month.month - 1)//3 + 1

    # Create DataFrame with same columns as training features
    X_new = pd.DataFrame([feat], index=[month]).reindex(columns=X.columns)

    # Predict next month sales
    pred = float(model.predict(X_new)[0])
    future_predictions.append(pred)
    future_index.append(month)

    # Append new prediction to historical list for future lags
    all_sales.append(pred)

# Combine forecast results
forecast_df = pd.DataFrame({"Predicted_Sales": future_predictions}, index=future_index)

print("\n📈 Next 6-Month Forecast:\n")
print(forecast_df)

# --- Plot ---
plt.figure(figsize=(10,5))
plt.plot(df.index, df["sales"], label="Historical", marker="o")
plt.plot(forecast_df.index, forecast_df["Predicted_Sales"], label="Forecast", marker="x", color="orange")
plt.title("Sales Forecast for Next 6 Months")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()